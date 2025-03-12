"""Evaluate precision/recall of top SAE latents for dependency parsing."""

import gc
import os
import pickle
import sys
import time
from itertools import repeat
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sae_lens import SAE
from task import DependencyTask
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("../matryoshka_sae")
from sae import GlobalBatchTopKMatryoshkaSAE
from utils import load_sae_from_wandb


def get_top_latents_per_dep(
    similarities: np.ndarray,  # [n_layers, n_deps, d_sae]
    frequent_deps: List[str],
    n_top: int = 10
) -> Dict[str, List[Tuple[int, int]]]:
    """Get the top n latents for each dependency type.

    Args:
        similarities: Similarities array [n_layers, n_deps, d_sae]
        frequent_deps: List of dependency types to evaluate
        n_top: Number of top latents to return per dependency

    Returns:
        Dictionary mapping dependency types to list of (layer, latent_idx) tuples
    """
    n_layers, n_deps, d_sae = similarities.shape
    top_latents = {}

    for dep_idx, dep_type in enumerate(frequent_deps):
        # Flatten across layers
        dep_sims = similarities[:, dep_idx, :]
        flat_indices = np.argsort(dep_sims.flatten())[-n_top:]

        # Convert flat indices back to (layer, latent) coordinates
        layers, latents = np.unravel_index(flat_indices, dep_sims.shape)
        top_latents[dep_type] = list(zip(layers, latents))

    return top_latents

def evaluate_latent_predictions(
    ud_model,
    test_loader: DataLoader,
    top_latents: Dict[str, List[Tuple[int, int]]],
    frequent_deps: List[str],
    which_sae: str,
    device: torch.device,
    threshold: float = 0,
    start_layer: int = 4,
    stop_layer: int = 9
) -> pd.DataFrame:
    """Evaluate precision/recall of top latents on test set.

    Args:
        ud_model: Universal Dependencies model
        test_loader: DataLoader for test set
        top_latents: Dictionary mapping dep types to (layer, latent) pairs
        frequent_deps: List of dependency types to evaluate
        which_sae: Which SAE to use
        device: Device to run evaluation on
        threshold: Activation threshold for considering a latent "active"
        start_layer: Start layer to evaluate
        stop_layer: Stop layer to evaluate

    Returns:
        DataFrame with precision/recall metrics
    """
    results = []
    entity = "adam-lowet-harvard-university"
    project = "batch-topk-matryoshka"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    width = 16

    dep_table = DependencyTask.dependency_table()
    dep_indices = [dep_table[dep] for dep in frequent_deps]

    # Initialize statistics tracking
    stats_by_dep = {}
    for dep_type in frequent_deps:
        stats_by_dep[dep_type] = {}
        for i_latent, (layer, latent) in enumerate(top_latents[dep_type]):
            if start_layer <= layer < stop_layer:
                stats_by_dep[dep_type][(layer, latent)] = {
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "support": 0,
                    "rank": i_latent
                }

    with torch.no_grad():
        # Process each layer that contains latents we care about
        for layer in tqdm(range(start_layer, stop_layer), desc="Processing layers"):
            # Check if we have any latents to evaluate in this layer
            layer_latents = set()
            for dep_type, latent_list in top_latents.items():
                layer_latents.update(latent for l, latent in latent_list if l == layer)

            if not layer_latents:
                continue

            # Load appropriate SAE
            if which_sae == "gemma_scope":
                sae_id = f"layer_{layer}/width_{width}k/canonical"
                sae = SAE.from_pretrained(sae_release, sae_id, device=str(device))[0]
            elif which_sae == "matryoshka":
                sae_id = f"gemma-2-2b_blocks.{layer + 1}.hook_resid_pre_36864_global-matryoshka-topk_32_0.0003_122069"
                sae, cfg = load_sae_from_wandb(f"{entity}/{project}/{sae_id}:latest", GlobalBatchTopKMatryoshkaSAE)

            # Process each batch
            for batch in tqdm(test_loader, desc=f"Layer {layer} batches"):
                # Get ground truth relations
                relations = batch["relations"]  # [batch, seq_len, num_relations]
                relations = relations[batch["relation_mask"]].to(device)  # [n_tokens, num_relations]

                # Get activations
                activations = ud_model.get_activations(batch, layer)  # [batch, seq_len, d_model]
                activations = activations[batch["relation_mask"]]  # [n_tokens, d_model]

                # Get SAE latent activations for this batch
                latent_acts = sae.encode(activations).squeeze()  # [n_tokens, d_sae]

                # Update statistics for each dependency type
                for dep_idx, dep_type in enumerate(frequent_deps):
                    # Get ground truth for this dependency
                    dep_mask = relations[:, dep_indices[dep_idx]].bool()
                    dep_support = dep_mask.sum().item()

                    # Check each latent for this dependency
                    for layer_i, latent_i in top_latents[dep_type]:
                        if layer_i == layer:  # Only process latents for current layer
                            stats = stats_by_dep[dep_type][(layer_i, latent_i)]

                            # Get predictions for this latent
                            pred_mask = latent_acts[:, latent_i] > threshold

                            # Update statistics
                            stats["TP"] += ((pred_mask & dep_mask) == 1).sum().item()
                            stats["FP"] += ((pred_mask & ~dep_mask) == 1).sum().item()
                            stats["FN"] += ((~pred_mask & dep_mask) == 1).sum().item()
                            stats["support"] += dep_support

            # Cleanup
            del sae
            torch.cuda.empty_cache()

        # Convert statistics to DataFrame
        for dep_type, dep_stats in stats_by_dep.items():
            for (layer, latent), stats in dep_stats.items():
                tp = stats["TP"]
                fp = stats["FP"]
                fn = stats["FN"]

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                results.append({
                    'dependency': dep_type,
                    'latent': latent,
                    'rank': stats["rank"],
                    'layer': layer,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': stats["support"],
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn
                })

    return pd.DataFrame(results)

def plot_precision_recall(
    results_df: pd.DataFrame,
    top_latents: Dict[str, List[Tuple[int, int]]],
    save_path: Optional[str] = None
):
    """Plot precision/recall results.

    Args:
        results_df: DataFrame with precision/recall metrics
        top_latents: Dictionary mapping dependency types to list of (layer, latent) tuples
        save_path: Optional path to save plot
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    print(np.unique(results_df["dependency"]))

    plot_df = results_df[results_df["support"] > 0]
    print(np.unique(plot_df["dependency"]))
    plot_df["id"] = plot_df['dependency'].astype(str) + '_' + plot_df['layer'].astype(str) + '_' + plot_df['latent'].astype(str)

    top_latents_ids = ['_'.join([dep_type, str(layer), str(latent)]) for dep_type, latents in top_latents.items() for layer, latent in latents]
    top_df = plot_df[np.logical_and(plot_df["setname"] == "test", plot_df["id"].isin(top_latents_ids))]

    # Precision/Recall scatter plot
    sns.scatterplot(
        data=top_df,
        x='recall',
        y='precision',
        size='support',
        hue='layer',
        ax=axs[0, 0]
    )
    axs[0, 0].set_title('Precision vs Recall by Dependency Type')

    # F1 score bar plot
    train_df = plot_df[plot_df["setname"] == "train"]
    test_df = plot_df[plot_df["setname"] == "test"]

    # Create the heatmap
    stats = {}
    for stat, ax in zip(['f1', 'precision', 'recall'], axs.flat[1:]):
        # First, find the indices of the rows with max F1 for each dependency and layer in the training set
        # We use idxmax() to get the index of the maximum value
        idx_max_train = train_df.groupby(['dependency', 'layer'], observed=True)[stat].idxmax()

        # Then use these indices to get the corresponding rows, which will include all columns
        max_train_rows = train_df.loc[idx_max_train]

        # Now we can get the IDs of these maximum F1 latents
        # max_train_ids = max_train_rows['id'].values

        # Filter test data to only include these IDs
        max_test_stat = test_df[test_df["id"].isin(max_train_rows['id'])].groupby(['dependency', 'layer'], observed=True)[stat].max().reset_index()

        stat_matrix = max_test_stat.pivot(index='layer', columns='dependency', values=stat)

        sns.heatmap(
            data=stat_matrix,
            cmap='viridis',
            vmin=0,
            vmax=1,
            ax=ax,
            yticklabels=np.arange(25, 25 - stat_matrix.shape[0], -1)[::-1]
        )
        ax.set_title(f'{stat.capitalize()} Score for Max Latent')
        ax.set_ylabel('Layer')
        ax.set_xlabel('Dependency')
        stats[stat] = stat_matrix

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    return stats

def main(
    ud_model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    similarities: np.ndarray,
    frequent_deps: List[str],
    n_top: int = 2,
    which_sae: str = "gemma_scope",
    device: Optional[torch.device] = None,
    model_name: str = "trail_tail_min_20",
    start_layer: int = 4,
    stop_layer: int = 9
):
    """Main function to evaluate top latents.

    Args:
        ud_model: Universal Dependencies model
        train_loader: DataLoader for train set
        test_loader: DataLoader for test set
        similarities: Similarities array [n_layers, n_deps, d_sae]
        frequent_deps: List of dependency types to evaluate
        n_top: Number of top latents to use per dependency
        which_sae: Which SAE to use
        device: Device to run evaluation on
        model_name: Name of the model
        start_layer: Start layer to evaluate
        stop_layer: Stop layer to evaluate
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get top latents for each dependency type
    top_latents = get_top_latents_per_dep(similarities, frequent_deps, n_top)

    # Evaluate on test set
    results_df = evaluate_all_latents(
            ud_model=ud_model,
            train_loader=train_loader,
            test_loader=test_loader,
            frequent_deps=frequent_deps,
            which_sae=which_sae,
            device=device,
            start_layer=start_layer,
            stop_layer=stop_layer
        )

    # save the results
    fpath = f"data/sae/{which_sae}/top_latents_evaluation_{model_name}.parquet"
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    results_df.to_parquet(fpath)

    stats = plot_precision_recall(results_df, top_latents=top_latents, save_path=f"figures/sae/{which_sae}/top_{n_top}_latents_evaluation_{model_name}.png")
    with open(f"data/sae/{which_sae}/stats_{model_name}.pkl", "wb") as f:
        pickle.dump(stats, f)

    return results_df, stats


def evaluate_all_latents(
    ud_model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    frequent_deps: List[str],
    which_sae: str,
    device: torch.device,
    start_layer: int = 4,
    stop_layer: int = 9,
    learning_rate: float = 1e-3,
    num_epochs: int = 5,
    batch_size: int = 1024
) -> pd.DataFrame:
    """Evaluate precision/recall of ALL latents using parallel probe training.

    Args:
        ud_model: Universal Dependencies model
        train_loader: DataLoader for train set
        test_loader: DataLoader for test set
        frequent_deps: List of dependency types to evaluate
        which_sae: Which SAE to use
        device: Device to run evaluation on
        start_layer: Start layer to evaluate
        stop_layer: Stop layer to evaluate
        learning_rate: Learning rate for probe training
        num_epochs: Number of epochs for probe training
        batch_size: Batch size for probe training
    """

    columns = ['setname', 'dependency', 'layer', 'latent', 'precision', 'recall', 'f1', 'support', 'true_positives', 'false_positives', 'false_negatives']
    results = pd.DataFrame(columns=columns)
    entity = "adam-lowet-harvard-university"
    project = "batch-topk-matryoshka"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    width = 16

    dep_table = DependencyTask.dependency_table()
    dep_indices = [dep_table[dep] for dep in frequent_deps]

    # class ParallelProbe(nn.Module):
    #     def __init__(self, d_sae):
    #         super().__init__()
    #         # Initialize with very small random values to avoid numerical issues
    #         self.weights = nn.Parameter(torch.randn(d_sae) * 0.001)
    #         self.biases = nn.Parameter(torch.zeros(d_sae))

    #     def forward(self, x):  # x: [batch_size, d_sae]
    #         # Parallel logistic regression for all latents
    #         # Returns: [batch_size, d_sae]
    #         logits = x * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
    #         return torch.sigmoid(logits)

    #     def compute_loss(self, x, y):
    #         """Compute BCE loss using the logits directly to avoid numerical issues.

    #         Args:
    #             x: Input tensor [batch_size, d_sae]
    #             y: Target tensor [batch_size]

    #         Returns:
    #             loss: Total loss
    #         """
    #         # Check for NaN or Inf in inputs
    #         if torch.isnan(x).any() or torch.isinf(x).any():
    #             print(f"Warning: NaN or Inf in input activations")
    #             # Replace NaN/Inf with zeros
    #             x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    #         # Check for extreme values in activations
    #         max_val = x.abs().max().item()
    #         if max_val > 100:
    #             print(f"Warning: Large activation values detected: {max_val}")
    #             # Scale down if values are too large
    #             x = x / (max_val / 10.0)

    #         # Compute logits directly with gradient clipping
    #         logits = x * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)  # [batch_size, d_sae]
    #         logits = logits.unsqueeze(1)  # [batch_size, 1, d_sae], where 1 is the class dimension
    #         logits = torch.clamp(logits, min=-10.0, max=10.0)  # Prevent extreme values

    #         n_pos = y.sum().item()
    #         n_total = y.shape[0]
    #         n_neg = n_total - n_pos

    #         # Expand y to match logits dimensions [batch_size, 1, d_sae]
    #         y = y.unsqueeze(1).unsqueeze(2).expand(logits.size(0), logits.size(1), logits.size(2))

    #         # Use BCEWithLogitsLoss which is more numerically stable
    #         loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor([[[n_neg/n_pos]]]).to(device), reduction='none')

    #         # Check for NaN in loss
    #         if torch.isnan(loss).any():
    #             print(f"Warning: NaN in loss computation")
    #             loss = torch.nan_to_num(loss, nan=0.0)

    #         # Average over batch dimension first, then sum over latents
    #         return loss.mean(dim=0).sum()

    # Process each layer
    for layer in tqdm(range(start_layer, stop_layer), desc="Processing layers"):
        # Load appropriate SAE
        if which_sae == "gemma_scope":
            sae_id = f"layer_{layer}/width_{width}k/canonical"
            sae = SAE.from_pretrained(sae_release, sae_id, device=str(device))[0]
        elif which_sae == "matryoshka":
            sae_id = f"gemma-2-2b_blocks.{layer + 1}.hook_resid_pre_36864_global-matryoshka-topk_32_0.0003_122069"
            sae, cfg = load_sae_from_wandb(f"{entity}/{project}/{sae_id}:latest", GlobalBatchTopKMatryoshkaSAE)

        d_sae = sae.W_enc.shape[1]  # Get number of latents

        # Initialize parallel probes for each dependency type
        # probes = {dep_type: ParallelProbe(d_sae).to(device) for dep_type in frequent_deps}
        # optimizers = {dep_type: torch.optim.Adam(probes[dep_type].parameters(), lr=learning_rate) for dep_type in frequent_deps}


        for setname, loader, n_epochs in [("train", train_loader, num_epochs), ("test", test_loader, 1)]:

            # Initialize counters for both train and test sets
            dep_counters = {
                dep_type: {
                    "TP": np.zeros(d_sae, dtype=np.int16),
                    "FP": np.zeros(d_sae, dtype=np.int16),
                    "FN": np.zeros(d_sae, dtype=np.int16),
                        "total": 0
                    } for dep_type in frequent_deps
                }

            # # Process each epoch
            # for epoch in range(n_epochs):

            #     total_losses = {dep_type: 0 for dep_type in frequent_deps}

            # Process each batch
            for batch_idx, batch in tqdm(enumerate(loader), desc=f"Layer {layer} batches"):

                # Get ground truth relations
                relations = batch["relations"]  # [batch, seq_len, num_relations]

                relations = relations[batch["relation_mask"]].to(device)  # [n_tokens, num_relations]
                # Get activations
                activations = ud_model.get_activations(batch, layer)  # [batch, seq_len, d_model]
                activations = activations[batch["relation_mask"]]  # [n_tokens, d_model]
                # Get SAE latent activations for this batch
                latent_acts = sae.encode(activations).squeeze().detach()  # [n_tokens, d_sae]
                for dep_idx, dep_type in enumerate(frequent_deps):
                    # if setname == "train":
                    #     train_labels = relations[:, dep_idx].float()  # [n_tokens]
                    #     if train_labels.sum() == 0:
                    #         continue

                    #     probes[dep_type].train()
                    #     optimizers[dep_type].zero_grad()
                    #     loss = probes[dep_type].compute_loss(latent_acts, train_labels)
                    #     loss.backward()
                    #     optimizers[dep_type].step()
                    #     total_losses[dep_type] += loss.item()

                    #     if batch_idx == len(loader) - 1:
                    #         print(f"Layer {layer}, {dep_type}, epoch {epoch}: loss = {loss.item():.4f}")

                    # if epoch == num_epochs - 1:

                    # Get ground truth for this dependency
                    dep_mask = relations[:, dep_indices[dep_idx]].bool()  # [n_tokens]

                    # batch_acts = probes[dep_type](latent_acts) > 0.5  # [n_tokens, d_sae]
                    batch_acts = latent_acts > 0  # [n_tokens, d_sae]

                    # Compute TP, FP, FN for each latent in the batch
                    # For each latent, we have a binary prediction for each token
                    tp = torch.sum(batch_acts & dep_mask.unsqueeze(1), dim=0).cpu().numpy()
                    fp = torch.sum(batch_acts & ~dep_mask.unsqueeze(1), dim=0).cpu().numpy()
                    fn = torch.sum(~batch_acts & dep_mask.unsqueeze(1), dim=0).cpu().numpy()

                    # Update counters
                    dep_counters[dep_type]["TP"] += tp
                    dep_counters[dep_type]["FP"] += fp
                    dep_counters[dep_type]["FN"] += fn
                    dep_counters[dep_type]["total"] += dep_mask.sum().item()

        #     # Compute metrics for all latents in this layer
        #     layer_results = compute_metrics_for_layer(dep_counters, layer, frequent_deps)
        #     results = pd.concat([results, layer_results], axis=0, ignore_index=True)
        #     gc.collect()
        #     torch.cuda.empty_cache()

        # del sae, dep_counters
            if dep_counters["ccomp"]["total"] == 0:
                raise Exception(f"No ccomp dependencies found for layer {layer}")

            layer_results = pd.DataFrame(columns=columns)
            for dep_type, counters in dep_counters.items():
                tp = counters["TP"]
                fp = counters["FP"]
                fn = counters["FN"]
                total = np.int16(counters["total"])

                # Compute precision, recall, F1
                precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float32), where=(tp + fp) > 0)
                recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float32), where=(tp + fn) > 0)
                f1 = np.divide(2 * precision * recall, precision + recall,
                            out=np.zeros_like(precision, dtype=np.float32),
                            where=(precision + recall) > 0)

                # Add to results
                tmp = pd.DataFrame(data=dict(zip(columns, [repeat(setname, d_sae), repeat(dep_type, d_sae), repeat(layer, d_sae), np.arange(d_sae), precision, recall, f1, repeat(total, d_sae), tp, fp, fn])))
                if layer_results.empty:
                    layer_results = tmp
                else:
                    layer_results = pd.concat([layer_results, tmp], axis=0, ignore_index=True)

            del dep_counters
            gc.collect()
            torch.cuda.empty_cache()
            if results.empty:
                results = layer_results
            else:
                results = pd.concat([results, layer_results], axis=0, ignore_index=True)
        # Cleanup
        # del sae, probes, optimizers, train_labels
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    results["setname"] = results["setname"].astype("category")
    results["dependency"] = results["dependency"].astype("category")

    return results
