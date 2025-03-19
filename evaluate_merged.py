"""Evaluation and visualization for dependency probes, supporting both multiclass and binary approaches."""

import pickle
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from model_utils import UDTransformer
from probing_merged import BinaryDependencyProbe, DependencyProbe, compute_loss
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from task import DependencyTask
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_probe(
    model: UDTransformer,
    probe: Union[DependencyProbe, Dict[str, BinaryDependencyProbe]],
    test_loader: DataLoader,
    device: torch.device,
    layer: int,
    train_toks: str = "tail",
    frequent_deps: Optional[list] = None,
    probe_type: str = "multiclass"
) -> Dict[str, float]:
    """Evaluate probe on test set.

    Args:
        model: Transformer model
        probe: Either a single DependencyProbe or a dictionary of BinaryDependencyProbes
        test_loader: DataLoader for test set
        layer: Layer to evaluate
        train_toks: Whether to use head or tail of dependency
        device: Device to use for computation
        frequent_deps: List of dependency types to evaluate on
        probe_type: Either "multiclass" or "binary"

    Returns:
        Dictionary containing metrics:
            - balanced_accuracy: Accuracy balanced across classes
            - per_class_accuracy: Accuracy for each dependency type
    """
    print(f"Evaluating {probe_type} probe(s) on device: {device}")

    # Get dependency list
    dep_list = frequent_deps if frequent_deps is not None else list(DependencyTask.dependency_table().keys())

    # Collect predictions and ground truth
    if probe_type == "multiclass":
        # Type check
        # if not isinstance(probe, DependencyProbe):
        #     raise TypeError(f"Expected DependencyProbe for multiclass evaluation, got {type(probe)}")

        # Move multiclass probe to device
        multiclass_probe = probe.to(device)
        multiclass_probe.eval()

        all_preds = []
        all_relations = []
        # all_tail_before_head = []
        all_probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Get model outputs
                preds_masked, relations_masked, scores_masked = compute_loss(
                    multiclass_probe, model, batch, layer, device,
                    train_toks=train_toks,
                    return_type="preds",
                    frequent_deps=frequent_deps,
                    binary=False
                )

                all_preds.append(preds_masked.cpu())
                all_relations.append(relations_masked.cpu())
                # all_tail_before_head.append(relations_masked.isnan().cpu())
                all_probs.append(torch.sigmoid(scores_masked).cpu())

        # Concatenate predictions and relations
        all_preds = np.concatenate(all_preds, axis=0)  # [n_tokens, n_deps]
        all_relations = np.concatenate(all_relations, axis=0)  # [n_tokens, n_deps]
        all_probs = np.concatenate(all_probs, axis=0)  # [n_tokens, n_deps]
        # all_tail_before_head = np.concatenate(all_tail_before_head, axis=0)  # [n_tokens, n_deps]
        print(all_preds.shape, all_relations.shape, all_probs.shape)

    elif probe_type == "binary":
        # Type check
        if not isinstance(probe, dict):
            raise TypeError(f"Expected Dict[str, BinaryDependencyProbe] for binary evaluation, got {type(probe)}")

        # Check that we have probes for all dependencies
        for dep in dep_list:
            if dep not in probe:
                raise ValueError(f"No probe found for dependency '{dep}'")

        # Initialize dictionaries for each dependency
        all_dep_preds = {}
        all_dep_relations = {}
        # all_dep_tail_before_head = {}
        all_dep_probs = {}

        # Evaluate each probe separately
        for dep_idx, dep_name in enumerate(dep_list):
            print(f"Evaluating probe for dependency: {dep_name}")

            # Move probe to device
            binary_probe = probe[dep_name].to(device)
            binary_probe.eval()

            dep_preds = []
            dep_relations = []
            # dep_tail_before_head = []
            dep_probs = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Evaluating {dep_name}"):
                    # Get model outputs
                    preds_masked, relations_masked, scores_masked = compute_loss(
                        binary_probe, model, batch, layer, device,
                        dep_idx=dep_idx,
                        train_toks=train_toks,
                        return_type="preds",
                        frequent_deps=frequent_deps,
                        binary=True
                    )

                    dep_preds.append(preds_masked.cpu().numpy())
                    dep_relations.append(relations_masked.cpu().numpy())
                    # dep_tail_before_head.append(relations_masked.isnan().cpu().numpy())
                    dep_probs.append(torch.sigmoid(scores_masked).cpu().numpy())
            # Store concatenated predictions and relations for this dependency
            all_dep_preds[dep_name] = np.concatenate(dep_preds, axis=0)
            all_dep_relations[dep_name] = np.concatenate(dep_relations, axis=0)
            # all_dep_tail_before_head[dep_name] = np.concatenate(dep_tail_before_head, axis=0)
            all_dep_probs[dep_name] = np.concatenate(dep_probs, axis=0)

        # Convert to format compatible with multiclass evaluation
        all_preds = np.column_stack([all_dep_preds[dep] for dep in dep_list])
        all_relations = np.column_stack([all_dep_relations[dep] for dep in dep_list])
        # all_tail_before_head = np.column_stack([all_dep_tail_before_head[dep] for dep in dep_list])
        all_probs = np.column_stack([all_dep_probs[dep] for dep in dep_list])
    else:
        raise ValueError(f"Invalid probe_type: {probe_type}. Must be 'multiclass' or 'binary'.")

    # Now calculate metrics (same for both approaches since data format is standardized)
    return calculate_metrics(all_preds, all_relations, all_probs, dep_list)


def calculate_metrics(preds: np.ndarray, relations: np.ndarray, probs: np.ndarray, dep_list: List[str]) -> Dict[str, np.ndarray]:
    """Calculate metrics from predictions and ground truth.

    Args:
        preds: Predictions array of shape [n_samples, n_deps]
        relations: Ground truth array of shape [n_samples, n_deps]
        probs: Probabilities array of shape [n_samples, n_deps]
        dep_list: List of dependency relation names

    Returns:
        Dictionary of metrics
    """

    # Overall metrics
    # argmaxes will unfairly penalize late classes in cases where multiple classes are correct
    # metrics["balanced_accuracy"] = balanced_accuracy_score(
    #     relations.argmax(axis=1),
    #     preds.argmax(axis=1)
    # )
    # metrics["macro_f1"] = f1_score(
    #     relations.argmax(axis=1),
    #     preds.argmax(axis=1),
    #     average='macro'
    # )
    # metrics["weighted_f1"] = f1_score(
    #     relations.argmax(axis=1),
    #     preds.argmax(axis=1),
    #     average='weighted'
    # )

    # Per-class metrics
    # metrics = {'probs': {dep: probs[:, idx] for idx, dep in enumerate(dep_list)}}
    metrics = {"probs": probs}
    per_class_acc = {}
    per_class_f1 = {}
    per_class_precision = {}
    per_class_recall = {}

    rels_bool = relations.astype(bool)
    preds_bool = preds.astype(bool)

    # for idx, dep in enumerate(dep_list):

        # per_class_acc[dep] = balanced_accuracy_score(rels_bool[:, idx].astype(int), preds_bool[:, idx].astype(int))
        # per_class_f1[dep] = f1_score(rels_bool.astype(int), bools.astype(int))

    metrics["accuracy"] = np.array([balanced_accuracy_score(rels_bool[:, idx].astype(int), preds_bool[:, idx].astype(int)) for idx in range(preds_bool.shape[1])])
    metrics["auroc"] = np.array([roc_auc_score(rels_bool[:, idx].astype(int), probs[:, idx]) for idx in range(preds_bool.shape[1])])
    print(metrics["auroc"])

    # per-class support
    metrics["support"] = np.sum(rels_bool, axis=0)

    tp = np.nansum(np.logical_and(preds_bool, rels_bool), axis=0)
    fp = np.nansum(np.logical_and(preds_bool, np.logical_not(rels_bool)), axis=0)
    fn = np.nansum(np.logical_and(np.logical_not(preds_bool), rels_bool), axis=0)

    metrics["precision"] = tp / (tp + fp)
    metrics["recall"] = tp / (tp + fn)
    metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])

    return metrics


def plot_layer_results(
    results: Dict[int, Dict[str, np.ndarray]],
    frequent_deps: List[str],
    save_path: Optional[str] = None
):
    """Plot precision/recall results.

    Args:
        results: Dictionary with precision/recall/f1 metrics in each layer
        save_path: Optional path to save plot
    """

    layers = sorted(list(results.keys()))
    plot_df = pd.DataFrame()

    for layer in layers:
        metrics = results[layer]
        layer_df = pd.DataFrame({k: v for k, v in metrics.items() if k != "probs"})  # probs is not a per-class metric
        layer_df["dependency"] = frequent_deps
        layer_df["layer"] = np.repeat(layer, len(frequent_deps))
        # layer_df["id"] = layer_df['dependency'].astype(str) + '_' + layer_df['layer'].astype(str)
        if plot_df.empty:
            plot_df = layer_df
        else:
            plot_df = pd.concat([plot_df, layer_df])

    plot_df = plot_df.reset_index(drop=True)
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))

    sns.scatterplot(
        data=plot_df,
        x='recall',
        y='precision',
        size='support',
        hue='dependency',
        ax=axs[0, 0]
    )
    axs[0, 0].set_title(f'Precision vs Recall for linear probes')
    axs[0, 0].set_xlim(-0.05, 1.05)
    axs[0, 0].set_ylim(-0.05, 1.05)
    sns.move_legend(axs[0, 0], "upper left", bbox_to_anchor=(-0.1, -0.1), ncol=5)

    # Create the heatmap
    stats = {"df": plot_df}
    for stat, ax in zip(['f1', 'auroc','precision', 'recall'], axs.flat[1:]):

        stat_matrix = plot_df.pivot(index='layer', columns='dependency', values=stat)

        sns.heatmap(
            data=stat_matrix,
            cmap='viridis',
            vmin=0,
            vmax=1,
            ax=ax,
            yticklabels=layers
        )
        ax.set_title(f'{stat.capitalize()} Score for linear probes')
        ax.set_ylabel('Layer')
        ax.set_xlabel('Dependency')

        stats[stat] = stat_matrix

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    return stats


# def plot_layer_results(results: Dict[int, Dict[str, Any]], baseline: Dict[str, Any], frequent_deps: Optional[list] = None, save_path: Optional[str] = None, probe_type: str = "multiclass"):
#     """Plot results across layers.

#     Args:
#         results: Dictionary mapping layer indices to metric dictionaries
#         baseline: Dictionary containing baseline metrics
#         frequent_deps: List of dependency types to evaluate on
#         save_path: Path to save plot to
#         probe_type: Type of probe used, either "multiclass" or "binary"
#     """
#     plt.figure(figsize=(20, 15))

#     # Get layers and metrics
#     layers = sorted(list(results.keys()))
#     layers_str = [str(layer) for layer in layers]  # Convert to strings for plotting

#     # Get list of dependency types
#     first_layer = layers[0]
#     dep_types = frequent_deps if frequent_deps is not None else list(results[first_layer].get("per_class_accuracy", {}).keys())

#     # Extract metrics across layers
#     balanced_accuracy = [np.mean(results[layer].get("accuracy", 0).values()) for layer in layers]

#     # balanced_accuracy = [results[layer].get("balanced_accuracy", 0) for layer in layers]
#     # macro_f1 = [results[layer].get("macro_f1", 0) for layer in layers]
#     # weighted_f1 = [results[layer].get("weighted_f1", 0) for layer in layers]

#     # Get baseline metrics
#     # baseline_macro_f1 = baseline.get("macro_f1", 0)
#     # baseline_weighted_f1 = baseline.get("weighted_f1", 0)

#     # Plot balanced accuracy
#     ax1 = plt.subplot(2, 2, 1)
#     ax1.plot(layers, balanced_accuracy, marker='o')
#     ax1.set_xlabel("Layer")
#     ax1.set_ylabel("Balanced Accuracy")
#     ax1.set_title(f"Balanced Accuracy by Layer ({probe_type.capitalize()} Probes)")
#     ax1.set_xticks(layers)
#     ax1.grid(True)

#     # Plot macro F1
#     ax2 = plt.subplot(2, 2, 2)
#     ax2.plot(layers, macro_f1, marker='o', label="Macro F1")
#     ax2.axhline(y=baseline_macro_f1, color='r', linestyle='--', label="Baseline Macro F1")
#     ax2.set_xlabel("Layer")
#     ax2.set_ylabel("Macro F1")
#     ax2.set_title(f"Macro F1 by Layer ({probe_type.capitalize()} Probes)")
#     ax2.set_xticks(layers)
#     ax2.legend()
#     ax2.grid(True)

#     # Extract per-class F1 scores across layers
#     per_class_f1 = np.zeros((len(layers), len(dep_types)))
#     for i, layer in enumerate(layers):
#         per_class_f1_dict = results[layer].get("per_class_f1", {})
#         for j, dep in enumerate(dep_types):
#             per_class_f1[i, j] = per_class_f1_dict.get(dep, 0)

#     # Plot per-class F1 heatmap
#     ax3 = plt.subplot(2, 2, 3)
#     sns.heatmap(
#         per_class_f1,
#         ax=ax3,
#         annot=True,
#         fmt=".2f",
#         xticklabels=dep_types,
#         yticklabels=layers_str,
#         cmap='viridis',
#         vmin=0,
#         vmax=1
#     )
#     ax3.set_xlabel("Dependency Type")
#     ax3.set_ylabel("Layer")
#     ax3.set_title(f"Per-Class F1 Score by Layer ({probe_type.capitalize()} Probes)")
#     plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.show()


# def compute_majority_baseline(test_loader: DataLoader, frequent_deps: Optional[list] = None) -> Dict[str, Any]:
#     """Compute baseline metrics assuming we always predict the most frequent class.

#     Args:
#         test_loader: DataLoader for test set
#         frequent_deps: List of dependency types to evaluate on

#     Returns:
#         Dictionary containing baseline metrics
#     """
#     # Get mapping of indices to relation names
#     dep_to_idx = DependencyTask.dependency_table()

#     # Get dependencies to evaluate
#     dep_list = frequent_deps if frequent_deps is not None else list(dep_to_idx.keys())

#     # Collect all relations
#     all_relations = []
#     for batch in test_loader:
#         relations = batch["relations"]
#         relation_mask = batch["relation_mask"]
#         relations_masked = relations[relation_mask]

#         # If using frequent_deps, only keep those columns
#         if frequent_deps is not None:
#             dep_indices = [dep_to_idx[dep] for dep in frequent_deps]
#             relations_masked = relations_masked[:, dep_indices]

#         all_relations.append(relations_masked)

#     # Concatenate all relations
#     all_relations = torch.cat(all_relations, dim=0).numpy()

#     # Find most frequent class
#     majority_class = all_relations.sum(axis=0).argmax()

#     # Create predictions array - all zeros except for majority class
#     all_preds = np.zeros_like(all_relations)
#     all_preds[:, majority_class] = 1

#     # Compute metrics
#     baseline_metrics: Dict[str, Any] = {
#         "macro_f1": f1_score(
#             all_relations.argmax(axis=1),
#             all_preds.argmax(axis=1),
#             average='macro'
#         ),
#         "weighted_f1": f1_score(
#             all_relations.argmax(axis=1),
#             all_preds.argmax(axis=1),
#             average='weighted'
#         )
#     }

#     # Also compute class frequencies for context
#     class_frequencies = {}
#     for i, dep in enumerate(dep_list):
#         class_frequencies[dep] = float(all_relations[:, i].mean())

#     baseline_metrics["class_frequencies"] = class_frequencies
#     baseline_metrics["majority_class"] = int(majority_class)

#     return baseline_metrics



# def plot_probe_pr_curves(
#     results: Dict[int, Dict[str, np.ndarray]],
#     frequent_deps: List[str],
#     save_path: Optional[str] = None
# ):
#     """Plot precision-recall curves for probes.

#     Args:
#         results: Dictionary mapping layer indices to evaluation results
#         frequent_deps: List of dependency types to evaluate
#         save_path: Optional path to save plot
#     """
#     plt.style.use('seaborn-v0_8')
#     n_deps = len(frequent_deps)
#     n_cols = min(7, n_deps)
#     n_rows = (n_deps + n_cols - 1) // n_cols

#     fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
#     if n_rows == 1:
#         axs = axs.reshape(1, -1)

#     # Create color map for layers
#     n_layers = len(results)
#     colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

#     for dep_idx, dep in enumerate(frequent_deps):
#         row = dep_idx // n_cols
#         col = dep_idx % n_cols
#         ax = axs[row, col]

#         max_f1 = 0
#         best_layer = None

#         for layer_idx, layer_results in results.items():
#             per_class = layer_results["per_class"][dep]
#             raw_preds = per_class["raw_preds"]
#             true_labels = per_class["true_labels"]

#             # Compute precision-recall curve
#             precision, recall, thresholds = precision_recall_curve(true_labels, raw_preds)
#             f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
#             max_layer_f1 = np.max(f1_scores)

#             if max_layer_f1 > max_f1:
#                 max_f1 = max_layer_f1
#                 best_layer = layer_idx

#             ax.plot(recall, precision, color=colors[layer_idx], alpha=0.5,
#                    label=f'Layer {layer_idx}' if col == 0 else None)

#         ax.set_title(f'{dep}\nBest F1={max_f1:.3f} (L{best_layer})')
#         ax.set_xlabel('Recall' if row == n_rows-1 else '')
#         ax.set_ylabel('Precision' if col == 0 else '')
#         ax.grid(True, alpha=0.3)

#     # Remove empty subplots
#     for idx in range(dep_idx + 1, n_rows * n_cols):
#         row = idx // n_cols
#         col = idx % n_cols
#         fig.delaxes(axs[row, col])

#     # Add legend
#     handles, labels = axs[0, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.show()


def main(
    test_loader: DataLoader,
    probes: Dict[int, Union[DependencyProbe, Dict[str, BinaryDependencyProbe]]],
    model: UDTransformer,
    train_toks: str = "tail",
    device: Optional[torch.device] = None,
    frequent_deps: Optional[list] = None,
    probe_type: str = "multiclass",
    model_name: str = None,
    results: Dict[int, Dict[str, Any]] | None = None
) -> Dict[int, Dict[str, Any]]:
    """Main evaluation function.

    Args:
        test_loader: DataLoader for test set
        probes: Dictionary mapping layers to probes (either DependencyProbe or Dict[str, BinaryDependencyProbe])
        model: Transformer model
        train_toks: Whether to use head or tail of dependency
        device: Device to use for computation
        frequent_deps: List of dependency types to evaluate on
        probe_type: Either "multiclass" or "binary"

    Returns:
        Dictionary mapping layer indices to metric dictionaries, including the baseline
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print(f"\nEvaluating {probe_type} probes on device: {device}")

    # First compute baseline
    # print("\nComputing majority class baseline...")
    # baseline = compute_majority_baseline(test_loader, frequent_deps)
    # print(f"Majority class baseline:")
    # print(f"  Macro F1: {baseline.get('macro_f1', 0):.3f}")
    # print(f"  Weighted F1: {baseline.get('weighted_f1', 0):.3f}")
    # print(f"\nClass frequencies:")

    dep_to_idx = DependencyTask.dependency_table()
    dep_list = frequent_deps if frequent_deps is not None else list(dep_to_idx.keys())

    # Safe access to class_frequencies dictionary
    # class_frequencies = baseline.get("class_frequencies", {})
    # for dep, freq in class_frequencies.items():
    #     if isinstance(freq, (int, float)) and freq > 0.01:  # Only show classes with >1% frequency
    #         print(f"  {dep}: {freq:.3f}")

    # # Safe access to majority_class
    # majority_class_idx = baseline.get("majority_class", 0)
    # if isinstance(majority_class_idx, int) and 0 <= majority_class_idx < len(dep_list):
    #     print(f"\nMajority class: {dep_list[majority_class_idx]}")

    save_path = f"data/evals/{model_name}_eval_results_layers_{min(probes.keys())}-{max(probes.keys())}.pkl"

    # Then evaluate probes
    if results is None:
        results: Dict[int, Dict[str, Any]] = {}

    for layer, probe in probes.items():
        if layer not in results.keys():
            print(f"\nEvaluating layer {layer}")
            metrics = evaluate_probe(
                model, probe, test_loader, device, layer,
                train_toks=train_toks,
                frequent_deps=dep_list,
                probe_type=probe_type
            )
            results[layer] = metrics
            # print(f"Balanced accuracy: {metrics.get('balanced_accuracy', 0):.3f}")
            # print(f"Improvement over majority baseline:")
            # print(f"  Macro F1: {metrics.get('macro_f1', 0) - baseline.get('macro_f1', 0):.3f}")
            # print(f"  Weighted F1: {metrics.get('weighted_f1', 0) - baseline.get('weighted_f1', 0):.3f}")

    print(f"Saving results to {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    # Plot results
    stats = plot_layer_results(
        results,
        dep_list,
        save_path=f"figures/evals/{probe_type}_eval_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps) if frequent_deps else len(dep_to_idx)}.png",
    )

    # save_path = f"figures/evals/eval_{probe_type}_{train_toks}_pr_curves_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps)}.png"
    # print(f"Saving PR curves to {save_path}")
    # plot_probe_pr_curves(results, frequent_deps, save_path=save_path)

    # Save results with baseline
    # results_with_baseline: Dict[str, Any] = {**{str(k): v for k, v in results.items()}, "baseline": baseline}


    return results
