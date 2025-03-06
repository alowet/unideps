"""Analyze sparsity of SAE latent activations."""

import gc
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
from analyze_sae import cleanup_gpu_memory

sys.path.append("../matryoshka_sae")
import os

import kaleido
from model_utils import UDTransformer
from sae import GlobalBatchTopKMatryoshkaSAE
from sae_lens import SAE, ActivationsStore, HookedSAETransformer
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def compute_latent_activations(
    model: UDTransformer,
    sae: SAE,
    batch: Dict,
    layer: int,
    device: torch.device,
) -> Dict:
    """Compute SAE latent activations for a batch.

    Args:
        model: Transformer model
        sae: Sparse autoencoder
        batch: Batch of data
        layer: Layer to analyze
        device: Device to use
    Returns:
        latent_counts: Number of non-zero activations in each latent dimension [d_sae]
        n_tokens: Number of tokens processed
    """
    # Get activations
    activations = model.get_activations(batch, layer)  # [batch, seq_len, d_model]

    # Mask out padding tokens
    activations = activations[batch["relation_mask"]]  # [n_tokens, d_model]

    relations = batch["relations"]  # [batch, seq_len, num_relations]
    relations = relations[batch["relation_mask"]].to(device)  # [n_tokens, num_relations] boolean array
    # print(relations.dtype)
    print(relations.shape)
    # print(relations[:5, :5])

    # Compute SAE latent activations
    with torch.no_grad():
        latent_activations = sae.encode(activations).squeeze()  # [n_tokens, d_sae]

    latent_fired = torch.tensor(latent_activations > 0, dtype=torch.float32)
    latent_counts = torch.sum(latent_fired, dim=0)  # [d_sae]
    n_tokens = latent_activations.shape[0]
    # print(latent_activations.shape)

    # For each relation and feature, compute firing fraction when relation is True and when it is False
    relation_means = torch.einsum('ts,tr->sr', latent_fired, relations)  # [d_sae, num_relations]
    relation_totals = relations.sum(dim=0)

    nonrelation_means = torch.einsum('ts,tr->sr', latent_fired, torch.abs(relations - 1))
    # [d_sae, num_relations]
    nonrelation_totals = (torch.abs(relations - 1)).sum(dim=0)

    # relation_diffs = relation_means - nonrelation_means  # [d_sae, num_relations]
    # print(relation_diffs.shape)

    return {
        'latent_counts': latent_counts,
        'n_tokens': n_tokens,
        'relation_means': relation_means,
        'relation_totals': relation_totals,
        'nonrelation_means': nonrelation_means,
        'nonrelation_totals': nonrelation_totals
    }

@torch.no_grad()
def compute_activation_sparsity(
    model: HookedSAETransformer,
    sae: SAE,
    act_store: ActivationsStore,
    total_batches: int = 50,
):
    """
    Displays the activation histogram for a particular latent, computed across `total_batches` batches from `act_store`.
    """
    sae_acts_post_hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    print(sae_acts_post_hook_name)
    n_active = np.zeros(sae.cfg.d_sae)
    n_total = 0

    for i in tqdm(range(total_batches), desc="Computing activation sparsity"):

        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache_with_saes(
            tokens,
            saes=[sae],
            stop_at_layer=sae.cfg.hook_layer + 1,
            names_filter=[sae_acts_post_hook_name],
        )
        acts = cache[sae_acts_post_hook_name].view(-1, sae.cfg.d_sae)  # [batch * seq_len, d_sae]
        n_active += (acts > 0).sum(dim=0).cpu().numpy()
        n_total += acts.shape[0]

        del acts, cache, tokens
        torch.cuda.empty_cache()

    frac_active = n_active / n_total

    return frac_active

@torch.no_grad()
def compute_matryoshka_activation_sparsity(
    model: HookedSAETransformer,
    sae: GlobalBatchTopKMatryoshkaSAE,
    act_store: ActivationsStore,
    cfg: Dict,
    total_batches: int = 50,
):
    """
    Displays the activation histogram for a particular latent, computed across `total_batches` batches from `act_store`.
    """
    n_active = np.zeros(cfg["dict_size"])
    n_total = 0

    for i in tqdm(range(total_batches), desc="Computing activation sparsity"):

        tokens = act_store.get_batch_tokens()
        _, cache = model.run_with_cache(
            tokens,
            stop_at_layer=cfg["layer"] + 1,
            names_filter=[cfg["hook_point"]],
        )
        acts = sae.encode(cache[cfg["hook_point"]]).view(-1, cfg["dict_size"])  # [batch * seq_len, d_sae]
        n_active += (acts > 0).sum(dim=0).cpu().numpy()
        n_total += acts.shape[0]

        del acts, cache, tokens
        torch.cuda.empty_cache()

    frac_active = n_active / n_total

    return frac_active

def plot_activation_density(frac_active: np.ndarray, which_sae: str, layer: int, width: int):
    """Plot the activation density for a given latent."""
    fig = px.histogram(
            frac_active,
            nbins=50,
            title="ACTIVATIONS DENSITY",
            width=800,
            template="ggplot2",
            color_discrete_sequence=["darkorange"],
            log_y=True,
        ).update_layout(bargap=0.02, showlegend=False)
    os.makedirs(f"figures/sparsity/{which_sae}", exist_ok=True)
    fig.write_image(f"figures/sparsity/{which_sae}/activation_density_hist_layer_{layer}_width_{width}k.png", engine="kaleido")
    fig.show()

def analyze_sparsity(
    model: UDTransformer,
    sae: SAE,
    act_store: ActivationsStore,
    layer: int,
    device: torch.device,
    save_path: str = None
):
    """Analyze sparsity of SAE latent activations.

    Args:
        model: Transformer model
        data_loader: Data loader
        layer: Layer to analyze
        device: Device to use
        sae_width: Width of SAE in thousands
        save_path: Path to save plot
    """
    print(f"\nAnalyzing SAE sparsity for layer {layer}...")

    compute_activation_sparsity(model.model, sae, act_store)


    # all_latent_counts = np.zeros(sae.cfg.d_sae)
    # all_n_tokens = 0
    # all_relation_means = []
    # n_relations = 53  # TODO: make this dynamic
    # all_relation_totals = np.zeros(n_relations)
    # all_nonrelation_means = []
    # all_nonrelation_totals = np.zeros(n_relations)

    # for batch in tqdm(data_loader):
    #     out = compute_latent_activations(model, sae, batch, layer, device)
    #     all_latent_counts += out['latent_counts'].cpu().numpy()
    #     all_n_tokens += out['n_tokens']
    #     all_relation_means.append(out['relation_means'].cpu().numpy())
    #     all_relation_totals += out['relation_totals'].cpu().numpy()
    #     all_nonrelation_means.append(out['nonrelation_means'].cpu().numpy())
    #     all_nonrelation_totals += out['nonrelation_totals'].cpu().numpy()
    # # Compute fraction of tokens that activate each feature
    # activation_fractions = all_latent_counts / all_n_tokens  # [d_sae]
    # print(np.flatnonzero(activation_fractions > 0.1))
    # log_act_fractions = np.log(activation_fractions)

    # all_relation_diffs = np.sum(np.stack(all_relation_means, axis=0), axis=0) / all_relation_totals - np.sum(np.stack(all_nonrelation_means, axis=0), axis=0) / all_nonrelation_totals  # [d_sae, num_relations]
    # print(all_relation_diffs.shape)

    # # top n latents for each relation
    # top_n = 5
    # top_latents = np.argsort(all_relation_diffs, axis=0)[-top_n:]  # shape [top_n, num_relations]
    # print(top_latents)

    # # for each relation, compute the rank correlation between activation_fraction and all_relation_diffs
    # correlations = []
    # for i in range(all_relation_diffs.shape[1]):
    #     correlations.append(np.corrcoef(log_act_fractions, all_relation_diffs[:, i])[0, 1])
    #     if i == 7:
    #         print(correlations[i])
    #         print(log_act_fractions)
    #         print(all_relation_diffs[:, i])
    # correlations = np.array(correlations)
    # print(correlations.shape)
    # print(correlations)

    # # plot the correlations for each relation
    # plt.figure(figsize=(10, 6))
    # plt.scatter(range(len(correlations)), correlations)
    # plt.xlabel('Relation')
    # plt.ylabel('Correlation')
    # plt.title(f'Correlation between activation fraction and relation diffs (Layer {layer})')
    # plt.show()

    # # # Plot histogram of relation_diffs
    # # plt.figure(figsize=(10, 6))
    # # plt.hist(all_relation_diffs)
    # # plt.xlabel('Relation diff')
    # # plt.ylabel('Number of features')
    # # plt.title(f'SAE Relation Diff Histogram (Layer {layer})')
    # # plt.show()

    # # for each relation, plot scatter plot of activation_fractions vs relation_diffs
    # fig, axs = plt.subplots(8, 7, figsize=(10, 6))
    # fig2, axs2 = plt.subplots(8, 7, figsize=(10, 6))
    # for i in range(all_relation_diffs.shape[1]):
    #     axs.flat[i].scatter(log_act_fractions, all_relation_diffs[:, i])
    #     axs2.flat[i].hist(all_relation_diffs[:, i])
    # # axs.flat[i].xlabel('Activation fraction')
    # # axs.flat[i].ylabel('Relation diff')
    # # axs.flat[i].title(f'SAE Activation Fraction vs Relation Diff (Layer {layer})')
    # plt.show()

    # # Plot histogram of activation fractions
    # plt.figure(figsize=(10, 6))
    # plt.hist(activation_fractions, bins=100)
    # plt.xlabel('Fraction of tokens')
    # plt.ylabel('Number of features')
    # plt.title(f'SAE Feature Activation Rates (Layer {layer})')
    # plt.semilogy()

    # if save_path:
    #     plt.savefig(save_path)
    # plt.show()

    # # Print summary statistics
    # print(f"\nSummary Statistics:")
    # print(f"Mean activation rate: {activation_fractions.mean():.4f}")
    # print(f"Median activation rate: {np.median(activation_fractions):.4f}")
    # print(f"Std activation rate: {activation_fractions.std():.4f}")
    # print(f"Max activation rate: {activation_fractions.max():.4f}")
    # print(f"Min activation rate: {activation_fractions.min():.4f}")

    # print(f"Number of features with activation rate > 10% of tokens: {np.sum(activation_fractions > 0.1)}")

    cleanup_gpu_memory()

def print_gpu_memory():
    """Print memory usage for all tensors on GPU."""
    import gc

    import torch

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Get all objects
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"{type(obj).__name__:<20} | {list(obj.shape):<20} | {obj.dtype} | {obj.device} | {obj.element_size() * obj.nelement() / 1024 / 1024:.2f} MB")
        except:
            pass

    # Print total memory usage
    print("\nTotal GPU memory:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
