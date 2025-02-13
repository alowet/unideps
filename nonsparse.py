"""Analyze sparsity of SAE latent activations."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from analyze_sae import cleanup_gpu_memory
from model_utils import UDTransformer
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from torch.utils.data import DataLoader
from tqdm import tqdm


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

    # Compute SAE latent activations
    with torch.no_grad():
        latent_activations = sae.encode(activations).squeeze()  # [n_tokens, d_sae]

    latent_counts = torch.sum(latent_activations > 0, dim=0)  # [d_sae]
    n_tokens = latent_activations.shape[0]

    return {
        'latent_counts': latent_counts,
        'n_tokens': n_tokens
    }


def analyze_sparsity(
    model: UDTransformer,
    data_loader: DataLoader,
    layer: int,
    device: torch.device,
    sae_width: int = 16,
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

    # Load pretrained SAE
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = f"layer_{layer}/width_{sae_width}k/canonical"
    sae = SAE.from_pretrained(sae_release, sae_id, device=str(device))[0]

    all_latent_counts = np.zeros(sae.cfg.d_sae)
    all_n_tokens = 0

    for batch in tqdm(data_loader):
        out = compute_latent_activations(model, sae, batch, layer, device)
        all_latent_counts += out['latent_counts'].cpu().numpy()
        all_n_tokens += out['n_tokens']

    # Compute fraction of tokens that activate each feature
    activation_fractions = all_latent_counts / all_n_tokens

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(activation_fractions, bins=100)
    plt.xlabel('Fraction of tokens')
    plt.ylabel('Number of features')
    plt.title(f'SAE Feature Activation Rates (Layer {layer})')

    if save_path:
        plt.savefig(save_path)
    plt.show()

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean activation rate: {activation_fractions.mean():.4f}")
    print(f"Median activation rate: {np.median(activation_fractions):.4f}")
    print(f"Std activation rate: {activation_fractions.std():.4f}")
    print(f"Max activation rate: {activation_fractions.max():.4f}")
    print(f"Min activation rate: {activation_fractions.min():.4f}")

    cleanup_gpu_memory()
