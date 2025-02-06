"""Analyze relationship between probe weights and SAE features."""

import gc
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from IPython.display import IFrame, display
from sae_lens import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from task import DependencyTask
from tqdm import tqdm


def display_dashboard(
    sae_release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_20/width_16k/canonical",
    latent_idx=0,
    width=800,
    height=600,
):
    release = get_pretrained_saes_directory()[sae_release]
    neuronpedia_id = release.neuronpedia_id[sae_id]

    url = f"https://neuronpedia.org/{neuronpedia_id}/{latent_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"

    print(url)
    display(IFrame(url, width=width, height=height))


def cleanup_gpu_memory():
    """Delete all CUDA tensors in the current scope."""
    # Get all variables in current scope
    for name, value in list(locals().items()):
        # Check if it's a CUDA tensor
        if isinstance(value, torch.Tensor) and value.is_cuda:
            print(f"Deleting {name} from locals")
            del locals()[name]

    # Force CUDA memory cleanup
    torch.cuda.empty_cache()
    gc.collect()


def analyze_probe_sae_alignment(
    probes: Dict[int, nn.Module],
    device_sae: torch.device | None = None,
    top_k: int = 5,
    plot_dashboards: bool = False,
    num_random_samples: int = 10,
    batch_size: int = 8192,  # Process this many features at a time
) -> Tuple[Dict[int, Dict[str, List[Tuple[int, float]]]], torch.Tensor, torch.Tensor]:
    """Analyze alignment between probe weights and SAE features.

    Args:
        probes: Dictionary mapping layer indices to probes
        device_sae: Device for SAE and similarity computations
        top_k: Number of top features to return
        plot_dashboards: Whether to display feature dashboards
        num_random_samples: Number of random samples for null distribution
        batch_size: Number of features to process at a time
    """
    if device_sae is None:
        device_sae = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")

    # Disable gradient computation
    torch.set_grad_enabled(False)

    gemmascope_sae_release = "gemma-scope-2b-pt-res-canonical"

    # Get mapping of indices to relation names
    idx_to_dep = DependencyTask.dependency_table()

    # Get all dependency types and layers
    dep_types = sorted(list(idx_to_dep.values()))
    layers = sorted(list(probes.keys()))

    # Create matrices for storing results
    max_similarities = torch.zeros((len(layers), len(dep_types)))
    relative_similarities = torch.zeros((len(layers), len(dep_types)))
    results = {}

    for layer_idx, layer in enumerate(tqdm(layers, desc="Analyzing layers")):
        # Load pretrained SAE
        gemmascope_sae_id = f"layer_{layer}/width_16k/canonical"
        sae = SAE.from_pretrained(gemmascope_sae_release, gemmascope_sae_id, device=str(device_sae))[0]

        # Move probe to SAE GPU and detach from computation graph
        probe = probes[layer].to(device_sae)
        weights = probe.probe.weight.detach()  # [num_relations, hidden_dim]
        sae_features = sae.W_enc.detach()  # [hidden_dim, num_features]

        # Normalize vectors
        weights_norm = torch.nn.functional.normalize(weights, p=2, dim=1)
        features_norm = torch.nn.functional.normalize(sae_features, p=2, dim=0)

        # Compute similarities in batches
        num_features = features_norm.size(1)
        similarities_list = []

        for start_idx in range(0, num_features, batch_size):
            end_idx = min(start_idx + batch_size, num_features)
            # Compute similarities for this batch of features
            batch_similarities = torch.matmul(
                weights_norm,
                features_norm[:, start_idx:end_idx]
            )
            similarities_list.append(batch_similarities)

        # Concatenate all batches
        similarities = torch.cat(similarities_list, dim=1)

        # Get max similarity for each relation
        max_sims, _ = similarities.max(dim=1)  # [num_relations]

        # Compute null distribution with streaming max computation
        null_sims = []
        for _ in range(num_random_samples):

            # Initialize max similarities for this sample
            current_max_sims = torch.full(
                (weights_norm.size(0),),
                float('-inf'),
                device=device_sae
            )

            for start_idx in range(0, num_features, batch_size):
                end_idx = min(start_idx + batch_size, num_features)

                # Randomly permute each dimension independently for this batch
                permuted_features = torch.stack([
                    features_norm[torch.randperm(features_norm.size(0), device=device_sae), i]
                    for i in range(start_idx, end_idx)
                ], dim=1)

                # Compute similarities for this small batch
                batch_similarities = torch.matmul(weights_norm, permuted_features)

                # Update running maximum
                current_max_sims = torch.maximum(
                    current_max_sims,
                    batch_similarities.max(dim=1)[0]
                )

                # Clear intermediate tensors
                del batch_similarities, permuted_features
                torch.cuda.empty_cache()

            # Store max similarities for this sample
            null_sims.append(current_max_sims.clone())

            # Clear memory after each sample
            del current_max_sims
            torch.cuda.empty_cache()

        # Stack all samples and compute statistics
        null_sims = torch.stack(null_sims)
        null_mean = null_sims.mean(dim=0)
        null_std = null_sims.std(dim=0)
        z_scores = (max_sims - null_mean) / null_std

        # Store results for both uses
        layer_results = {}
        for i, dep_type in enumerate(dep_types):
            # Store max similarities and z-scores (move to CPU for storage)
            max_similarities[layer_idx, i] = max_sims[i].cpu()
            relative_similarities[layer_idx, i] = z_scores[i].cpu()

            # Get top-k features and similarities
            top_k_values, top_k_indices = torch.topk(similarities[i], top_k)

            if plot_dashboards:
                for idx in top_k_indices:
                    display_dashboard(
                        sae_release=gemmascope_sae_release,
                        sae_id=gemmascope_sae_id,
                        latent_idx=int(idx)
                    )

            layer_results[dep_type] = [
                (int(idx.cpu()), float(sim.cpu()))
                for idx, sim in zip(top_k_indices, top_k_values)
            ]

        results[layer] = layer_results

        # Cleanup
        cleanup_gpu_memory()

    # Re-enable gradients if necessary
    torch.set_grad_enabled(True)

    return results, max_similarities, relative_similarities


def print_alignment_results(results: Dict[int, Dict[str, List[Tuple[int, float]]]]):
    """Print human-readable alignment results."""
    for layer in sorted(results.keys()):
        print(f"\nLayer {layer}:")
        layer_results = results[layer]

        # Sort dependency types by maximum similarity
        dep_types = sorted(
            layer_results.keys(),
            key=lambda x: max(sim for _, sim in layer_results[x]),
            reverse=True
        )

        for dep_type in dep_types:
            print(f"\n  {dep_type}:")
            for feat_idx, sim in layer_results[dep_type]:
                print(f"    Feature {feat_idx}: {sim:.3f}")


def plot_sae_alignment_heatmaps(
    max_similarities: torch.Tensor,
    relative_similarities: torch.Tensor,
    dep_types: List[str],
    layers: List[int],
    save_path: Optional[str] = None
):
    """Plot heatmaps showing SAE alignment with dependency relations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

    # Max similarities heatmap
    sns.heatmap(
        max_similarities.cpu(),
        xticklabels=dep_types,
        yticklabels=layers,
        ax=ax1,
        cmap='viridis',
        vmin=0,
        vmax=1
    )
    ax1.set_xlabel("Dependency Type")
    ax1.set_ylabel("Layer")
    ax1.set_title("Maximum Cosine Similarity with SAE Features")
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    # Relative similarities heatmap
    sns.heatmap(
        relative_similarities.cpu(),
        xticklabels=dep_types,
        yticklabels=layers,
        ax=ax2,
        cmap='RdBu',
        center=0,
        vmin=-5,
        vmax=5
    )
    ax2.set_xlabel("Dependency Type")
    ax2.set_ylabel("Layer")
    ax2.set_title("Z-Score Relative to Random Feature Alignment")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def main(
    probes: Dict[int, nn.Module],
    train_toks: str = "tail",
    device_sae: Optional[torch.device] = None
):
    """Main analysis function."""
    print("\nAnalyzing probe-SAE alignment...")
    results, max_sims, rel_sims = analyze_probe_sae_alignment(
        probes,
        device_sae
    )

    print("\nResults:")
    print_alignment_results(results)

    print("\nGenerating alignment heatmaps...")
    dep_types = sorted(list(results[list(results.keys())[0]].keys()))
    layers = sorted(list(results.keys()))

    plot_sae_alignment_heatmaps(
        max_sims,
        rel_sims,
        dep_types,
        layers,
        save_path=f"figures/sae/sae_alignment_{train_toks}_heatmaps_layers_{min(probes.keys())}-{max(probes.keys())}.png"
    )

    # Save results
    torch.save(
        {"results": results, "max_similarities": max_sims, "relative_similarities": rel_sims},
        f"data/sae/sae_alignment_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}.pt"
    )
