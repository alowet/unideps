"""Evaluation and visualization for binary dependency probes."""

import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from binary_probing import BinaryDependencyProbe, compute_binary_loss
from model_utils import UDTransformer
from sklearn.metrics import balanced_accuracy_score, f1_score
from task import DependencyTask
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_binary_probes(
    model: UDTransformer,
    probes: Dict[str, BinaryDependencyProbe],
    test_loader: DataLoader,
    device: torch.device,
    layer: int,
    train_toks: str = "tail",
    frequent_deps: Optional[list] = None,
) -> Dict[str, float]:
    """Evaluate binary probes on test set.

    Args:
        model: Transformer model
        probes: Dictionary of trained binary probes, one per dependency relation
        test_loader: DataLoader for test set
        layer: Layer to evaluate
        train_toks: Whether to use head or tail of dependency
        device: Device to use for computation
        frequent_deps: List of dependency types to evaluate on

    Returns:
        Dictionary containing metrics:
            - balanced_accuracy: Accuracy averaged across all binary probes
            - per_class_accuracy: Accuracy for each dependency type
    """
    print(f"Evaluating binary probes on device: {device}")

    # Get list of dependencies to evaluate
    dep_list = frequent_deps if frequent_deps is not None else list(DependencyTask.dependency_table().keys())

    # Check that we have probes for all dependencies
    for dep in dep_list:
        if dep not in probes:
            raise ValueError(f"No probe found for dependency '{dep}'")

    # Initialize dictionaries to store metrics
    all_preds_dict = {}
    all_relations_dict = {}

    # Evaluate each probe separately
    for dep_idx, dep_name in enumerate(dep_list):
        print(f"Evaluating probe for dependency: {dep_name}")

        # Move probe to device
        probe = probes[dep_name].to(device)
        probe.eval()

        all_preds = []
        all_relations = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {dep_name}"):
                # Get model outputs
                preds_masked, relations_masked = compute_binary_loss(
                    probe, model, batch, layer, device,
                    dep_idx=dep_idx,
                    train_toks=train_toks,
                    return_type="preds",
                    frequent_deps=frequent_deps
                )

                all_preds.append(preds_masked.cpu())
                all_relations.append(relations_masked.cpu())

        # Concatenate all predictions and relations
        all_preds_dict[dep_name] = torch.cat(all_preds, dim=0).numpy()  # [n_tokens]
        all_relations_dict[dep_name] = torch.cat(all_relations, dim=0).numpy()  # [n_tokens]

        # Move probe back to CPU to free GPU memory
        probe = probe.cpu()

    # Compute metrics
    metrics = {}

    # Per-class metrics
    per_class_acc = {}
    per_class_f1 = {}

    for dep_name in dep_list:
        per_class_acc[dep_name] = balanced_accuracy_score(
            all_relations_dict[dep_name],
            all_preds_dict[dep_name]
        )
        per_class_f1[dep_name] = f1_score(
            all_relations_dict[dep_name],
            all_preds_dict[dep_name]
        )

    # Stack all predictions and relations into 2D arrays for overall metrics
    # Create a combined array where each column is a dependency
    all_preds_combined = np.column_stack([all_preds_dict[dep] for dep in dep_list])
    all_relations_combined = np.column_stack([all_relations_dict[dep] for dep in dep_list])

    # Overall metrics
    metrics["macro_f1"] = f1_score(
        all_relations_combined.argmax(axis=1),
        all_preds_combined.argmax(axis=1),
        average='macro'
    )
    metrics["weighted_f1"] = f1_score(
        all_relations_combined.argmax(axis=1),
        all_preds_combined.argmax(axis=1),
        average='weighted'
    )
    metrics["balanced_accuracy"] = balanced_accuracy_score(
        all_relations_combined.argmax(axis=1),
        all_preds_combined.argmax(axis=1)
    )

    # Add per-class metrics to the results
    metrics["per_class_accuracy"] = per_class_acc
    metrics["per_class_f1"] = per_class_f1

    return metrics


def plot_binary_layer_results(results: Dict[int, Dict[str, float]], baseline: Dict[str, float], frequent_deps: Optional[list] = None, save_path: Optional[str] = None):
    """Plot evaluation results across layers.

    Args:
        results: Dictionary mapping layer indices to metric dictionaries
        baseline: Dictionary containing baseline metrics
        frequent_deps: List of dependency types to evaluate on
        save_path: Optional path to save plot
    """
    # Set up plot style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))

    # Plot overall metrics
    layers = sorted(results.keys())
    accuracies = [results[l]["balanced_accuracy"] for l in layers]
    macro_f1s = [results[l]["macro_f1"] for l in layers]
    weighted_f1s = [results[l]["weighted_f1"] for l in layers]

    ax1.plot(layers, accuracies, marker='o', label='Balanced Accuracy')
    ax1.plot(layers, macro_f1s, marker='s', label='Macro F1')
    ax1.plot(layers, weighted_f1s, marker='^', label='Weighted F1')
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Score")
    ax1.set_title("Overall Binary Probe Performance by Layer")
    ax1.legend()
    ax1.grid(True)

    # Add baseline lines
    ax1.axhline(y=baseline["macro_f1"], color='r', linestyle='--', label='Majority Baseline (Macro F1)')
    ax1.axhline(y=baseline["weighted_f1"], color='g', linestyle='--', label='Majority Baseline (Weighted F1)')
    ax1.legend()

    # Plot per-class accuracy heatmap
    dep_types = frequent_deps if frequent_deps is not None else list(results[layers[0]]["per_class_accuracy"].keys())
    acc_matrix = np.zeros((len(layers), len(dep_types)))
    f1_matrix = np.zeros((len(layers), len(dep_types)))

    for i, layer in enumerate(layers):
        for j, dep in enumerate(dep_types):
            acc_matrix[i, j] = results[layer]["per_class_accuracy"][dep]
            f1_matrix[i, j] = results[layer]["per_class_f1"][dep]

    # Plot accuracy heatmap
    sns.heatmap(
        acc_matrix,
        xticklabels=dep_types,
        yticklabels=layers,
        ax=ax2,
        cmap='viridis',
        vmin=0,
        vmax=1
    )
    ax2.set_xlabel("Dependency Type")
    ax2.set_ylabel("Layer")
    ax2.set_title("Per-Class Balanced Accuracy by Layer (Binary Probes)")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # Plot F1 heatmap
    sns.heatmap(
        f1_matrix,
        xticklabels=dep_types,
        yticklabels=layers,
        ax=ax3,
        cmap='viridis',
        vmin=0,
        vmax=1
    )
    ax3.set_xlabel("Dependency Type")
    ax3.set_ylabel("Layer")
    ax3.set_title("Per-Class F1 Score by Layer (Binary Probes)")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def compute_binary_majority_baseline(test_loader: DataLoader, frequent_deps: Optional[list] = None) -> Dict[str, float]:
    """Compute baseline metrics assuming we always predict the most frequent class.

    Args:
        test_loader: DataLoader for test set
        frequent_deps: List of dependency types to evaluate on

    Returns:
        Dictionary containing baseline metrics
    """
    # Get mapping of indices to relation names
    dep_to_idx = DependencyTask.dependency_table()

    # Get dependencies to evaluate
    dep_list = frequent_deps if frequent_deps is not None else list(dep_to_idx.keys())

    # Collect all relations
    all_relations_dict = {dep: [] for dep in dep_list}

    for batch in test_loader:
        relations = batch["relations"]
        relation_mask = batch["relation_mask"]

        # For each dependency, extract its binary labels
        for dep_idx, dep_name in enumerate(dep_list):
            # Get the index in the full relation set
            if frequent_deps is not None:
                full_dep_idx = dep_to_idx[dep_name]
            else:
                full_dep_idx = dep_idx

            # Extract binary labels for this dependency
            dep_relations = relations[relation_mask][:, full_dep_idx]
            all_relations_dict[dep_name].append(dep_relations)

    # Create majority class predictions for each dependency
    all_preds_dict = {}

    for dep_name in dep_list:
        # Concatenate all relations for this dependency
        dep_relations = torch.cat(all_relations_dict[dep_name], dim=0).numpy()

        # Determine majority class (0 or 1)
        majority_class = int(dep_relations.mean() > 0.5)

        # Create predictions array with all majority class
        all_preds_dict[dep_name] = np.full_like(dep_relations, majority_class)

    # Stack all predictions and relations into 2D arrays for overall metrics
    all_preds_combined = np.column_stack([all_preds_dict[dep] for dep in dep_list])
    all_relations_combined = np.column_stack([torch.cat(all_relations_dict[dep], dim=0).numpy() for dep in dep_list])

    # Compute overall metrics
    baseline_metrics = {
        "macro_f1": f1_score(
            all_relations_combined.argmax(axis=1),
            all_preds_combined.argmax(axis=1),
            average='macro'
        ),
        "weighted_f1": f1_score(
            all_relations_combined.argmax(axis=1),
            all_preds_combined.argmax(axis=1),
            average='weighted'
        )
    }

    # Compute class frequencies
    class_frequencies = {dep: all_relations_combined[:, i].mean() for i, dep in enumerate(dep_list)}
    baseline_metrics["class_frequencies"] = class_frequencies

    # Find majority class
    majority_class_idx = all_relations_combined.sum(axis=0).argmax()
    baseline_metrics["majority_class"] = dep_list[majority_class_idx]

    return baseline_metrics


def main(
    test_loader: DataLoader,
    probes: Dict[int, Dict[str, BinaryDependencyProbe]],
    model: UDTransformer,
    train_toks: str = "tail",
    device: Optional[torch.device] = None,
    frequent_deps: Optional[list] = None
):
    """Main evaluation function for binary probes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print(f"\nEvaluating binary probes on device: {device}")

    # First compute baseline
    print("\nComputing majority class baseline...")
    baseline = compute_binary_majority_baseline(test_loader, frequent_deps)
    print(f"Majority class baseline:")
    print(f"  Macro F1: {baseline['macro_f1']:.3f}")
    print(f"  Weighted F1: {baseline['weighted_f1']:.3f}")
    print(f"\nClass frequencies:")
    for dep, freq in baseline["class_frequencies"].items():
        if freq > 0.01:  # Only show classes with >1% frequency
            print(f"  {dep}: {freq:.3f}")
    print(f"\nMajority class: {baseline['majority_class']}")

    # Then evaluate probes
    results = {}
    for layer, layer_probes in probes.items():
        print(f"\nEvaluating layer {layer}")
        metrics = evaluate_binary_probes(model, layer_probes, test_loader, device, layer, train_toks, frequent_deps)
        results[layer] = metrics
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"Improvement over majority baseline:")
        print(f"  Macro F1: {metrics['macro_f1'] - baseline['macro_f1']:.3f}")
        print(f"  Weighted F1: {metrics['weighted_f1'] - baseline['weighted_f1']:.3f}")

    # Plot results
    plot_binary_layer_results(results, baseline, frequent_deps, save_path=f"figures/evals/binary_eval_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps) if frequent_deps else len(DependencyTask.dependency_table())}.png")

    # Save results with baseline
    results["baseline"] = baseline
    with open(f"data/evals/binary_eval_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps) if frequent_deps else len(DependencyTask.dependency_table())}.pkl", "wb") as f:
        pickle.dump(results, f)
