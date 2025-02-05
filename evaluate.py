"""Evaluation and visualization for dependency probes."""

import pickle
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from model_utils import UDTransformer
from probing import DependencyProbe, compute_loss
from sklearn.metrics import balanced_accuracy_score, f1_score
from task import DependencyTask
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_probe(
    model: UDTransformer,
    probe: DependencyProbe,
    test_loader: DataLoader,
    device: torch.device,
    layer: int,
    train_toks: str = "tail",
) -> Dict[str, float]:
    """Evaluate probe on test set.

    Args:
        model: Transformer model
        probe: Trained probe
        test_loader: DataLoader for test set
        layer: Layer to evaluate
        train_toks: Whether to use head or tail of dependency
        device: Device to use for computation

    Returns:
        Dictionary containing metrics:
            - balanced_accuracy: Accuracy balanced across classes
            - per_class_accuracy: Accuracy for each dependency type
    """

    print(f"Evaluating on device: {device}")

    probe.to(device)
    probe.eval()

    # Get mapping of indices to relation names
    dep_table = DependencyTask.dependency_table()
    idx_to_dep = {v-1: k for k, v in dep_table.items()}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get model outputs
            preds_masked, labels_masked = compute_loss(probe, model, batch, layer, device, train_toks, return_type="preds")

            all_preds.append(preds_masked.cpu())
            all_labels.append(labels_masked.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute metrics
    metrics = {}

    # Overall metrics
    metrics["balanced_accuracy"] = balanced_accuracy_score(
        all_labels.argmax(axis=1),
        all_preds.argmax(axis=1)
    )
    metrics["macro_f1"] = f1_score(
        all_labels.argmax(axis=1),
        all_preds.argmax(axis=1),
        average='macro'  # Treats all classes equally regardless of support
    )
    metrics["weighted_f1"] = f1_score(
        all_labels.argmax(axis=1),
        all_preds.argmax(axis=1),
        average='weighted'  # Weights by class support
    )

    # Per-class metrics
    per_class_acc = {}
    per_class_f1 = {}
    for i in range(all_labels.shape[1]):
        acc = balanced_accuracy_score(all_labels[:, i], all_preds[:, i])
        f1 = f1_score(all_labels[:, i], all_preds[:, i])
        per_class_acc[idx_to_dep[i]] = acc
        per_class_f1[idx_to_dep[i]] = f1
    metrics["per_class_accuracy"] = per_class_acc
    metrics["per_class_f1"] = per_class_f1

    return metrics


def plot_layer_results(results: Dict[int, Dict[str, float]], baseline: Dict[str, float], save_path: Optional[str] = None):
    """Plot evaluation results across layers.

    Args:
        results: Dictionary mapping layer indices to metric dictionaries
        baseline: Dictionary containing baseline metrics
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
    ax1.set_title("Overall Probe Performance by Layer")
    ax1.legend()
    ax1.grid(True)

    # Add baseline lines
    ax1.axhline(y=baseline["macro_f1"], color='r', linestyle='--', label='Majority Baseline (Macro F1)')
    ax1.axhline(y=baseline["weighted_f1"], color='g', linestyle='--', label='Majority Baseline (Weighted F1)')
    ax1.legend()

    # Plot per-class accuracy heatmap
    dep_types = list(results[layers[0]]["per_class_accuracy"].keys())
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
    ax2.set_title("Per-Class Balanced Accuracy by Layer")
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
    ax3.set_title("Per-Class F1 Score by Layer")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def compute_majority_baseline(test_loader: DataLoader) -> Dict[str, float]:
    """Compute baseline metrics assuming we always predict the most frequent class.

    Args:
        test_loader: DataLoader for test set

    Returns:
        Dictionary containing baseline metrics
    """
    # Collect all labels
    all_labels = []
    for batch in test_loader:
        labels = batch["labels"]
        dep_mask = batch["dep_mask"]
        labels_masked = labels[dep_mask]
        all_labels.append(labels_masked)

    # Concatenate all labels
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Find most frequent class
    majority_class = all_labels.sum(axis=0).argmax()

    # Create predictions array - all zeros except for majority class
    all_preds = np.zeros_like(all_labels)
    all_preds[:, majority_class] = 1

    # Compute metrics
    baseline_metrics = {
        "macro_f1": f1_score(
            all_labels.argmax(axis=1),
            all_preds.argmax(axis=1),
            average='macro'
        ),
        "weighted_f1": f1_score(
            all_labels.argmax(axis=1),
            all_preds.argmax(axis=1),
            average='weighted'
        )
    }

    # Also compute class frequencies for context
    class_frequencies = all_labels.sum(axis=0) / len(all_labels)
    baseline_metrics["class_frequencies"] = class_frequencies
    baseline_metrics["majority_class"] = majority_class

    return baseline_metrics


def main(
    test_loader: DataLoader,
    probes: Dict[int, DependencyProbe],
    model: UDTransformer,
    train_toks: str = "tail",
    device: Optional[torch.device] = None
):
    """Main evaluation function."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print(f"\nEvaluating on device: {device}")

    # First compute baseline
    print("\nComputing majority class baseline...")
    baseline = compute_majority_baseline(test_loader)
    print(f"Majority class baseline:")
    print(f"  Macro F1: {baseline['macro_f1']:.3f}")
    print(f"  Weighted F1: {baseline['weighted_f1']:.3f}")
    print(f"\nClass frequencies:")
    dep_table = DependencyTask.dependency_table()
    idx_to_dep = {v-1: k for k, v in dep_table.items()}
    for i, freq in enumerate(baseline["class_frequencies"]):
        if freq > 0.01:  # Only show classes with >1% frequency
            print(f"  {idx_to_dep[i]}: {freq:.3f}")
    print(f"\nMajority class: {idx_to_dep[baseline['majority_class']]}")

    # Then evaluate probes as before
    results = {}
    for layer, probe in probes.items():
        print(f"\nEvaluating layer {layer}")
        metrics = evaluate_probe(model, probe, test_loader, layer, train_toks, device)
        results[layer] = metrics
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"Improvement over majority baseline:")
        print(f"  Macro F1: {metrics['macro_f1'] - baseline['macro_f1']:.3f}")
        print(f"  Weighted F1: {metrics['weighted_f1'] - baseline['weighted_f1']:.3f}")

    # Plot results
    plot_layer_results(results, baseline, save_path=f"figures/evals/eval_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}.png")

    # Save results with baseline
    results["baseline"] = baseline
    with open(f"data/evals/eval_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}.pkl", "wb") as f:
        pickle.dump(results, f)
