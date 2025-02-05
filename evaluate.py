"""Evaluation and visualization for dependency probes."""

import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from model_utils import UDTransformer
from probing import DependencyProbe, compute_loss
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from task import DependencyTask
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_probe(
    model: UDTransformer,
    probe: DependencyProbe,
    test_loader: DataLoader,
    layer: int,
    train_toks: str = "tail",
    device: Optional[torch.device] = None,
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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")

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
            activations = model.get_activations(batch, layer_idx=layer, train_toks=train_toks)
            scores = probe(activations)

            # Get labels and mask
            labels = batch["labels"].to(device)
            dep_mask = batch["dep_mask"].to(device)

            # Get predictions (sigmoid since we're using BCE loss)
            preds = torch.sigmoid(scores) > 0.5

            # Only evaluate on unmasked tokens
            preds_masked = preds[dep_mask]
            labels_masked = labels[dep_mask]

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


def plot_layer_results(results: Dict[int, Dict[str, float]], save_path: Optional[str] = None):
    """Plot evaluation results across layers.

    Args:
        results: Dictionary mapping layer indices to metric dictionaries
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


def main(test_loader: DataLoader, probes: Dict[int, DependencyProbe], model: UDTransformer, train_toks: str = "tail"):
    """Main evaluation function.

    Args:
        test_loader: DataLoader for test set
        model_path: Path to saved probes
        train_toks: Whether to use head or tail of dependency
    """
    # Evaluate each probe
    results = {}
    for layer, probe in probes.items():
        print(f"\nEvaluating layer {layer}")
        metrics = evaluate_probe(model, probe, test_loader, layer, train_toks)
        results[layer] = metrics
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")

    # Plot results
    plot_layer_results(results, save_path=f"figures/{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}.png")

    # Save results
    with open(f"data/{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}.pkl", "wb") as f:
        pickle.dump(results, f)
