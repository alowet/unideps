"""Evaluation and visualization for dependency probes."""

import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from model_utils import UDTransformer
from probing import DependencyProbe, compute_loss
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_curve
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
    frequent_deps: Optional[list] = None,
) -> Dict[str, float]:
    """Evaluate probe on test set.

    Args:
        model: Transformer model
        probe: Trained probe
        test_loader: DataLoader for test set
        layer: Layer to evaluate
        train_toks: Whether to use head or tail of dependency
        device: Device to use for computation
        frequent_deps: List of dependency types to evaluate on

    Returns:
        Dictionary containing metrics:
            - balanced_accuracy: Accuracy balanced across classes
            - per_class_accuracy: Accuracy for each dependency type
    """

    print(f"Evaluating on device: {device}")

    probe.to(device)
    probe.eval()

    # Get mapping of indices to relation names
    dep_to_idx = DependencyTask.dependency_table()
    idx_to_dep = {v: k for k, v in dep_to_idx.items()}

    all_preds = []
    all_relations = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Get model outputs
            preds_masked, relations_masked = compute_loss(
                probe, model, batch, layer, device,
                train_toks=train_toks,
                return_type="preds",
                frequent_deps=frequent_deps
            )

            all_preds.append(preds_masked.cpu())
            all_relations.append(relations_masked.cpu())

    # Concatenate all predictions and relations
    all_preds = torch.cat(all_preds, dim=0).numpy()  # [n_tokens, n_deps]
    all_relations = torch.cat(all_relations, dim=0).numpy()  # [n_tokens, n_deps]
    print(all_preds.shape, all_relations.shape)

    # Compute metrics
    metrics = {}

    # Overall metrics
    metrics["balanced_accuracy"] = balanced_accuracy_score(
        all_relations.argmax(axis=1),
        all_preds.argmax(axis=1)
    )
    metrics["macro_f1"] = f1_score(
        all_relations.argmax(axis=1),
        all_preds.argmax(axis=1),
        average='macro'
    )
    metrics["weighted_f1"] = f1_score(
        all_relations.argmax(axis=1),
        all_preds.argmax(axis=1),
        average='weighted'
    )

    # Per-class metrics
    per_class_acc = {}
    per_class_f1 = {}

    # If using frequent_deps, only evaluate those dependencies
    eval_deps = frequent_deps if frequent_deps is not None else dep_to_idx.keys()

    for idx, dep in enumerate(eval_deps):
        acc = balanced_accuracy_score(all_relations[:, idx], all_preds[:, idx])
        f1 = f1_score(all_relations[:, idx], all_preds[:, idx])
        per_class_acc[dep] = acc
        per_class_f1[dep] = f1

    metrics["per_class_accuracy"] = per_class_acc
    metrics["per_class_f1"] = per_class_f1

    return metrics


def plot_layer_results(results: Dict[int, Dict[str, float]], baseline: Dict[str, float], frequent_deps: Optional[list] = None, save_path: Optional[str] = None):
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
    ax1.set_title("Overall Probe Performance by Layer")
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


def compute_majority_baseline(test_loader: DataLoader, frequent_deps: Optional[list] = None) -> Dict[str, float]:
    """Compute baseline metrics assuming we always predict the most frequent class.

    Args:
        test_loader: DataLoader for test set
        frequent_deps: List of dependency types to evaluate on

    Returns:
        Dictionary containing baseline metrics
    """
    # Get mapping of indices to relation names
    dep_to_idx = DependencyTask.dependency_table()

    # Collect all relations
    all_relations = []
    for batch in test_loader:
        relations = batch["relations"]
        relation_mask = batch["relation_mask"]
        relations_masked = relations[relation_mask]

        # If using frequent_deps, only keep those columns
        if frequent_deps is not None:
            dep_indices = [dep_to_idx[dep] for dep in frequent_deps]
            relations_masked = relations_masked[:, dep_indices]

        all_relations.append(relations_masked)

    # Concatenate all relations
    all_relations = torch.cat(all_relations, dim=0).numpy()

    # Find most frequent class
    majority_class = all_relations.sum(axis=0).argmax()

    # Create predictions array - all zeros except for majority class
    all_preds = np.zeros_like(all_relations)
    all_preds[:, majority_class] = 1

    # Compute metrics
    baseline_metrics = {
        "macro_f1": f1_score(
            all_relations.argmax(axis=1),
            all_preds.argmax(axis=1),
            average='macro'
        ),
        "weighted_f1": f1_score(
            all_relations.argmax(axis=1),
            all_preds.argmax(axis=1),
            average='weighted'
        )
    }

    # Also compute class frequencies for context
    class_frequencies = all_relations.sum(axis=0) / len(all_relations)
    baseline_metrics["class_frequencies"] = class_frequencies
    baseline_metrics["majority_class"] = majority_class

    return baseline_metrics


def plot_probe_pr_curves(
    results: Dict[int, Dict],
    frequent_deps: List[str],
    save_path: Optional[str] = None
):
    """Plot precision-recall curves for probes.

    Args:
        results: Dictionary mapping layer indices to evaluation results
        frequent_deps: List of dependency types to evaluate
        save_path: Optional path to save plot
    """
    plt.style.use('seaborn-v0_8')
    n_deps = len(frequent_deps)
    n_cols = min(7, n_deps)
    n_rows = (n_deps + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    # Create color map for layers
    n_layers = len(results)
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

    for dep_idx, dep in enumerate(frequent_deps):
        row = dep_idx // n_cols
        col = dep_idx % n_cols
        ax = axs[row, col]

        max_f1 = 0
        best_layer = None

        for layer_idx, layer_results in results.items():
            per_class = layer_results["per_class"][dep]
            raw_preds = per_class["raw_preds"]
            true_labels = per_class["true_labels"]

            # Compute precision-recall curve
            precision, recall, thresholds = precision_recall_curve(true_labels, raw_preds)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            max_layer_f1 = np.max(f1_scores)

            if max_layer_f1 > max_f1:
                max_f1 = max_layer_f1
                best_layer = layer_idx

            ax.plot(recall, precision, color=colors[layer_idx], alpha=0.5,
                   label=f'Layer {layer_idx}' if col == 0 else None)

        ax.set_title(f'{dep}\nBest F1={max_f1:.3f} (L{best_layer})')
        ax.set_xlabel('Recall' if row == n_rows-1 else '')
        ax.set_ylabel('Precision' if col == 0 else '')
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for idx in range(dep_idx + 1, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axs[row, col])

    # Add legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def main(
    test_loader: DataLoader,
    probes: Dict[int, DependencyProbe],
    model: UDTransformer,
    train_toks: str = "tail",
    device: Optional[torch.device] = None,
    frequent_deps: Optional[list] = None,
    probe_type: str = "multiclass"
):
    """Main evaluation function."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    print(f"\nEvaluating on device: {device}")

    # First compute baseline
    print("\nComputing majority class baseline...")
    baseline = compute_majority_baseline(test_loader, frequent_deps)
    print(f"Majority class baseline:")
    print(f"  Macro F1: {baseline['macro_f1']:.3f}")
    print(f"  Weighted F1: {baseline['weighted_f1']:.3f}")
    print(f"\nClass frequencies:")
    dep_to_idx = DependencyTask.dependency_table()
    idx_to_dep = {v: k for k, v in dep_to_idx.items()}
    for i, freq in enumerate(baseline["class_frequencies"]):
        if freq > 0.01:  # Only show classes with >1% frequency
            print(f"  {idx_to_dep[i]}: {freq:.3f}")
    print(f"\nMajority class: {idx_to_dep[baseline['majority_class']]}")

    # Then evaluate probes as before
    results = {}
    for layer, probe in probes.items():
        print(f"\nEvaluating layer {layer}")
        metrics = evaluate_probe(model, probe, test_loader, device, layer, train_toks, frequent_deps)
        results[layer] = metrics
        print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
        print(f"Improvement over majority baseline:")
        print(f"  Macro F1: {metrics['macro_f1'] - baseline['macro_f1']:.3f}")
        print(f"  Weighted F1: {metrics['weighted_f1'] - baseline['weighted_f1']:.3f}")

    # Plot results
    plot_layer_results(results, baseline, frequent_deps, save_path=f"figures/evals/eval_{probe_type}_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps)}.png")

    save_path = f"figures/evals/eval_{probe_type}_{train_toks}_pr_curves_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps)}.png"
    print(f"Saving PR curves to {save_path}")
    plot_probe_pr_curves(results, frequent_deps, save_path=save_path)

    # Save results with baseline
    results["baseline"] = baseline
    with open(f"data/evals/eval_{probe_type}_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps)}.pkl", "wb") as f:
        pickle.dump(results, f)
