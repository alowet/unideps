"""Evaluation and visualization for dependency probes, supporting both multiclass and binary approaches."""

import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from model_utils import UDTransformer
from probing_merged import BinaryDependencyProbe, DependencyProbe, compute_loss
from sklearn.metrics import balanced_accuracy_score, f1_score
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
        if not isinstance(probe, DependencyProbe):
            raise TypeError(f"Expected DependencyProbe for multiclass evaluation, got {type(probe)}")

        # Move multiclass probe to device
        multiclass_probe = probe.to(device)
        multiclass_probe.eval()

        all_preds = []
        all_relations = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Get model outputs
                preds_masked, relations_masked = compute_loss(
                    multiclass_probe, model, batch, layer, device,
                    train_toks=train_toks,
                    return_type="preds",
                    frequent_deps=frequent_deps,
                    binary=False
                )

                all_preds.append(preds_masked.cpu())
                all_relations.append(relations_masked.cpu())

        # Concatenate predictions and relations
        all_preds = torch.cat(all_preds, dim=0).numpy()  # [n_tokens, n_deps]
        all_relations = torch.cat(all_relations, dim=0).numpy()  # [n_tokens, n_deps]
        print(all_preds.shape, all_relations.shape)

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

        # Evaluate each probe separately
        for dep_idx, dep_name in enumerate(dep_list):
            print(f"Evaluating probe for dependency: {dep_name}")

            # Move probe to device
            binary_probe = probe[dep_name].to(device)
            binary_probe.eval()

            dep_preds = []
            dep_relations = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Evaluating {dep_name}"):
                    # Get model outputs
                    preds_masked, relations_masked = compute_loss(
                        binary_probe, model, batch, layer, device,
                        dep_idx=dep_idx,
                        train_toks=train_toks,
                        return_type="preds",
                        frequent_deps=frequent_deps,
                        binary=True
                    )

                    dep_preds.append(preds_masked.cpu())
                    dep_relations.append(relations_masked.cpu())

            # Store concatenated predictions and relations for this dependency
            all_dep_preds[dep_name] = torch.cat(dep_preds, dim=0).numpy()
            all_dep_relations[dep_name] = torch.cat(dep_relations, dim=0).numpy()

            # Move probe back to CPU to free GPU memory
            binary_probe = binary_probe.cpu()

        # Convert to format compatible with multiclass evaluation
        all_preds = np.column_stack([all_dep_preds[dep] for dep in dep_list])
        all_relations = np.column_stack([all_dep_relations[dep] for dep in dep_list])

    else:
        raise ValueError(f"Invalid probe_type: {probe_type}. Must be 'multiclass' or 'binary'.")

    # Now calculate metrics (same for both approaches since data format is standardized)
    return calculate_metrics(all_preds, all_relations, dep_list)


def calculate_metrics(preds: np.ndarray, relations: np.ndarray, dep_list: List[str]) -> Dict[str, float]:
    """Calculate metrics from predictions and ground truth.

    Args:
        preds: Predictions array of shape [n_samples, n_deps]
        relations: Ground truth array of shape [n_samples, n_deps]
        dep_list: List of dependency relation names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Overall metrics
    metrics["balanced_accuracy"] = balanced_accuracy_score(
        relations.argmax(axis=1),
        preds.argmax(axis=1)
    )
    metrics["macro_f1"] = f1_score(
        relations.argmax(axis=1),
        preds.argmax(axis=1),
        average='macro'
    )
    metrics["weighted_f1"] = f1_score(
        relations.argmax(axis=1),
        preds.argmax(axis=1),
        average='weighted'
    )

    # Per-class metrics
    per_class_acc = {}
    per_class_f1 = {}

    for idx, dep in enumerate(dep_list):
        per_class_acc[dep] = balanced_accuracy_score(relations[:, idx], preds[:, idx])
        per_class_f1[dep] = f1_score(relations[:, idx], preds[:, idx])

    metrics["per_class_accuracy"] = per_class_acc
    metrics["per_class_f1"] = per_class_f1

    return metrics


def plot_layer_results(results: Dict[int, Dict[str, Any]], baseline: Dict[str, Any], frequent_deps: Optional[list] = None, save_path: Optional[str] = None, probe_type: str = "multiclass"):
    """Plot results across layers.

    Args:
        results: Dictionary mapping layer indices to metric dictionaries
        baseline: Dictionary containing baseline metrics
        frequent_deps: List of dependency types to evaluate on
        save_path: Path to save plot to
        probe_type: Type of probe used, either "multiclass" or "binary"
    """
    plt.figure(figsize=(20, 15))

    # Get layers and metrics
    layers = sorted(list(results.keys()))
    layers_str = [str(layer) for layer in layers]  # Convert to strings for plotting

    # Get list of dependency types
    first_layer = layers[0]
    dep_types = frequent_deps if frequent_deps is not None else list(results[first_layer].get("per_class_accuracy", {}).keys())

    # Extract metrics across layers
    balanced_accuracy = [results[layer].get("balanced_accuracy", 0) for layer in layers]
    macro_f1 = [results[layer].get("macro_f1", 0) for layer in layers]
    weighted_f1 = [results[layer].get("weighted_f1", 0) for layer in layers]

    # Get baseline metrics
    baseline_macro_f1 = baseline.get("macro_f1", 0)
    baseline_weighted_f1 = baseline.get("weighted_f1", 0)

    # Plot balanced accuracy
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(layers, balanced_accuracy, marker='o')
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Balanced Accuracy")
    ax1.set_title(f"Balanced Accuracy by Layer ({probe_type.capitalize()} Probes)")
    ax1.set_xticks(layers)
    ax1.grid(True)

    # Plot macro F1
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(layers, macro_f1, marker='o', label="Macro F1")
    ax2.axhline(y=baseline_macro_f1, color='r', linestyle='--', label="Baseline Macro F1")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Macro F1")
    ax2.set_title(f"Macro F1 by Layer ({probe_type.capitalize()} Probes)")
    ax2.set_xticks(layers)
    ax2.legend()
    ax2.grid(True)

    # Extract per-class F1 scores across layers
    per_class_f1 = np.zeros((len(layers), len(dep_types)))
    for i, layer in enumerate(layers):
        per_class_f1_dict = results[layer].get("per_class_f1", {})
        for j, dep in enumerate(dep_types):
            per_class_f1[i, j] = per_class_f1_dict.get(dep, 0)

    # Plot per-class F1 heatmap
    ax3 = plt.subplot(2, 2, 3)
    sns.heatmap(
        per_class_f1,
        ax=ax3,
        annot=True,
        fmt=".2f",
        xticklabels=dep_types,
        yticklabels=layers_str,
        cmap='viridis',
        vmin=0,
        vmax=1
    )
    ax3.set_xlabel("Dependency Type")
    ax3.set_ylabel("Layer")
    ax3.set_title(f"Per-Class F1 Score by Layer ({probe_type.capitalize()} Probes)")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def compute_majority_baseline(test_loader: DataLoader, frequent_deps: Optional[list] = None) -> Dict[str, Any]:
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
    baseline_metrics: Dict[str, Any] = {
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
    class_frequencies = {}
    for i, dep in enumerate(dep_list):
        class_frequencies[dep] = float(all_relations[:, i].mean())

    baseline_metrics["class_frequencies"] = class_frequencies
    baseline_metrics["majority_class"] = int(majority_class)

    return baseline_metrics


def main(
    test_loader: DataLoader,
    probes: Dict[int, Union[DependencyProbe, Dict[str, BinaryDependencyProbe]]],
    model: UDTransformer,
    train_toks: str = "tail",
    device: Optional[torch.device] = None,
    frequent_deps: Optional[list] = None,
    probe_type: str = "multiclass"
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
    print("\nComputing majority class baseline...")
    baseline = compute_majority_baseline(test_loader, frequent_deps)
    print(f"Majority class baseline:")
    print(f"  Macro F1: {baseline.get('macro_f1', 0):.3f}")
    print(f"  Weighted F1: {baseline.get('weighted_f1', 0):.3f}")
    print(f"\nClass frequencies:")

    dep_to_idx = DependencyTask.dependency_table()
    dep_list = frequent_deps if frequent_deps is not None else list(dep_to_idx.keys())

    # Safe access to class_frequencies dictionary
    class_frequencies = baseline.get("class_frequencies", {})
    for dep, freq in class_frequencies.items():
        if isinstance(freq, (int, float)) and freq > 0.01:  # Only show classes with >1% frequency
            print(f"  {dep}: {freq:.3f}")

    # Safe access to majority_class
    majority_class_idx = baseline.get("majority_class", 0)
    if isinstance(majority_class_idx, int) and 0 <= majority_class_idx < len(dep_list):
        print(f"\nMajority class: {dep_list[majority_class_idx]}")

    # Then evaluate probes
    results: Dict[int, Dict[str, Any]] = {}
    for layer, probe in probes.items():
        print(f"\nEvaluating layer {layer}")
        metrics = evaluate_probe(
            model, probe, test_loader, device, layer,
            train_toks=train_toks,
            frequent_deps=frequent_deps,
            probe_type=probe_type
        )
        results[layer] = metrics
        print(f"Balanced accuracy: {metrics.get('balanced_accuracy', 0):.3f}")
        print(f"Improvement over majority baseline:")
        print(f"  Macro F1: {metrics.get('macro_f1', 0) - baseline.get('macro_f1', 0):.3f}")
        print(f"  Weighted F1: {metrics.get('weighted_f1', 0) - baseline.get('weighted_f1', 0):.3f}")

    # Plot results
    plot_layer_results(
        results,
        baseline,
        frequent_deps,
        save_path=f"figures/evals/{probe_type}_eval_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps) if frequent_deps else len(dep_to_idx)}.png",
        probe_type=probe_type
    )

    # Save results with baseline
    results_with_baseline: Dict[str, Any] = {**{str(k): v for k, v in results.items()}, "baseline": baseline}
    save_path = f"data/evals/{probe_type}_eval_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps) if frequent_deps else len(dep_to_idx)}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(results_with_baseline, f)

    return results
