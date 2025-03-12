"""Linear probes for dependency parsing, supporting both multiclass and binary classification approaches."""

import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import wandb
from model_utils import UDTransformer
from task import DependencyTask
from torch.utils.data import DataLoader
from tqdm import tqdm


class DependencyProbe(nn.Module):
    """Probe for predicting dependency relations at each position."""

    def __init__(self, input_dim: int, num_relations: int):
        super().__init__()
        self.probe = nn.Linear(input_dim, num_relations)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Maps token embeddings to dependency relation logits."""
        return self.probe(embeddings)


class BinaryDependencyProbe(nn.Module):
    """Binary probe for predicting a single dependency relation at each position."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.probe = nn.Linear(input_dim, 1)  # Binary output

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Maps token embeddings to a single dependency relation logit."""
        return self.probe(embeddings)


def compute_loss(
    probe: Union[DependencyProbe, BinaryDependencyProbe],
    model: UDTransformer,
    batch: Dict,
    layer: int,
    device: torch.device,
    train_toks: str = "tail",
    criterion: nn.Module = nn.BCEWithLogitsLoss(reduction='mean'),
    return_type: str = "loss",
    frequent_deps: Optional[list] = None,
    binary: bool = False,
    dep_idx: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute loss for a single batch.

    Args:
        probe: The probe model (either multiclass or binary)
        model: The transformer model
        batch: Batch of data
        layer: Layer to probe
        device: Device to use
        train_toks: Whether to use head or tail of dependency, or some other token entirely
        criterion: Loss function
        return_type: Whether to return loss (training/dev) or predictions (testing)
        frequent_deps: Set of dependency relations that appear frequently enough
        binary: Whether the probe is a binary probe
        dep_idx: For binary probes, the index of the dependency relation to predict

    Returns:
        loss: The computed loss
        scores: The probe's predictions
    """
    # Get activations and compute scores
    activations = model.get_activations(batch, layer_idx=layer, train_toks=train_toks)  # size [batch, max_len, hidden_dim]
    scores = probe(activations)  # size [batch, max_len, num_relations] or [batch, max_len, 1]

    # Get labels and mask
    relations = batch["relations"].to(device)  # size [batch, max_tokens, num_relations]
    relation_mask = batch["relation_mask"].to(device)  # size [batch, max_tokens]

    dep_table = DependencyTask.dependency_table()
    dep_indices = [dep_table[dep] for dep in frequent_deps] if frequent_deps is not None else list(dep_table.values())

    if binary:
        # Binary classification: extract specific dependency relation
        assert dep_idx is not None, "dep_idx must be specified for binary probes"
        full_dep_idx = dep_indices[dep_idx]  # Get the index in the full relation set

        # Squeeze out the singleton dimension for scores and predictions
        scores_squeezed = scores.squeeze(-1)  # [batch, max_len]
        preds = (torch.sigmoid(scores_squeezed) > 0.5)  # [batch, max_len]

        relations_notnan = ~(relations.isnan()[relation_mask][:, dep_indices].any(1))
        scores_masked = scores_squeezed[relation_mask][relations_notnan]  # [relation_mask.sum()]
        preds_masked = preds[relation_mask][relations_notnan]  # [relation_mask.sum()]

        if train_toks == "tail":
            relations_masked = relations[relation_mask][:, full_dep_idx]  # [relation_mask.sum()]
            relations_masked = relations_masked[relations_notnan]
        else:
            if train_toks == "head":
                # integer tensor, where each int is the index of the head token for the relation
                head_idxs = batch["head_idxs"].to(device)  # size [batch, max_heads, num_relations]

                # Get dimensions
                batch_size, max_heads, num_relations = head_idxs.shape

                # Create a binary tensor of same shape as relations [batch, max_tokens, num_relations]
                relations_head = torch.zeros_like(relations)

                # Create batch and relation indices for indexing
                batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand_as(head_idxs)
                rel_idx = torch.arange(num_relations, device=device).view(1, 1, -1).expand_as(head_idxs)

                # Mask out padded/invalid indices
                valid_mask = head_idxs >= 0

                # Set 1s at head positions for each relation
                relations_head[
                    batch_idx[valid_mask],
                    head_idxs[valid_mask].long(),
                    rel_idx[valid_mask]
                ] = 1.0

                # Apply relation mask to get final tensor
                relations_masked = relations_head[relation_mask][:, full_dep_idx]  # [relation_mask.sum()]
                relations_masked = relations_masked[relations_notnan]
            elif train_toks == "last":
                # try to decode all relations present in the batch based only on the last token
                seq_lens = relation_mask.sum(dim=1)  # shape [batch]
                full_relations = torch.any(relations[relation_mask][:, dep_indices], dim=1).float()  # [batch, num_relations]
                relations_masked = full_relations[:, dep_idx]  # [batch]

                # Get scores/preds at last token position for each sequence
                batch_idx = torch.arange(len(seq_lens), device=device)
                scores_masked = scores_squeezed[batch_idx, seq_lens - 1]  # [batch]
                preds_masked = preds[batch_idx, seq_lens - 1]  # [batch]
    else:
        # Multiclass classification
        preds = torch.sigmoid(scores) > 0.5  # sigmoid since we're using BCE loss

        relations_notnan = ~(relations.isnan()[relation_mask][:, dep_indices].any(1))
        scores_masked = scores[relation_mask][relations_notnan]  # [relation_mask.sum(), num_relations]
        preds_masked = preds[relation_mask][relations_notnan]  # [relation_mask.sum(), num_relations]

        if train_toks == "tail":
            relations_masked = relations[relation_mask][:, dep_indices]  # [relation_mask.sum(), num_relations]
            relations_masked = relations_masked[relations_notnan]
        else:
            if train_toks == "head":
                # integer tensor, where each int is the index of the head token for the relation
                head_idxs = batch["head_idxs"].to(device)  # size [batch, max_heads, num_relations]

                # Get dimensions
                batch_size, max_heads, num_relations = head_idxs.shape

                # Create a binary tensor of same shape as relations [batch, max_tokens, num_relations]
                relations_head = torch.zeros_like(relations)

                # Create batch and relation indices for indexing
                batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1).expand_as(head_idxs)
                rel_idx = torch.arange(num_relations, device=device).view(1, 1, -1).expand_as(head_idxs)

                # Mask out padded/invalid indices
                valid_mask = head_idxs >= 0

                # Set 1s at head positions for each relation
                relations_head[
                    batch_idx[valid_mask],
                    head_idxs[valid_mask].long(),
                    rel_idx[valid_mask]
                ] = 1.0

                # Apply relation mask to get final tensor
                relations_masked = relations_head[relation_mask][:, dep_indices]  # [relation_mask.sum(), num_relations]
                relations_masked = relations_masked[relations_notnan]
            elif train_toks == "last":
                # try to decode all relations present in the batch based only on the last token
                seq_lens = relation_mask.sum(dim=1)  # shape [batch]
                relations_masked = torch.any(relations[relation_mask][:, dep_indices], dim=1).float()  # [batch, num_relations]

                # Get scores/preds at last token position for each sequence
                batch_idx = torch.arange(len(seq_lens), device=device)
                scores_masked = scores[batch_idx, seq_lens - 1]  # [batch, num_relations]
                preds_masked = preds[batch_idx, seq_lens - 1]  # [batch, num_relations]

    # If no valid instances in batch, return zero loss
    if scores_masked.numel() == 0:
        return torch.tensor(0.0, device=device), scores

    if return_type == "loss":
        # Compute masked loss
        loss = criterion(scores_masked, relations_masked)
        return loss, scores
    else:  # return_type == "preds":
        return preds_masked, relations_masked


def train_probe(
    model: UDTransformer,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    layer: int = -1,
    learning_rate: float = 1e-3,
    num_epochs: int = 10,
    train_toks: str = "tail",
    run_name: Optional[str] = None,
    run_group: Optional[str] = None,
    frequent_deps: Optional[list] = None,
    probe_type: str = "multiclass"
) -> Union[DependencyProbe, Dict[str, BinaryDependencyProbe]]:
    """Train dependency probe, either multiclass or multiple binary probes.

    Args:
        model: The transformer model
        train_loader: DataLoader for training data
        dev_loader: DataLoader for validation data
        device: Device to use for computation
        layer: Model layer to probe
        learning_rate: Learning rate for optimizer
        num_epochs: Maximum number of training epochs
        train_toks: Whether to use head or tail of dependency
        run_name: Optional name for wandb run
        run_group: Optional group for wandb runs
        frequent_deps: List of dependency types to evaluate on
        probe_type: Either "multiclass" or "binary"

    Returns:
        Either a single DependencyProbe (multiclass) or a dictionary of BinaryDependencyProbes (binary)
    """
    print(f"Training {probe_type} probe(s) on device: {device}")

    # Get list of dependencies
    dep_list = frequent_deps if frequent_deps is not None else list(DependencyTask.dependency_table().keys())

    # Initialize models based on approach
    hidden_dim = model.model.cfg.d_model
    num_relations = len(dep_list)

    # Helper function for the training loop (common to both approaches)
    def train_single_probe(
        probe: Union[DependencyProbe, BinaryDependencyProbe],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        is_binary: bool = False,
        dep_idx: Optional[int] = None,
        dep_name: Optional[str] = None
    ) -> Union[DependencyProbe, BinaryDependencyProbe]:
        """Train a single probe (either multiclass or binary)."""
        # Track best model
        best_dev_loss = float('inf')
        best_epoch = -1
        best_state = None
        patience = 3
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            probe.train()
            train_losses = []

            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                if is_binary and dep_idx is not None:
                    # Binary classification for specific dependency
                    loss, _ = compute_loss(
                        probe, model, batch, layer, device,
                        train_toks=train_toks,
                        return_type="loss",
                        binary=True,
                        dep_idx=dep_idx
                    )
                else:
                    # Multiclass classification
                    loss, _ = compute_loss(
                        probe, model, batch, layer, device,
                        train_toks=train_toks, frequent_deps=frequent_deps,
                        return_type="loss"
                    )

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            # Validation
            probe.eval()
            dev_losses = []

            with torch.no_grad():
                for batch in dev_loader:
                    if is_binary and dep_idx is not None:
                        # Binary classification for specific dependency
                        loss, _ = compute_loss(
                            probe, model, batch, layer, device,
                            train_toks=train_toks,
                            return_type="loss",
                            binary=True,
                            dep_idx=dep_idx
                        )
                    else:
                        # Multiclass classification
                        loss, _ = compute_loss(
                            probe, model, batch, layer, device,
                            train_toks=train_toks, frequent_deps=frequent_deps,
                            return_type="loss"
                        )
                    dev_losses.append(loss.item())

            train_loss = sum(train_losses) / len(train_losses)
            dev_loss = sum(dev_losses) / len(dev_losses)

            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "epoch": epoch + 1
            }
            wandb.log(metrics)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")

            # Save best model
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(probe.state_dict())
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"No improvement for {patience} epochs. Early stopping.")
                break

            # Update learning rate
            scheduler.step(dev_loss)

        print(f"Best epoch: {best_epoch+1}, Best dev loss: {best_dev_loss:.4f}")

        # Load best model
        if best_state is not None:
            probe.load_state_dict(best_state)

        return probe

    if probe_type == "multiclass":
        # Single multiclass probe
        wandb.init(
            project="dependency-probing",
            group=run_group,
            name=run_name or f"multiclass_layer_{layer}_probe",
            config={
                "layer": layer,
                "train_toks": train_toks,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "model": model.model.cfg.model_name,
                "hidden_dim": hidden_dim,
                "probe_type": probe_type
            }
        )

        probe = DependencyProbe(hidden_dim, num_relations).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

        print(f"\nTraining multiclass probe for layer {layer}")
        trained_probe = train_single_probe(probe, optimizer, scheduler, is_binary=False)

        wandb.finish()
        # Explicit type conversion for clarity
        return trained_probe  # type: ignore

    elif probe_type == "binary":
        # Multiple binary probes, one per dependency relation
        probes: Dict[str, BinaryDependencyProbe] = {}

        for dep_idx, dep_name in enumerate(dep_list):
            print(f"\nTraining binary probe for dependency: {dep_name} ({dep_idx+1}/{len(dep_list)})")

            # Initialize wandb
            wandb.init(
                project="binary-dependency-probing",
                group=run_group,
                name=run_name or f"binary_layer_{layer}_dep_{dep_name}",
                config={
                    "layer": layer,
                    "dependency": dep_name,
                    "train_toks": train_toks,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "model": model.model.cfg.model_name,
                    "hidden_dim": hidden_dim,
                    "probe_type": probe_type
                }
            )

            # Initialize model and optimizer
            binary_probe = BinaryDependencyProbe(hidden_dim).to(device)
            optimizer = torch.optim.Adam(binary_probe.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

            # Train probe
            trained_binary_probe = train_single_probe(
                binary_probe, optimizer, scheduler, is_binary=True, dep_idx=dep_idx, dep_name=dep_name
            )

            # Store the trained probe
            if isinstance(trained_binary_probe, BinaryDependencyProbe):
                probes[dep_name] = trained_binary_probe

            wandb.finish()

        # Final cleanup
        print(f"\nTrained {len(probes)}/{len(dep_list)} binary probes")
        return probes

    else:
        raise ValueError(f"Invalid probe_type: {probe_type}. Must be 'multiclass' or 'binary'.")
