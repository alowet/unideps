"""Linear binary probes for dependency parsing - one probe per dependency relation"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import wandb
from model_utils import UDTransformer
from task import DependencyTask
from torch.utils.data import DataLoader
from tqdm import tqdm


class BinaryDependencyProbe(nn.Module):
    """Binary probe for predicting a single dependency relation at each position."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.probe = nn.Linear(input_dim, 1)  # Binary output

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Maps token embeddings to a single dependency relation logit."""
        return self.probe(embeddings)


def compute_binary_loss(
    probe: BinaryDependencyProbe,
    model: UDTransformer,
    batch: Dict,
    layer: int,
    device: torch.device,
    dep_idx: int,
    train_toks: str = "tail",
    criterion: nn.Module = nn.BCEWithLogitsLoss(reduction='mean'),
    return_type: str = "loss",
    frequent_deps: Optional[list] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute loss for a single batch and a single dependency relation.

    Args:
        probe: The binary probe model for a specific dependency relation
        model: The transformer model
        batch: Batch of data
        layer: Layer to probe
        device: Device to use
        dep_idx: Index of the dependency relation this probe is for (relative to frequent_deps)
        train_toks: Whether to use head or tail of dependency, or some other token entirely
        criterion: Loss function
        return_type: Whether to return loss (training/dev) or predictions (testing)
        frequent_deps: Set of dependency relations that appear frequently enough

    Returns:
        loss: The computed loss
        scores: The probe's predictions
    """
    # Get activations and compute scores
    activations = model.get_activations(batch, layer_idx=layer, train_toks=train_toks)  # size [batch, max_len, hidden_dim]
    scores = probe(activations)  # size [batch, max_len, 1]
    preds = torch.sigmoid(scores) > 0.5  # sigmoid since we're using BCE loss

    # Get labels and mask
    relations = batch["relations"].to(device)  # size [batch, max_tokens, num_relations]
    relation_mask = batch["relation_mask"].to(device)  # size [batch, max_tokens]

    dep_table = DependencyTask.dependency_table()
    dep_indices = [dep_table[dep] for dep in frequent_deps] if frequent_deps is not None else list(dep_table.values())

    # Get the specific dependency relation index for the full relation set
    full_dep_idx = dep_indices[dep_idx]

    # Extract binary labels for this dependency relation
    relations_notnan = ~(relations.isnan()[relation_mask][:, dep_indices].any(1))
    scores_masked = scores.squeeze(-1)[relation_mask][relations_notnan]  # [relation_mask.sum()]
    preds_masked = preds.squeeze(-1)[relation_mask][relations_notnan]  # [relation_mask.sum()]

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
            scores_masked = scores.squeeze(-1)[batch_idx, seq_lens - 1]  # [batch]
            preds_masked = preds.squeeze(-1)[batch_idx, seq_lens - 1]  # [batch]

    # If no valid instances in batch, return zero loss
    if scores_masked.numel() == 0:
        return torch.tensor(0.0, device=device), scores

    if return_type == "loss":
        # Compute masked loss
        loss = criterion(scores_masked, relations_masked)
        return loss, scores
    else:  # return_type == "preds":
        return preds_masked, relations_masked


def train_binary_probes(
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
    frequent_deps: Optional[list] = None
) -> Dict[str, BinaryDependencyProbe]:
    """Train binary dependency probes - one per dependency relation.

    Returns a dictionary mapping dependency names to trained probes.
    """
    print(f"Training binary probes for layer {layer} on device: {device}")

    # Get list of dependency relations to train probes for
    dep_list = frequent_deps if frequent_deps is not None else list(DependencyTask.dependency_table().keys())

    # Dictionary to store probes
    probes = {}

    # Train a separate probe for each dependency relation
    for dep_idx, dep_name in enumerate(dep_list):
        print(f"\nTraining probe for dependency: {dep_name} ({dep_idx+1}/{len(dep_list)})")

        # Initialize wandb
        # wandb.init(
        #     project="binary-dependency-probing",
        #     group=run_group,
        #     name=run_name or f"layer_{layer}_dep_{dep_name}",
        #     config={
        #         "layer": layer,
        #         "dependency": dep_name,
        #         "train_toks": train_toks,
        #         "learning_rate": learning_rate,
        #         "num_epochs": num_epochs,
        #         "model": model.model.cfg.model_name,
        #         "hidden_dim": model.model.cfg.d_model,
        #     }
        # )

        # Initialize model and optimizer
        hidden_dim = model.model.cfg.d_model
        probe = BinaryDependencyProbe(hidden_dim).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # Track best model
        best_dev_loss = float('inf')
        best_epoch = -1
        best_state = None

        for epoch in range(num_epochs):
            # Training
            probe.train()
            total_loss = 0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()

                # Compute loss and update
                loss, _ = compute_binary_loss(
                    probe, model, batch, layer, device,
                    dep_idx=dep_idx,
                    train_toks=train_toks,
                    criterion=criterion,
                    return_type="loss",
                    frequent_deps=frequent_deps
                )
                loss.backward()
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                current_loss = total_loss / num_batches
                progress_bar.set_postfix({'train_loss': current_loss})

                # Log batch metrics
                # wandb.log({
                #     "train/batch_loss": loss.item(),
                #     "train/running_loss": current_loss,
                #     "epoch": epoch,
                #     "batch": batch_idx
                # })

            # Evaluate on dev set
            probe.eval()
            total_loss = 0
            num_batches = 0

            with torch.no_grad():
                for batch in tqdm(dev_loader, desc='Evaluating'):
                    loss, _ = compute_binary_loss(
                        probe, model, batch, layer, device,
                        dep_idx=dep_idx,
                        train_toks=train_toks,
                        criterion=criterion,
                        return_type="loss",
                        frequent_deps=frequent_deps
                    )
                    total_loss += loss.item()
                    num_batches += 1

            dev_loss = total_loss / num_batches
            scheduler.step(dev_loss)

            # Log epoch metrics
            # wandb.log({
            #     "train/epoch_loss": current_loss,
            #     "dev/epoch_loss": dev_loss,
            #     "epoch": epoch
            # })

            print(f"Epoch {epoch+1} - Train loss: {current_loss:.4f}, Dev loss: {dev_loss:.4f}")

            # Save best model
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_epoch = epoch
                best_state = probe.state_dict().copy()
                print(f"New best model! Dev loss: {dev_loss:.4f}")

            # Early stopping check
            if epoch - best_epoch > 3:  # 3 epochs patience
                print("No improvement for 3 epochs, stopping early")
                break

        # Restore best model
        if best_state is not None:
            probe.load_state_dict(best_state)

        # Store the probe
        probes[dep_name] = probe.cpu()

        # Finish wandb run
        # wandb.finish()

    return probes
