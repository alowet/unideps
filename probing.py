"""Linear probes for dependency parsing"""

from typing import Dict, Optional, Tuple

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


def compute_loss(probe: DependencyProbe,
                model: UDTransformer,
                batch: Dict,
                layer: int,
                device: torch.device,
                train_toks: str = "tail",
                criterion: nn.Module = nn.BCEWithLogitsLoss(reduction='mean'),
                return_type: str = "loss",
                frequent_deps: Optional[list] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute loss for a single batch.

    Args:
        probe: The probe model
        model: The transformer model
        batch: Batch of data
        layer: Layer to probe
        criterion: Loss function
        device: Device to use
        train_toks: Whether to use head or tail of dependency, or some other token entirely (e.g. first or last)
        return_type: Whether to return loss (training/dev) or predictions (testing)
        frequent_deps: Set of indices for dependency relations that appear frequently enough
    Returns:
        loss: The computed loss
        scores: The probe's predictions
    """
    # Get activations and compute scores
    activations = model.get_activations(batch, layer_idx=layer, train_toks=train_toks)  # size [batch, max_len, hidden_dim]
    scores = probe(activations)  # size [batch, max_len, num_relations], for each token, get a predicted relation logit
    preds = torch.sigmoid(scores) > 0.5  # sigmoid since we're using BCE loss

    # Get labels and mask
    relations = batch["relations"].to(device)  # size [batch, max_tokens, num_relations]
    relation_mask = batch["relation_mask"].to(device)  # size [batch, max_tokens]

    dep_table = DependencyTask.dependency_table()
    dep_indices = [dep_table[dep] for dep in frequent_deps] if frequent_deps is not None else list(dep_table.values())

    relations_notnan = ~(relations.isnan()[relation_mask][:, dep_indices].any(1))  # shape [relation_mask.sum(), num_relations]
    scores_masked = scores[relation_mask][relations_notnan]  # shape [relation_mask.sum(), num_relations], scores for all relations that exist at each token position in the batch
    preds_masked = preds[relation_mask][relations_notnan]  # shape [relation_mask.sum(), num_relations], predictions for all relations that exist at each token position in the batch

    if train_toks == "tail":
        relations_masked = relations[relation_mask][:, dep_indices]  # shape [relation_mask.sum(), num_relations]
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
            relations_masked = relations_head[relation_mask][:, dep_indices]  # shape [relation_mask.sum(), num_relations]

        elif train_toks == "last":
            # try to decode all relations present in the batch based only on the last token; ignore all the other tokens in the sentence
            seq_lens = relation_mask.sum(dim=1)  # shape [batch]
            relations_masked = torch.any(relations[relation_mask][:, dep_indices], dim=1).float()  # shape [batch, num_relations]
            # Get scores/preds at last token position for each sequence
            batch_idx = torch.arange(len(seq_lens), device=device)
            scores_masked = scores[batch_idx, seq_lens - 1]  # shape [batch, num_relations]
            preds_masked = preds[batch_idx, seq_lens - 1]  # shape [batch, num_relations]

    # If no frequent dependencies in batch, return zero loss
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
    frequent_deps: Optional[list] = None
) -> DependencyProbe:
    """Train dependency probe."""

    print(f"Training probe on device: {device}")

    # Initialize wandb
    wandb.init(
        project="dependency-probing",
        group=run_group,
        name=run_name or f"layer_{layer}_probe",
        config={
            "layer": layer,
            "train_toks": train_toks,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "model": model.model.cfg.model_name,
            "hidden_dim": model.model.cfg.d_model,
        }
    )

    # Initialize models and optimizer
    hidden_dim = model.model.cfg.d_model
    num_relations = len(frequent_deps) if frequent_deps is not None else len(DependencyTask.dependency_table())
    probe = DependencyProbe(hidden_dim, num_relations).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

    # Track best model
    best_dev_loss = float('inf')
    best_epoch = -1
    best_state = None

    print(f"\nTraining probe for layer {layer}")
    for epoch in range(num_epochs):
        # Training
        probe.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # Compute loss and update
            loss, _ = compute_loss(probe, model, batch, layer, device, train_toks=train_toks, return_type="loss", frequent_deps=frequent_deps)
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            current_loss = total_loss / num_batches
            progress_bar.set_postfix({'train_loss': current_loss})

            # Log batch metrics
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/running_loss": current_loss,
                "epoch": epoch,
                "batch": batch_idx
            })

        # Evaluate on dev set
        probe.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dev_loader, desc='Evaluating'):
                loss, _ = compute_loss(probe, model, batch, layer, device, train_toks=train_toks, return_type="loss", frequent_deps=frequent_deps)
                total_loss += loss.item()
                num_batches += 1

        dev_loss = total_loss / num_batches
        scheduler.step(dev_loss)
        # Log epoch metrics
        wandb.log({
            "train/epoch_loss": current_loss,
            "dev/epoch_loss": dev_loss,
            "epoch": epoch
        })

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
    wandb.finish()

    return probe
