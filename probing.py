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
                criterion: nn.modules.loss._Loss = nn.BCEWithLogitsLoss,
                return_type: str = "loss") -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute loss for a single batch.

    Args:
        probe: The probe model
        model: The transformer model
        batch: Batch of data
        layer: Layer to probe
        criterion: Loss function
        device: Device to use
        train_toks: Whether to use head or tail of dependency
        return_type: Whether to return loss (training/dev) or predictions (testing)
    Returns:
        loss: The computed loss
        scores: The probe's predictions
    """
    # Get activations and compute scores
    activations = model.get_activations(batch, layer_idx=layer, train_toks=train_toks)
    scores = probe(activations)

    # Get labels and mask
    labels = batch["labels"].to(device)
    dep_mask = batch["dep_mask"].to(device)
    labels_masked = labels[dep_mask]

    if return_type == "loss":
        # Compute masked loss
        scores_masked = scores[dep_mask]
        loss = criterion(scores_masked, labels_masked)
        return loss, scores

    else:  # return_type == "preds":
        preds = torch.sigmoid(scores) > 0.5  # sigmoid since we're using BCE loss
        preds_masked = preds[dep_mask]
        return preds_masked, labels_masked


def train_probe(model: UDTransformer,
                train_loader: DataLoader,
                dev_loader: DataLoader,
                layer: int = -1,
                learning_rate: float = 1e-3,
                num_epochs: int = 10,
                train_toks: str = "tail",
                run_name: Optional[str] = None) -> DependencyProbe:
    """Train dependency probe with integrated evaluation."""

    # Initialize wandb
    wandb.init(
        project="dependency-probing",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    # Initialize models and optimizer
    hidden_dim = model.model.cfg.d_model
    num_relations = len(DependencyTask.dependency_table())
    probe = DependencyProbe(hidden_dim, num_relations).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

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
            loss, _ = compute_loss(probe, model, batch, layer, device, train_toks=train_toks, criterion=criterion, return_type="loss")
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
                loss, _ = compute_loss(probe, model, batch, layer, device, train_toks=train_toks, criterion=criterion, return_type="loss")
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
