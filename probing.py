"""Linear probes for dependency parsing"""

from typing import Dict, List, Optional

import numpy as np
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
        """Maps token embeddings to dependency relation logits.

        Args:
            embeddings: Token embeddings [batch_size, seq_len, hidden_dim]

        Returns:
            logits: Scores for each dependency relation at each position
                   [batch_size, seq_len, num_relations]
                   Higher values indicate higher probability of that relation
                   applying to that token position
        """
        # Project from hidden_dim to num_relations
        # For each token, get a score for each possible dependency relation
        logits = self.probe(embeddings)  # [batch_size, seq_len, num_relations]
        return logits

# Modified BCE loss that handles padding
def masked_bce_loss(scores, labels, dep_mask, criterion):
    """Compute BCE loss only on actual tokens (not padding)."""
    num_relations = scores.size(2)

    # Apply mask to both scores and labels
    scores_masked = scores[dep_mask].reshape(-1, num_relations)
    labels_masked = labels[dep_mask].reshape(-1, num_relations)

    return criterion(scores_masked, labels_masked)

def train_probe(model: UDTransformer,
                train_loader: DataLoader,
                dev_loader: DataLoader,
                layer: int = -1,
                learning_rate: float = 1e-3,
                num_epochs: int = 10,
                run_name: Optional[str] = None):
    """Train dependency probe"""

    # Initialize wandb
    wandb.init(
        project="dependency-probing",
        name=run_name or f"layer_{layer}_probe",
        config={
            "layer": layer,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "model": model.model.cfg.model_name,
            "hidden_dim": model.model.cfg.d_model,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    hidden_dim = model.model.cfg.d_model
    num_relations = len(DependencyTask.dependency_table())

    probe = DependencyProbe(hidden_dim, num_relations).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    print(f"\nTraining probe for layer {layer}")
    for epoch in range(num_epochs):

        probe.train()
        total_loss = 0
        num_batches = 0

        # Training loop
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(progress_bar):
            # Clear cache before getting new activations
            torch.cuda.empty_cache()

            activations = model.get_activations(batch, layer_idx=layer)
            scores = probe(activations)

            # Free memory
            del activations

            labels = batch["labels"].to(device)
            dep_mask = batch["dep_mask"].to(device)
            loss = masked_bce_loss(scores, labels, dep_mask, criterion)

            # Free more memory
            del scores

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            current_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': current_loss})

            wandb.log({
                "train/batch_loss": loss.item(),
                "train/running_loss": current_loss,
                "epoch": epoch,
                "batch": batch_idx
            })

            avg_train_loss = total_loss / num_batches
            # scheduler.step(avg_train_loss)
            print(f"Epoch {epoch+1} - Avg train loss: {current_loss:.4f}")

    wandb.finish()
    return probe

def evaluate_probe(model: UDTransformer,
                   probe: DependencyProbe,
                   dev_loader: DataLoader,
                   layer: int = -1):
    """Evaluate probe"""
    # Evaluate on dev set

    probe.eval()
    dev_loss = 0
    num_dev_batches = 0
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    print("\nEvaluating on dev set...")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    with torch.no_grad():
        for dev_batch_idx, batch in enumerate(tqdm(dev_loader)):
            activations = model.get_activations(batch, layer_idx=layer)

            labels = batch["labels"].to(device)
            dep_mask = batch["dep_mask"].to(device)
            scores = probe(activations)
            loss = masked_bce_loss(scores, labels, dep_mask, criterion)
            dev_loss += loss.item()
            num_dev_batches += 1

    avg_dev_loss = dev_loss / num_dev_batches

    print(f"Avg dev loss: {avg_dev_loss:.4f}")
