"""Utilities for data loading and processing"""

import torch
from task import DependencyTask


def collate_fn(batch):
    """Collate batch of examples into padded tensors.

    Args:
        batch: List of UDSentence objects

    Returns:
        Dictionary containing:
            - dep_mask: torch.Tensor - mask for dependency relations [batch_size, max_len]
            - labels: torch.Tensor - padded label tensors [batch_size, max_len, num_relations]
            - sentences: List of UDSentence objects
    """
    # Get max length in this batch
    max_len = max(len(sent.tokens) for sent in batch)

    # Create label tensors with padding
    labels = []
    dep_mask = torch.ones((len(batch), max_len), dtype=torch.bool)
    for i, sent in enumerate(batch):
        label = DependencyTask.labels(sent)
        # Pad to max length
        pad_len = max_len - len(sent.tokens)
        if pad_len > 0:
            padding = torch.zeros((pad_len, label.size(1)), dtype=label.dtype)
            label = torch.cat([label, padding], dim=0)
        labels.append(label)
        dep_mask[i, pad_len:] = False
    return {
        "dep_mask": dep_mask,
        "labels": torch.stack(labels),  # Now [batch_size, max_len, num_relations]
        "sentences": batch,
        "max_len": max_len
    }
