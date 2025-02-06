"""Utilities for data loading and processing"""

import torch
from task import DependencyTask


def collate_fn(batch, train_toks: str = "tail"):
    """Collate batch of examples into padded tensors.

    Args:
        batch: List of UDSentence objects
        train_toks: "tail" or "head" to indicate which tokens to use as the target for training
    Returns:
        Dictionary containing:
            - relation_mask: torch.Tensor - mask for relations [batch_size, max_tokens]
            - head_mask: torch.Tensor - mask for heads [batch_size, max_heads]
            - relations: torch.Tensor - padded relation tensors [batch_size, max_tokens, num_relations]
            - head_nums: torch.Tensor - padded head number tensors [batch_size, max_heads, num_relations]
            - sentences: List of UDSentence objects
    """
    # Get max length in this batch
    max_tokens = max(len(sent.tokens) for sent in batch)
    # need to add the else 0 for the case where there's only ever (max) one head per relation,
    # but there are some where head is root, otherwise seq_length can exceed max_heads and that's a problem
    max_heads = max(len([x for lst in sent.deps for x in (lst if lst else [0])]) for sent in batch)

    # Create label tensors with padding
    tensors_info = {
        'relations': {'max_len': max_tokens, 'tensors': []},
        'head_nums': {'max_len': max_heads, 'tensors': []}
    }
    masks = {
        'relations': torch.ones((len(batch), max_tokens), dtype=torch.bool),
        'head_nums': torch.ones((len(batch), max_heads), dtype=torch.bool)
    }

    for i, sent in enumerate(batch):
        relation, head_num = DependencyTask.relations(sent)
        for name, tensor in [('relations', relation), ('head_nums', head_num)]:
            info = tensors_info[name]
            pad_len = info['max_len'] - len(tensor)
            if pad_len > 0:
                padding = torch.zeros((pad_len, tensor.size(1)), dtype=tensor.dtype)
                tensor = torch.cat([tensor, padding], dim=0)
                masks[name][i, len(tensor)-pad_len:] = False
            info['tensors'].append(tensor)

    return {
        "relation_mask": masks['relations'],
        "head_mask": masks['head_nums'],
        "relations": torch.stack(tensors_info['relations']['tensors']),
        "head_nums": torch.stack(tensors_info['head_nums']['tensors']),
        "max_tokens": max_tokens,
        "max_heads": max_heads,
        "sentences": batch
    }
