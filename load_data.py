"""Functions for loading and processing UD_English_EWT data in CoNLL-U format"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from conllu import parse_incr
from torch.utils.data import Dataset


@dataclass
class UDSentence:
    """Holds data for a single sentence from UD dataset"""
    tokens: List[str]  # Words/tokens
    ids: List[str]    # Token IDs (can be non-sequential due to multi-word tokens)
    deps: List[List[Tuple[str, int]]]   # Enhanced dependencies: list of (relation, head_idx) pairs per token
    upos: List[str]   # Universal POS tags
    text: str         # Original text

class UDDataset(Dataset):
    def __init__(self, filepath: str, max_length: int = 256, max_sentences: int = None):
        self.sentences = []
        self.max_length = max_length
        self.max_sentences = max_sentences

        # Load CoNLL-U file
        with open(filepath, "r", encoding="utf-8") as f:
            for tokenlist in parse_incr(f):
                if len(tokenlist) > max_length:
                    # continue
                    raise ValueError(f"Sentence length {len(tokenlist)} exceeds max length {max_length}")
                if self.max_sentences is not None and len(self.sentences) >= self.max_sentences:
                    break

                sentence = UDSentence(
                    tokens=[t["form"] for t in tokenlist],
                    ids=[t["id"] for t in tokenlist],
                    deps=[t["deps"] for t in tokenlist],  # formatted as lists of (relation, head_idx) tuples
                    upos=[t["upos"] for t in tokenlist],
                    text=tokenlist.metadata["text"]
                )
                self.sentences.append(sentence)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
