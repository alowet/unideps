"""Contains classes for extracting dependency relations from UD data."""

from typing import Dict, List, Optional

import numpy as np
import torch
from load_data import UDSentence


class DependencyTask:
    """Maps UD sentences to dependency relation labels."""

    @staticmethod
    def dependency_table() -> Dict[str, int]:
        """Returns mapping of dependency relation names to integers.
        Based on Universal Dependencies v2.
        """
        dependency_table = {
            "acl": 1,
            "acl:relcl": 2,
            "advcl": 3,
            "advcl:relcl": 4,
            "advmod": 5,
            "amod": 6,
            "appos": 7,
            "appos:nmod": 8,
            "aux": 9,
            "aux:pass": 10,
            "case": 11,
            "cc": 12,
            "cc:preconj": 13,
            "ccomp": 14,
            "compound": 15,
            "compound:prt": 16,
            "conj": 17,
            "cop": 18,
            "csubj": 19,
            "csubj:outer": 20,
            "csubj:pass": 21,
            "dep": 22,
            "det": 23,
            "det:predet": 24,
            "discourse": 25,
            "dislocated": 26,
            "expl": 27,
            "fixed": 28,
            "flat": 29,
            "goeswith": 30,
            "iobj": 31,
            "list": 32,
            "mark": 33,
            "nmod": 34,
            "nmod:desc": 35,
            "nmod:poss": 36,
            "nmod:unmarked": 37,
            "nsubj": 38,
            "nsubj:outer": 39,
            "nsubj:pass": 40,
            "nummod": 41,
            "obj": 42,
            "obl": 43,
            "obl:agent": 44,
            "obl:unmarked": 45,
            "orphan": 46,
            "parataxis": 47,
            "punct": 48,
            "reparandum": 49,
            "root": 50,
            "vocative": 51,
            "xcomp": 52,
            "ref": 53
        }
        return dependency_table


    @staticmethod
    def labels(sentence: UDSentence) -> torch.Tensor:
        """Creates multi-hot encoded tensor of dependency relations for each word.

        Args:
            sentence: UDSentence object containing tokens and deps fields

        Returns:
            Tensor of shape (sequence_length, num_relations) where each row is a
            multi-hot vector indicating which dependency relations apply to that word
        """
        dep_table = DependencyTask.dependency_table()
        num_relations = len(dep_table)
        seq_length = len(sentence.tokens)

        # Initialize tensor of zeros with actual sequence length
        relations = torch.zeros((seq_length, num_relations), dtype=torch.float)

        # For each word position
        for i, deps in enumerate(sentence.deps):
            # Get list of relations for this word
            word_relations = [x[0] for x in deps]  # deps is already parsed

            # Set corresponding indices to 1 in the multi-hot vector
            for rel in word_relations:
                if rel not in dep_table:
                    if ':' in rel:
                        # certain relations show specific entity, e.g. "nmod:of". Remove the last part (allowing for multiple preceding colons, potentially)
                        rel = rel.rsplit(':', 1)[0]
                    if rel == '_':
                        continue
                if rel in dep_table:
                    rel_idx = dep_table[rel]
                    relations[i, rel_idx-1] = 1  # -1 because indices start at 1
                else:
                    print(f"Relation {rel} not found in dependency table")

        return relations
