"""Contains classes for extracting dependency relations from UD data."""

from typing import Dict, Tuple

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
            "acl": 0,
            "acl:relcl": 1,
            "advcl": 2,
            "advcl:relcl": 3,
            "advmod": 4,
            "amod": 5,
            "appos": 6,
            "appos:nmod": 7,
            "aux": 8,
            "aux:pass": 9,
            "case": 10,
            "cc": 11,
            "cc:preconj": 12,
            "ccomp": 13,
            "compound": 14,
            "compound:prt": 15,
            "conj": 16,
            "cop": 17,
            "csubj": 18,
            "csubj:outer": 19,
            "csubj:pass": 20,
            "dep": 21,
            "det": 22,
            "det:predet": 23,
            "discourse": 24,
            "dislocated": 25,
            "expl": 26,
            "fixed": 27,
            "flat": 28,
            "goeswith": 29,
            "iobj": 30,
            "list": 31,
            "mark": 32,
            "nmod": 33,
            "nmod:desc": 34,
            "nmod:poss": 35,
            "nmod:unmarked": 36,
            "nsubj": 37,
            "nsubj:outer": 38,
            "nsubj:pass": 39,
            "nummod": 40,
            "obj": 41,
            "obl": 42,
            "obl:agent": 43,
            "obl:unmarked": 44,
            "orphan": 45,
            "parataxis": 46,
            "punct": 47,
            "ref": 48,
            "reparandum": 49,
            "root": 50,
            "vocative": 51,
            "xcomp": 52,
        }
        return dependency_table


    @staticmethod
    def relations(sentence: UDSentence) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates multi-hot encoded tensor of dependency relations for each word.

        Args:
            sentence: UDSentence object containing tokens and deps fields
            train_toks: "tail" or "head" to indicate which tokens to use as the target for training
        Returns:
            Tensor of shape (sequence_length, num_relations) where each row is a
            multi-hot vector indicating which dependency relations apply to that word
        """
        dep_table = DependencyTask.dependency_table()
        num_relations = len(dep_table)
        seq_length = len(sentence.tokens)

        # Initialize tensor of zeros with actual sequence length
        relations = torch.zeros((seq_length, num_relations), dtype=torch.float)
        head_idxs = torch.ones((seq_length, num_relations), dtype=torch.int) * -1  # -1 will be masked out later
        # For each word position
        for i, deps in enumerate(sentence.deps):
            if deps is not None:
                # Set corresponding indices to 1 in the multi-hot vector
                for rel, head_num in deps: # deps is already parsed
                    if rel not in dep_table:
                        if ':' in rel:
                            # certain relations show specific entity, e.g. "nmod:of". Remove the last part (allowing for multiple preceding colons, potentially)
                            rel = rel.rsplit(':', 1)[0]
                        if rel == '_':
                            continue
                    if rel in dep_table:
                        rel_idx = dep_table[rel]
                        relations[i, rel_idx] = 1
                        if head_num != 0:
                            head_idxs[i, rel_idx] = sentence.ids.index(head_num)
                    else:
                        print(f"Relation {rel} not found in dependency table")

        return relations, head_idxs
