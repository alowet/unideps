import os
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import login
from load_data import UDSentence

# from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import HookedSAETransformer
from tqdm import tqdm


class UDTransformer:
    def __init__(self, model_name: str, device: torch.device):
        """Initialize the UDTransformer.

        Args:
            model_name: Name of the model to use from HuggingFace
            device: Device to load the model on
        """
        # Read HuggingFace token
        with open(os.path.join(os.environ["HF_HOME"], 'token'), 'r') as f:
            token = f.read()
        login(token)

        print(f"Loading {model_name} on device: {device}")
        self.model = HookedSAETransformer.from_pretrained(model_name, device=device)

    def get_token_masks(self, sentences: List[UDSentence], do_print: bool = False):
        """Get token masks for sentences.

        Args:
            sentences: List of UDSentence objects
            do_print: Whether to print validation examples
            train_toks: Which tokens to use for training
        Returns:
            List of token masks
        """
        sentence_texts = [sent.text for sent in sentences]
        model_tokens = self.model.to_str_tokens(sentence_texts)

        final_model_token_masks = []
        validation_examples = []

        for sent_idx, (sent, model_tokens_sent) in enumerate(zip(sentences, model_tokens)):
            # Map conllu tokens to model tokens
            conllu_to_model_token_mapping = {}  # conllu_idx -> model_token_idx
            current_token_start = 1  # Skip BOS token

            # Filter out multi-word tokens (e.g. "2-3") and get valid token indices
            for i, (token_id, word) in enumerate(zip(sent.ids, sent.tokens)):
                if not isinstance(token_id, int) and '-' in token_id:
                    continue

                # Find this word in model tokens starting from current_token_start
                for model_token_idx in range(current_token_start, len(model_tokens_sent)):
                    span = ''.join(model_tokens_sent[current_token_start:model_token_idx + 1]).strip()
                    if word in span or any([x in span for x in ["'t", "'d", "'ll", "'re", "'ve"]]):
                        conllu_to_model_token_mapping[i] = model_token_idx
                        current_token_start = model_token_idx + 1
                        break

            # For tail mode, use mapped indices directly
            final_model_token_masks.append(list(conllu_to_model_token_mapping.values()))

            # Add validation info
            if sent_idx < 5 and do_print:
                validation_info = {
                    'original_sentence': sent.text,
                    'model_tokens': model_tokens_sent,
                    'conllu_tokens': [
                        f"{id_}:{tok}" for id_, tok in zip(sent.ids, sent.tokens)
                    ],
                    'mapped_tokens': {
                        f"{sent.ids[i]}:{sent.tokens[i]}": model_tokens_sent[model_idx]
                        for i, model_idx in conllu_to_model_token_mapping.items()
                    }
                }
                validation_examples.append(validation_info)

        if do_print:
            print("\nValidation Examples:")
            for idx, example in enumerate(validation_examples):
                print(f"\nExample {idx + 1}:")
                print(f"Original sentence: {example['original_sentence']}")
                print("Token mappings:")
                for conllu_tok, model_tok in example['mapped_tokens'].items():
                    print(f"  CoNLL-U {conllu_tok} -> Model token '{model_tok}'")
                print("-" * 80)

        return final_model_token_masks


    def get_activations(self, batch: Dict, layer_idx: int, train_toks: str = "tail") -> torch.Tensor:
        """Get aligned token-level embeddings.

        Args:
            batch: Batch of token sequences
            layer_idx: Layer index to get embeddings from
            train_toks: Which tokens to use (tail, head, first, last)

        Returns:
            Tensor of shape [batch, seq_len, hidden_dim] containing embeddings
            aligned with the original tokens/dependencies
        """

        # Get encodings for all sentences
        token_masks = self.get_token_masks(batch["sentences"])

        # Run model with cache, but only store the layer we want
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                [sent.text for sent in batch["sentences"]],
                stop_at_layer=layer_idx+1,
                names_filter=f"blocks.{layer_idx}.hook_resid_pre"
            )

        # Get states for the target layer
        states = cache[f"blocks.{layer_idx}.hook_resid_pre"]  # size [batch, seq_len, hidden_dim]

        del cache
        torch.cuda.empty_cache()

        trimmed_padded_states = torch.zeros(
                states.size(0), batch["max_tokens"], states.size(2),
                device=states.device
            )

        for i_sent, token_mask in enumerate(token_masks):
            # each token mask should be an integer tensor of size [num_tokens]
            # trimmed_padded_states[i_sent, :token_mask.sum()] = states[i_sent, :token_mask.size(0)][token_mask]  # if boolean
            trimmed_padded_states[i_sent, :len(token_mask)] = states[i_sent][torch.tensor(token_mask, dtype=torch.int)]  # if integer

        # shape: [batch, max_tokens, hidden_dim], containing the activations for each token in the batch
        return trimmed_padded_states
