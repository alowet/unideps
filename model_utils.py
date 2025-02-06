from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import login
from load_data import UDSentence
from tqdm import tqdm

# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer


class UDTransformer:
    def __init__(self, model_name: str, device: torch.device):
        """Initialize the UDTransformer.

        Args:
            model_name: Name of the model to use from HuggingFace
            device: Device to load the model on
        """
        # Read HuggingFace token
        with open('/n/home06/alowet/.cache/huggingface/token', 'r') as f:
            token = f.read()
        login(token)

        print(f"Loading {model_name} on device: {device}")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)

    def get_token_masks(self, sentences: List[UDSentence], do_print: bool = False, train_toks: str = "tail"):
        """Get token masks for sentences.

        Args:
            sentences: List of UDSentence objects
            do_print: Whether to print validation examples
            train_toks: Which tokens to use for training:
                - "tail": use tail token of dependency (default)
                - "head": use head token of dependency
                - "first": use first token in sentence
                - "last": use last token in sentence
        Returns:
            List of token masks
        """
        # Now we use the provided text directly
        sentence_texts = [sent.text for sent in sentences]
        model_tokens = self.model.to_str_tokens(sentence_texts)

        final_model_token_masks = []
        validation_examples = []

        for sent_idx, (sent, model_tokens_sent) in enumerate(zip(sentences, model_tokens)):
            # Map conllu tokens to model tokens
            token_mapping = {}  # conllu_idx -> model_token_idx
            final_model_token_mask = torch.zeros(len(model_tokens_sent), dtype=torch.bool)
            current_word_idx = 0
            current_token_start = 1  # Skip BOS token
            current_word = sent.tokens[current_word_idx]

            for model_token_idx in range(1, len(model_tokens_sent)):
                current_span = ''.join(model_tokens_sent[current_token_start:model_token_idx + 1]).strip()
                if current_span == current_word:
                    final_model_token_mask[model_token_idx] = True
                    token_mapping[current_word_idx] = current_word_idx + 1
                    current_word_idx += 1

                    if current_word_idx < len(sent.tokens):
                        current_word = sent.tokens[current_word_idx]
                        current_token_start = model_token_idx + 1

            # Create mask based on train_toks
            final_model_token_masks.append(final_model_token_mask)

            # Add validation info
            if sent_idx < 10 and do_print:
                validation_info = {
                    'original_sentence': sent.text,
                    'model_tokens': model_tokens_sent,
                    'selected_tokens': [
                        (i, tok) for i, tok in enumerate(model_tokens_sent)
                        if final_model_token_mask[i]
                    ],
                    'train_toks': train_toks
                }
                validation_examples.append(validation_info)

        if do_print:
            print("\nValidation Examples:")
            for idx, example in enumerate(validation_examples):
                print(f"\nExample {idx + 1}:")
                print(f"Original sentence: {example['original_sentence']}")
                print(f"Training mode: {example['train_toks']}")
                print("Selected tokens:")
                for i, tok in example['selected_tokens']:
                    print(f"  Position {i}: '{tok}'")
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
        token_masks = self.get_token_masks(batch["sentences"], train_toks=train_toks, do_print=True)

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

        # the way "labels" is unpacked is decided in collate_fn and is different for head and tail modes
        for i_sent, token_mask in enumerate(token_masks):
            # each token mask should be an integer tensor of size [num_tokens]
            trimmed_padded_states[i_sent, :token_mask.sum()] = states[i_sent, :token_mask.size(0)][token_mask]

        return trimmed_padded_states
