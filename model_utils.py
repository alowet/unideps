from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import login
from load_data import UDSentence

# from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import HookedSAETransformer
from tqdm import tqdm


class UDTransformer:
    def __init__(self, model_name: str = "gemma-2-2b"):
        """Initialize the UDTransformer.

        Args:
            model_name: Name of the model to use from HuggingFace
        """

        with open('/n/home06/alowet/.cache/huggingface/token', 'r') as f:
            token = f.read()
        login(token)
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        self.model = HookedSAETransformer.from_pretrained(model_name, device=device)

    def get_token_masks(self, sentences: List[UDSentence], do_print: bool = False, train_toks: str = "tail"):
        """Get token masks for sentences.

        Args:
            sentences: List of UDSentence objects
            do_print: Whether to print validation examples
            train_toks: "tail" or "head": whether to use the tail of the dependency relation or the head. Right now only tail is supported.
        Returns:
            List of token masks
        """
        # Now we use the provided text directly
        sentence_texts = [sent.text for sent in sentences]
        model_tokens = self.model.to_str_tokens(sentence_texts)

        final_model_token_masks = []
        validation_examples = []  # Store examples for validation

        for sent_idx, (sent, model_tokens_sent) in enumerate(zip(sentences, model_tokens)):

            final_model_token_mask = torch.zeros(len(model_tokens_sent), dtype=torch.bool)

            # Skip special tokens at start
            start_idx = 1  # if str_tokens[0] == '<|endoftext|>' else 0

            current_word_idx = 0
            conllu_tokens = sent.tokens  # Get the tokens from the sentence data
            current_word = conllu_tokens[current_word_idx]
            current_token_start = start_idx

            # Store validation info for this sentence
            validation_info = {
                'original_sentence': sent.text,
                'model_tokens': model_tokens_sent,
                'word_to_final_token': {},
                'reconstructed_words': []
            }

            for model_token_idx in range(start_idx, len(model_tokens_sent)):

                # Get the text from current_token_start to current token
                current_span = ''.join(model_tokens_sent[current_token_start:model_token_idx + 1])
                current_span = current_span.strip()

                if current_span == current_word:

                    final_model_token_mask[model_token_idx] = True

                    # Store validation info
                    validation_info['word_to_final_token'][current_word] = {
                        'final_token': model_tokens_sent[model_token_idx],
                        'all_tokens': model_tokens_sent[current_token_start:model_token_idx + 1]
                    }
                    validation_info['reconstructed_words'].append(current_span)

                    current_word_idx += 1

                    if current_word_idx < len(conllu_tokens):
                        current_word = conllu_tokens[current_word_idx]
                        current_token_start = model_token_idx + 1

            final_model_token_masks.append(final_model_token_mask)

            # Store first 5 sentences for validation
            if sent_idx < 5:
                validation_examples.append(validation_info)

        if do_print:
        # Print validation examples
            print("\nValidation Examples:")
            for idx, example in enumerate(validation_examples):
                print(f"\nExample {idx + 1}:")
                print(f"Original sentence: {example['original_sentence']}")
                print("Model tokens:", example['model_tokens'])
                print("\nWord to token mapping:")
                for word, token_info in example['word_to_final_token'].items():
                    print(f"Word: '{word}'")
                    print(f"  Final token: '{token_info['final_token']}'")
                    print(f"  All tokens: {token_info['all_tokens']}")
                print(f"Reconstructed sentence: {' '.join(example['reconstructed_words'])}")
                print("-" * 80)

        return final_model_token_masks


    def get_activations(self, batch: Dict, layer_idx: int, train_toks: str = "tail") -> torch.Tensor:
        """Get aligned token-level embeddings.

        Args:
            batch: Batch of token sequences
            layer_idx: Layer index to get embeddings from

        Returns:
            Tensor of shape [batch, seq_len, hidden_dim] containing embeddings
            aligned with the original tokens
        """

        # Get encodings for all sentences
        token_masks = self.get_token_masks(batch["sentences"], train_toks=train_toks)

        # Run model with cache, but only store the layer we want
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                [sent.text for sent in batch["sentences"]],
                stop_at_layer=layer_idx+1,
                names_filter=f"blocks.{layer_idx}.hook_resid_pre"
            )

        # Get states for the target layer
        states = cache[f"blocks.{layer_idx}.hook_resid_pre"]  # size [batch, seq_len, hidden_dim]

        # Clear cache explicitly
        del cache
        torch.cuda.empty_cache()

        # Pad to max sequence length
        trimmed_padded_states = torch.zeros(
            states.size(0), batch["max_len"], states.size(2),
            device=states.device
        )
        for i_sent, token_mask in enumerate(token_masks):
            trimmed_padded_states[i_sent, :token_mask.sum()] = states[i_sent, :token_mask.size(0)][token_mask]

        return trimmed_padded_states
