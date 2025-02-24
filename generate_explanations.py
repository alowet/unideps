# %%
import importlib
import os
from functools import partial
from typing import cast

import orjson
import torch
from datasets import load_dataset
from delphi.clients import OpenRouter
from delphi.config import CacheConfig, ExperimentConfig, LatentConfig, RunConfig
from delphi.explainers import DefaultExplainer
from delphi.latents import Example, Latent, LatentCache, LatentDataset, LatentRecord
from delphi.latents.collect_activations import collect_activations
from delphi.latents.constructors import default_constructor
from delphi.latents.samplers import sample
from delphi.pipeline import Pipeline, process_wrapper
from delphi.sparse_coders import load_hooks_sparse_coders
from delphi.utils import load_tokenized_data
from openai import OpenAI
from sparsify.data import chunk_and_tokenize
from torchtyping import TensorType
from transformers import AutoModel, AutoTokenizer

# %%
# Load the model
model = AutoModel.from_pretrained("google/gemma-2-2b", device_map="cuda", torch_dtype="float16")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# Load the autoencoders, the function returns a dictionary of the submodules with the autoencoders and the edited model.
# it takes as arguments the model, the layers to load the autoencoders into,
# the average L0 sparsity per layer, the size of the autoencoders and the type of autoencoders (residuals or MLPs).

run_cfg = RunConfig(
    sparse_model="google/gemma-scope-2b-pt-res",
    hookpoints=["layer_5/width_16k/average_l0_68"]
    )

hookpoint_to_sparse_encode = load_hooks_sparse_coders(model, run_cfg)

# %%
# Loading the tokens and creating the cache

# There is a default cache config that can also be modified when using a "production" script.
# 98304 / 256 = 384. Therefore, we expect there to be 384 separate prompts, batched into (a max of) 8 examples each.
cfg = CacheConfig(
    dataset_repo="EleutherAI/rpj-v2-sample",
    dataset_split="train[:1%]",
    batch_size=8,
    ctx_len=256,
    n_tokens=100_000,
    n_splits=5
    )

# %%
data = load_dataset(cfg.dataset_repo, name=cfg.dataset_name, split=cfg.dataset_split)
tokens_ds = chunk_and_tokenize(
        data, tokenizer, max_seq_len=cfg.ctx_len, text_key="raw_content", num_proc=2
    )

seed = 22
tokens_ds = tokens_ds.shuffle(seed)

tokens = cast(TensorType["batch", "seq"], tokens_ds["input_ids"])
# %%
# Tokens should have the shape (n_batches,ctx_len)
cache = LatentCache(
    model,
    hookpoint_to_sparse_encode,
    batch_size = cfg.batch_size
)

# %%

# Running the cache and saving the results
cache.run(cfg.n_tokens, tokens)

cache.save_splits(
    n_splits=cfg.n_splits,  # We split the activation and location indices into different files to make loading faster
    save_dir="latents"
)

# The config of the cache should be saved with the results such that it can be loaded later.
cache.save_config(
    save_dir="latents",
    cfg=cfg,
    model_name="google/gemma-2-2b"
)

# %%

API_KEY = os.getenv("OPENAI_API_KEY")
latent_cfg = LatentConfig(
    # width=131072, # The number of latents of your SAE
    min_examples=200, # The minimum number of examples to consider for the latent to be explained
    max_examples=10000, # The maximum number of examples to be sampled from
    # n_splits=5 # How many splits was the cache split into
)
module = "layers.5" # The layer to explain
latent_idx = 11910
latent_dict = {module: torch.tensor([latent_idx])} # which latents to explain

experiment_cfg = ExperimentConfig(
    n_examples_train=40, # Number of examples to sample for training
    n_examples_test=100,
    n_quantiles=10,
    example_ctx_len=32, # Length of each example
    n_non_activating=100,
    train_type="quantiles", # Type of sampler to use for training.
    test_type="even"
)

constructor=partial(
            default_constructor,
            n_not_active=experiment_cfg.n_non_activating,
            ctx_len=experiment_cfg.example_ctx_len,
            max_examples=latent_cfg.max_examples
        )
sampler=partial(sample, cfg=experiment_cfg)

# %%

dataset = LatentDataset(
        raw_dir="latents", # The folder where the cache is stored
        cfg=latent_cfg,
        modules=[module],
        latents=latent_dict,
        constructor=constructor,
        sampler=sampler
)


# %%
from importlib import reload

import delphi

reload(delphi.explainers)
reload(delphi.explainers.default)
from delphi.explainers import DefaultExplainer

# client = OpenRouter("anthropic/claude-3.5-sonnet",api_key=API_KEY)
# client = OpenRouter("openai/gpt-4o-mini",api_key=API_KEY)
with open(os.path.expanduser('~/.config/gpt/keys')) as f:
    API_KEY = f.read().strip()
    os.environ["OPENAI_API_KEY"] = API_KEY
API_KEY = os.environ.get("OPENAI_API_KEY", None)
client = OpenAI(api_key=API_KEY)

# cache.cache.latent_locations[module].max(0) has values=tensor([ 383,  255, 16383]),
# corresponding to the 384 prompts, 256 ctx_len, and 16384 latents of GemmaScope 16k.
# cache.cache.tokens[module].shape is (384, 256)
# cache.cache.latent_activations[module].shape is (9905079,), because it holds only nonzero activations
# cache.cache.latent_locations[module].shape is (9905079, 3), where the last column holds the latent index
latent = Latent(module, latent_idx)

is_my_latent = cache.cache.latent_locations[module][:, 2] == latent_idx
my_latent_activations = cache.cache.latent_activations[module][is_my_latent]
active_on_which_tokens = cache.cache.latent_locations[module][is_my_latent][:, :2]
shuffling = torch.randperm(active_on_which_tokens.shape[0])

examples = []
for i_example in range(experiment_cfg.n_examples_train + experiment_cfg.n_examples_test):
    example_tokens = cache.cache.tokens[module][active_on_which_tokens[shuffling[i_example], 0]]
    # activations = torch.tensor(my_latent_activations[shuffling[i_example]])
    with torch.no_grad():
        with collect_activations(
            model, list(hookpoint_to_sparse_encode.keys())
        ) as activations:
            model(example_tokens.unsqueeze(0).to(model.device))
            for hookpoint, latents in activations.items():
                sae_latents = hookpoint_to_sparse_encode[hookpoint].encode(latents)
                # print(sae_latents.shape)

    example = Example(example_tokens, sae_latents[0, :, latent_idx])
    examples.append(example)
latent_record = LatentRecord(latent)
latent_record.train = examples[:experiment_cfg.n_examples_train]
latent_record.test = examples[experiment_cfg.n_examples_train:]

# client = OpenRouter(api_key=data["api_key"], model=data["model"])

explainer = DefaultExplainer(client, tokenizer=tokenizer, threshold=0.6)
# messages = explainer._build_prompt(latent_record.train)

result = explainer(latent_record)
# print(result.explanation)

# # %%
# explainer = DefaultExplainer(
#         client,
#         tokenizer=dataset.tokenizer,
#     )

# # The function that saves the explanations
# def explainer_postprocess(result):
#     with open(f"explanations/{result.record.latent}.txt", "wb") as f:
#         f.write(orjson.dumps(result.explanation))
#     del result
#     return None

# explainer_pipe = process_wrapper(
#     explainer,
#     postprocess=explainer_postprocess,
# )

# pipeline = Pipeline(
#     dataset,
#     explainer_pipe,
# )
# number_of_parallel_latents = 1
# await pipeline.run(number_of_parallel_latents) # This will start generating the explanations.
# # asyncio.run(pipeline.run(number_of_parallel_latents))
# # pipeline.run(number_of_parallel_latents)

# %%
