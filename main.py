# %% Imports
import os
import pickle

import torch
from analyze_sae import main as analyze_sae_main
from configure_cuda import *
from data_utils import collate_fn
from evaluate import main as evaluate_main
from load_data import UDDataset
from model_utils import UDTransformer
from probing import train_probe
from torch.cuda import empty_cache
from torch.utils.data import DataLoader

# %% Set up devices

if torch.cuda.device_count() > 1:
    device_model = torch.device("cuda:0")
    device_sae = torch.device("cuda:1")
else:
    if torch.cuda.is_available():
        device_model = torch.device("cuda")
    elif torch.mps.is_available():
        device_model = torch.device("mps")
    else:
        device_model = torch.device("cpu")
    device_sae = device_model

print(f"Using devices - Model: {device_model}, SAE: {device_sae}")

# %% Load data
train_data = UDDataset("data/UD_English-EWT/en_ewt-ud-train.conllu", max_sentences=1024)
dev_data = UDDataset("data/UD_English-EWT/en_ewt-ud-dev.conllu", max_sentences=1024)

# %%
train_loader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    collate_fn=collate_fn
)
dev_loader = DataLoader(
    dev_data,
    batch_size=128,
    collate_fn=collate_fn
)

# %%
# Initialize model
model = UDTransformer(model_name="gemma-2-2b", device=device_model)

# %%
train_toks = "tail"
model_path = f"data/probes/{train_toks}.pkl"

# # %%
# # Train probes for different layers
# # Note this is inefficient in that it re-runs the model (one forward pass per layer), though it stops at layer+1 each time

# # comment out if you just want to load probes
# probes = {}
# # for layer in range(model.model.cfg.n_layers):
# for layer in range(7):
#     probe = train_probe(
#         model,
#         train_loader,
#         dev_loader,
#         device_model,
#         layer=layer,
#         train_toks=train_toks,
#     )
#     probes[layer] = probe.cpu()
#     del probe
#     empty_cache()

# # %%
# # comment this out if you want to overwrite existing probes
# if not os.path.exists(model_path):
#     with open(model_path, "wb") as f:
#         pickle.dump(probes, f)
# %%
# Load already trained probes
with open(model_path, "rb") as f:
    probes = pickle.load(f)
# %%
# for speed, select subset of layers
probes = {k: v for k, v in probes.items() if k < 7}

# %% Evaluate
test_data = UDDataset("data/UD_English-EWT/en_ewt-ud-test.conllu", max_sentences=1024)
test_loader = DataLoader(
    test_data,
    batch_size=128,
    collate_fn=collate_fn
)

evaluate_main(
    test_loader,
    probes,
    model,
    device_model,
    train_toks=train_toks
    )

# %%

analyze_sae_main(
    probes,
    train_toks=train_toks,
    device_model=device_model,
    device_sae=device_sae
)

# %%
