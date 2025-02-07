# %% Imports
import importlib
import os
import pickle
from datetime import datetime

import torch
from analyze_sae import main as analyze_sae_main
from data_utils import collate_fn
from evaluate import main as evaluate_main
from load_data import UDDataset
from model_utils import UDTransformer
from probing import train_probe
from torch.cuda import empty_cache
from torch.utils.data import DataLoader

# %% Set up devices
# For now, we're using a single GPU, but in future, could add an option for UDTransformer and SAE to live on different devices
device_model = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
device_sae = device_model

# %%
train_toks = "tail"
max_layer =
model_path = f"data/probes/{train_toks}.pkl"


# %% Load data
train_data = UDDataset("data/UD_English-EWT/en_ewt-ud-train.conllu", max_sentences=1024)
dev_data = UDDataset("data/UD_English-EWT/en_ewt-ud-dev.conllu", max_sentences=1024)

# %%
train_loader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    collate_fn=lambda x: collate_fn(x, train_toks=train_toks)
)
dev_loader = DataLoader(
    dev_data,
    batch_size=128,
    collate_fn=lambda x: collate_fn(x, train_toks=train_toks)
)

# %%
# Initialize model
model = UDTransformer(model_name="gemma-2-2b", device=device_model)


# %%
# # Load already trained probes
# with open(model_path, "rb") as f:
#     probes = pickle.load(f)


#  %%
# Train probes for different layers
# Note this is inefficient in that it re-runs the model (one forward pass per layer), though it stops at layer+1 each time
# comment this block out if you just want to load probes

probes = {}
for layer in range(model.model.cfg.n_layers):
# for layer in range(1):
    probe = train_probe(
        model,
        train_loader,
        dev_loader,
        device_model,
        layer=layer,
        train_toks=train_toks,
        run_group=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    probes[layer] = probe.cpu()
    del probe
    empty_cache()

# %%
# comment the next line out if you want to overwrite existing probes
# if not os.path.exists(model_path):
with open(model_path, "wb") as f:
    pickle.dump(probes, f)

# %%
# for speed, select subset of layers
# probes = {k: v for k, v in probes.items() if k < 7}

# %%
# import analyze_sae
# importlib.reload(analyze_sae)
# from analyze_sae import main as analyze_sae_main

analyze_sae_main(
    probes,
    train_toks=train_toks,
    device_sae=device_sae
)

# %%
# A feature that really jumps out here is Layer 1: Feature 7634, which has a cosine similarity of 0.387 with the conj relation
# Inspect this feature
# from analyze_sae import display_dashboard

# display_dashboard(
#     sae_release="gemma-scope-2b-pt-res-canonical",
#     sae_id="layer_1/width_16k/canonical",
#     latent_idx=7634,
#     width=800,
#     height=600
# )

# %% Evaluate
# import evaluate
# importlib.reload(evaluate)
# from evaluate import main as evaluate_main

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
    train_toks=train_toks,
    device=device_model
    )


# %%
