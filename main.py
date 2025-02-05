# %% Imports
import os
import pickle

from data_utils import collate_fn
from evaluate import main as evaluate_main
from load_data import UDDataset
from model_utils import UDTransformer
from probing import train_probe
from torch.cuda import empty_cache
from torch.utils.data import DataLoader

# %%
# Load data
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
model = UDTransformer()
train_toks = "tail"

# %%
# Train probes for different layers
# Note this is inefficient in that it re-runs the model (one forward pass per layer), though it stops at layer+1 each time

# comment out if you just want to load probes
probes = {}
# for layer in range(model.model.cfg.n_layers):
for layer in [11]:
    probe = train_probe(
        model,
        train_loader,
        dev_loader,
        layer=layer,
        train_toks=train_toks
    )
    probes[layer] = probe.cpu()
    del probe
    empty_cache()

# %%
model_path = f"data/probes/{train_toks}.pkl"
# comment this out if you want to overwrite existing probes
if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        pickle.dump(probes, f)
# %%
# Load probes
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

evaluate_main(test_loader, probes, model, train_toks=train_toks)

# %%
