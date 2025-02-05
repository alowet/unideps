# %% Imports
import os
import pickle

from data_utils import collate_fn
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

# %%
# Train probes for different layers
# Note this is inefficient in that it re-runs the model (one forward pass per layer), though it stops at layer+1 each time
probes = {}
train_toks = "tail"
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
# comment this out if you want to overwrite existing probes
if not os.path.exists(f"data/{train_toks}.pkl"):
    with open(f"data/{train_toks}.pkl", "wb") as f:
        pickle.dump(probes, f)

# %%
# Visualize results
# ... visualization code ...
