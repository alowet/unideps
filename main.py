# %% Imports
import importlib
import os
import pickle
import sys
from datetime import datetime

from rich import print as rprint
from rich.table import Table
from tqdm import tqdm

sys.path.append("../matryoshka_sae")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
from analyze_sae import display_dashboard
from analyze_sae import main as analyze_sae_main
from darkmatter import main as dm_main
from data_utils import collate_fn
from evaluate import main as evaluate_main
from load_data import UDDataset
from model_utils import UDTransformer
from nonsparse import analyze_sparsity, compute_activation_sparsity
from probing import train_probe
from sae import GlobalBatchTopKMatryoshkaSAE
from sae_lens import SAE, ActivationsStore
from scipy.stats import hypergeom, pearsonr, ranksums
from steering import generate_with_steering
from task import DependencyTask
from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from utils import load_sae_from_wandb

# %% Set up devices
# For now, we're using a single GPU, but in future, could add an option for UDTransformer and SAE to live on different devices
device_model = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
device_sae = device_model

# %%
train_toks = "tail"
deps = DependencyTask.dependency_table()

# for train_toks in ["last","head"]:
model_path = f"data/probes/{train_toks}_tail_trail.pkl"
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print("Run group:", start_time)

# %% Load data
train_data = UDDataset("data/UD_English-EWT/en_ewt-ud-train.conllu", max_sentences=1024)
dev_data = UDDataset("data/UD_English-EWT/en_ewt-ud-dev.conllu", max_sentences=1024)

# %%
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda x: collate_fn(x)
)
dev_loader = DataLoader(
    dev_data,
    batch_size=64,
    collate_fn=lambda x: collate_fn(x)
)

# %%
# Initialize model
ud_model = UDTransformer(model_name="gemma-2-2b", device=device_model)


# %%
# Load already trained probes
# with open(model_path, "rb") as f:
#     probes = pickle.load(f)


#  %%
# Train probes for different layers
# Note this is inefficient in that it re-runs the model (one forward pass per layer), though it stops at layer+1 each time
# comment this block out if you just want to load probes

probes = {}
for layer in range(ud_model.model.cfg.n_layers):
# for layer in range(1):
    probe = train_probe(
        ud_model,
        train_loader,
        dev_loader,
        device_model,
        layer=layer,
        train_toks=train_toks,
        run_group=start_time
    )
    probes[layer] = probe.cpu()
    del probe
    empty_cache()

# %%
# comment the next line out if you want to overwrite existing probes
if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        pickle.dump(probes, f)

# %%
# for speed, select subset of layers
# probes = {k: v for k, v in probes.items() if k < 7}

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
    ud_model,
    train_toks=train_toks,
    device=device_model
    )

# %%
layer = 5  # pre layer 5 for probes and SAE
width = 16
save_path = f'figures/sparsity_layer_{layer}_width_{width}.png'
which_sae = "pretrained"  # "pretrained" or "matryoshka"
pre_or_post = "pre"  # "pre" or "post"
scope_layer = layer if pre_or_post == "pre" else layer - 1

# Load pretrained SAE
sae_release = "gemma-scope-2b-pt-res-canonical"
# use layer - 1 for pretrained SAE, because pretrained SAEs use resid_post, but probes use resid_pre
sae_id = f"layer_{scope_layer}/width_{width}k/canonical"
sae = SAE.from_pretrained(sae_release, sae_id, device=str(device_sae))[0]
d_sae = sae.cfg.d_sae

# use ActivationsStore to stream in data from a given dataset
act_store = ActivationsStore.from_sae(
    model=ud_model.model,
    sae=sae,
    streaming=True,
    store_batch_size_prompts=16,
    n_batches_in_buffer=16,
    device=str(device_sae),
)

if which_sae == "matryoshka":  # overwrite pretrained SAE with matryoshka SAE
    del sae
    # Load Matryoshka SAE
    # note that this and the probes use resid_pre, whereas the pretrained SAEs use resid_post!!
    entity = "adam-lowet-harvard-university"
    project = "batch-topk-matryoshka"
    sae_id = "gemma-2-2b_blocks.5.hook_resid_pre_36864_global-matryoshka-topk_32_0.0003_122069"

    sae, cfg = load_sae_from_wandb(f"{entity}/{project}/{sae_id}:latest", GlobalBatchTopKMatryoshkaSAE)
    d_sae = cfg['dict_size']

# %%
# # steer layer 5, latent 11910, which corresponded strongly to `amod` (adjectival modifier)

# GENERATE_KWARGS = dict(temperature=0.5, freq_penalty=2.0, verbose=False)
# prompt = "When I look at myself in the mirror, I see"
# latent_idx = 11910
# max_new_tokens = 500

# no_steering_output = ud_model.model.generate(prompt, max_new_tokens=max_new_tokens, **GENERATE_KWARGS)

# table = Table(show_header=False, show_lines=True, title="Steering Output")
# table.add_row("Normal", no_steering_output)
# for i in tqdm(range(3), "Generating steered examples..."):
#     table.add_row(
#         f"Steered #{i}",
#         generate_with_steering(
#             ud_model.model,
#             sae,
#             prompt,
#             latent_idx,
#             steering_coefficient=30.0,  # roughly 1.5x the latent's max activation
#             max_new_tokens=max_new_tokens,
#             **GENERATE_KWARGS
#         )
#     )
# rprint(table)


# %%
if which_sae == "matryoshka":
    frac_active = compute_matryoshka_activation_sparsity(model.model, sae, act_store, cfg)
else:
    frac_active = compute_activation_sparsity(model.model, sae, act_store)
# analyze_sparsity(
#     model,
#     sae,
#     act_store,
#     layer,
#     device_sae,
#     save_path
# )
# %%
px.histogram(
        frac_active,
        nbins=50,
        title="ACTIVATIONS DENSITY",
        width=800,
        template="ggplot2",
        color_discrete_sequence=["darkorange"],
        log_y=True,
    ).update_layout(bargap=0.02, showlegend=False).show()
# %%
# (frac_active > .1).sum()
batch_size = 8192

with torch.no_grad():
    # Move probe to SAE GPU and detach from computation graph
    probe = probes[layer].to(device_sae)
    weights = probe.probe.weight.detach()  # [num_relations, hidden_dim]
    sae_features = sae.W_enc.detach()  # [hidden_dim, num_features]

    # Normalize vectors
    weights_norm = torch.nn.functional.normalize(weights, p=2, dim=1)
    features_norm = torch.nn.functional.normalize(sae_features, p=2, dim=0)

    # Compute similarities in batches
    similarities_list = []
    for start_idx in range(0, d_sae, batch_size):
        end_idx = min(start_idx + batch_size, d_sae)
        # Compute similarities for this batch of features
        batch_similarities = torch.matmul(
            weights_norm,
            features_norm[:, start_idx:end_idx]
        )
        similarities_list.append(batch_similarities)

    # Concatenate all batches
    similarities = torch.cat(similarities_list, dim=1)

    # Get max similarity for each relation
    max_sims, _ = similarities.max(dim=1)  # [num_relations]

    similarities = similarities.cpu().numpy()
    # max_sims = max_sims.cpu().numpy()

# %%
fig, axs = plt.subplots(8, 7, figsize=(10, 10))
for i, (ax, rel) in enumerate(zip(axs.flatten(), deps.keys())):
    ax.hist(similarities[i], bins=50)
    ax.set_title(f"{rel}")
    ax.semilogy()
plt.tight_layout()
plt.show()

# %%
# We now have similarities of shape [num_relations, d_sae] and frac_active of shape [d_sae]
# So we can ask if sparse latents are also the ones with high cosine similarity to the probe weights

# scatter plot of sims vs. frac_active for each relation separately
log_frac_active = np.log(frac_active)
is_valid = np.logical_and(~np.isnan(log_frac_active), ~np.isneginf(log_frac_active))

fig, axs = plt.subplots(8, 7, figsize=(14, 12))
for i, (ax, rel) in enumerate(zip(axs.flatten(), deps.keys())):
    ax.scatter(similarities[i][is_valid], log_frac_active[is_valid], s=1, alpha=0.1)
    r, p = pearsonr(similarities[i][is_valid], log_frac_active[is_valid])
    ax.set_title(f"r={r:.2f}, p={p:.2e}")
    ax.set_ylabel(rel)
    # ax.set_xlabel("Cosine Similarity")
    # ax.set_ylabel("Activation Density")
plt.tight_layout()
plt.show()

# %%
# See what the overlap is between the top_n latents (cosine similarity with each dependency relation) and the the top_n active latents (frac_active)
top_n = 100

top_sims = np.argsort(similarities, axis=1)[:, -top_n:]
top_active = np.argsort(frac_active)[-top_n:]

all_inters = []
for i, rel in enumerate(deps.keys()):
    inter = np.intersect1d(top_sims[i], top_active)
    n = len(inter)
    p = hypergeom(M=d_sae,
                  n=top_n,
                  N=top_n).sf(n-1)
    if p < 0.05:
        all_inters.extend(inter)
    print(f"{rel}: {inter}, p={p:.3f}")

# %%
# for latent_idx in np.unique(all_inters):
# for latent_idx in top_active[:22]:
for latent_idx in [6409]:
    display_dashboard(
        sae_release="gemma-scope-2b-pt-res-canonical",
        sae_id=f"layer_{layer - 1}/width_{width}k/canonical",
        latent_idx=latent_idx,
        width=400,
        height=300
    )

# %%
# Let's try a different tack: Select only those latents that have activation density > some threshold and see if the cosine sims are higher than for others
frac_active_threshold = 0.1
which_active = frac_active > frac_active_threshold
cos_sim_threshold = 0.1

sim_df = pd.DataFrame(similarities.T, columns=deps.keys())
sim_df["frac_active"] = frac_active
sim_df["is_active"] = which_active
# print(sim_df.head())
sim_df = sim_df.melt(id_vars=["frac_active", "is_active"], var_name="relation", value_name="cosine_sim")
sim_df["is_high_sim"] = sim_df["cosine_sim"] > cos_sim_threshold
# print(sim_df.head())

all_pvals = []
all_ranksums = []
for dep in deps.keys():
    x = sim_df.loc[sim_df["is_high_sim"] & (sim_df["relation"] == dep), "cosine_sim"]
    y = sim_df.loc[~sim_df["is_high_sim"] & (sim_df["relation"] == dep), "cosine_sim"]
    stats = ranksums(x, y)
    all_pvals.append(stats.pvalue)
    all_ranksums.append(stats.statistic)
    # print(f"{dep}: {pval:.3f}")

# %%
incr = 8
for start in range(0, len(deps), incr):
    # sns.violinplot(data=sim_df, x="relation", y="cosine_sim", hue="is_active", order=list(deps.keys())[start:start+incr])
    for y_var, hue_var, log_scale in [("frac_active", "is_high_sim", False), ("cosine_sim", "is_active", False)]:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.violinplot(data=sim_df[sim_df["frac_active"] < 0.7], x="relation", y=y_var, hue=hue_var, order=list(deps.keys())[start:start+incr], cut=0, ax=ax, log_scale=log_scale)
        if y_var == "frac_active":
            end = min(start+incr, len(deps))
            for i, idx in enumerate(range(start, end)):
                ax.text(i, .5, f"p={all_pvals[idx]:.3f}", ha="center", va="center")
                ax.text(i, .45, f"stat={all_ranksums[idx]:.3f}", ha="center", va="center")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"figures/sparsity/frac_active_yvar_{y_var}_hue_{hue_var}_{sae_id.replace('/', '_')}_{list(deps.keys())[start]}_to_{list(deps.keys())[end-1]}.png")
    plt.show()


# %%
# Following Engels et al., 2024, see how much SAE error is linearly predictable (a) based on the residual stream itself and (b) based on the universal dependencies at that position
import darkmatter

importlib.reload(darkmatter)
from darkmatter import main as dm_main

dm_main(
    model=ud_model,
    train_loader=train_loader,
    dev_loader=dev_loader,
    device=device_sae,
    layer=5,
    sae_width=16
)

# %%
# import analyze_sae
# importlib.reload(analyze_sae)
# from analyze_sae import main as analyze_sae_main

# analyze_sae_main(
#     probes,
#     train_toks=train_toks,
#     device_sae=device_sae
# )

# # %%
# # A feature that really jumps out here is Layer 1: Feature 7634, which has a cosine similarity of 0.387 with the conj relation
# # Inspect this feature
# from analyze_sae import display_dashboard

# # display_dashboard(
# #     sae_release="gemma-scope-2b-pt-res-canonical",
# #     sae_id="layer_1/width_16k/canonical",
# #     latent_idx=7634,
# #     width=800,
# #     height=600
# # )

# # Here's one with high cosine sim to the obj relation (layer 3 latent idx 1634: cosine sim 0.243)
# display_dashboard(
#     sae_release="gemma-scope-2b-pt-res-canonical",
#     sae_id="layer_3/width_16k/canonical",
#     latent_idx=1634,
#     width=800,
#     height=600
# )

# # And to the nmod relation (layer 3 latent idx 10933: 0.189)
# display_dashboard(
#     sae_release="gemma-scope-2b-pt-res-canonical",
#     sae_id="layer_3/width_16k/canonical",
#     latent_idx=10933,
#     width=800,
#     height=600
# )



# # %%
