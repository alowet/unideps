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
import analyze_sae
import matplotlib.pyplot as plt
import nonsparse
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch

importlib.reload(analyze_sae)
from analyze_sae import *
from analyze_sae import main as analyze_sae_main
from darkmatter import main as dm_main
from data_utils import collate_fn
from evaluate import main as evaluate_main
from load_data import UDDataset
from model_utils import UDTransformer

importlib.reload(nonsparse)
from nonsparse import *
from probing import train_probe
from sae import GlobalBatchTopKMatryoshkaSAE
from sae_lens import SAE, ActivationsStore
from scipy.stats import hypergeom, pearsonr, ranksums
from steering import generate_with_steering
from task import DependencyTask
from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from utils import load_sae_from_wandb

from wandb import CommError

# %% Set up devices
# For now, we're using a single GPU, but in future, could add an option for UDTransformer and SAE to live on different devices
device_model = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")
device_sae = device_model


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
train_toks = "tail"
which_pos = "trail"
min_occurrences = 50
deps = DependencyTask.dependency_table()
dep_counts = DependencyTask.count_dependencies(train_data)

# for train_toks in ["last","head"]:
model_name = f"{train_toks}_{which_pos}_min_{min_occurrences}"
model_path = f"data/probes/{model_name}.pkl"
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print("Run group:", start_time)


#  %%
# Count dependencies and filter for frequent ones

frequent_deps = {
    rel: count for rel, count in dep_counts.items()
    if count >= min_occurrences
}

print(f"Found {len(frequent_deps)} dependency relations with >= {min_occurrences} occurrences:")
for rel, count in sorted(dep_counts.items(), key=lambda x: x[1], reverse=True):
    if count >= min_occurrences:
        print(f"{rel}: {count}")


# %%
# Load already trained probes
with open(model_path, "rb") as f:
    probes = pickle.load(f)


# %%
# # Train probes for different layers
# # Note this is inefficient in that it re-runs the model (one forward pass per layer), though it stops at layer+1 each time
# # comment this block out if you just want to load probes
# import probing

# importlib.reload(probing)
# from probing import train_probe

# probes = {}
# for layer in range(ud_model.model.cfg.n_layers):
#     # Pass frequent_deps to compute_loss during training
#     probe = train_probe(
#         ud_model,
#         train_loader,
#         dev_loader,
#         device_model,
#         layer=layer,
#         train_toks=train_toks,
#         run_group=start_time,
#         frequent_deps=list(frequent_deps.keys())
#     )
#     probes[layer] = probe.cpu()
#     del probe
#     empty_cache()

# # %%
# # comment the next line out if you want to overwrite existing probes
# if not os.path.exists(model_path):
#     with open(model_path, "wb") as f:
#         pickle.dump(probes, f)


# %% Evaluate
import evaluate

importlib.reload(evaluate)
from evaluate import main as evaluate_main

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
    device=device_model,
    frequent_deps=list(frequent_deps.keys())
)

# %%
# use ActivationsStore to stream in data the dataset used to train the Gemma Scope SAEs
width = 16
sae_release = "gemma-scope-2b-pt-res-canonical"
sae_id = f"layer_1/width_{width}k/canonical"
sae = SAE.from_pretrained(sae_release, sae_id, device=str(device_sae))[0]
act_store = ActivationsStore.from_sae(
    model=ud_model.model,
    sae=sae,
    streaming=True,
    store_batch_size_prompts=16,
    n_batches_in_buffer=16,
    device=str(device_sae),
)
d_sae = sae.cfg.d_sae
del sae

# %%
# compute alignment between probes and SAE at each layer
# compute activation density at each layer
# plot activation density broken out by high vs. low alignment for each dependency relation

# critical choice: gemma_scope vs. matryoshka SAE
which_sae = "gemma_scope"  # "gemma_scope" or "matryoshka"
pre_or_post = "resid_post" if which_sae == "gemma_scope" else "resid_pre"  # Gemma Scope hooked resid_post; the linear probes and Matryoshka SAEs hooked resid_pre
if which_sae == "matryoshka":
    d_sae = 2304*16

# wandb settings from Matryoshka SAE training
entity = "adam-lowet-harvard-university"
project = "batch-topk-matryoshka"
all_similarities = np.zeros((ud_model.model.cfg.n_layers - 1, len(frequent_deps), d_sae))
all_frac_active = np.zeros((ud_model.model.cfg.n_layers - 1, d_sae))
cfgs = []

# %%

for i_layer, layer in enumerate(range(1, ud_model.model.cfg.n_layers)):

    save_path = f'figures/sparsity_layer_{layer}_width_{width}.png'
    sae_layer = layer if pre_or_post == "resid_pre" else layer - 1

    if which_sae == "matryoshka":  # overwrite pretrained SAE with matryoshka SAE
        # Load Matryoshka SAE
        sae_id = f"gemma-2-2b_blocks.{sae_layer}.hook_resid_pre_36864_global-matryoshka-topk_32_0.0003_122069"
        try:
            sae, cfg = load_sae_from_wandb(f"{entity}/{project}/{sae_id}:latest", GlobalBatchTopKMatryoshkaSAE)
        except CommError:
            tmp_sae_id = f"gemma-2-2b_blocks.{sae_layer}.hook_resid_pre_36864_batch-topk_32_0.0003_122069"
            sae, cfg = load_sae_from_wandb(f"{entity}/{project}/{tmp_sae_id}:latest", GlobalBatchTopKMatryoshkaSAE)

        assert d_sae == cfg['dict_size']
        frac_active = compute_matryoshka_activation_sparsity(ud_model.model, sae, act_store, cfg)
        cfgs.append(cfg)
    else:
        # Load pretrained SAE
        # use layer - 1 for pretrained SAE, because pretrained SAEs use resid_post, but probes use resid_pre
        sae_id = f"layer_{sae_layer}/width_{width}k/canonical"
        sae = SAE.from_pretrained(sae_release, sae_id, device=str(device_sae))[0]
        assert d_sae == sae.cfg.d_sae

        frac_active = compute_activation_sparsity(ud_model.model, sae, act_store)
        cfgs.append(sae.cfg)

    plot_activation_density(frac_active, which_sae, layer)
    all_frac_active[i_layer] = frac_active
    all_similarities[i_layer] = compute_probe_sae_alignment(probes[layer], sae, device_sae, d_sae)

    plot_similarities_hist(all_similarities[i_layer], list(frequent_deps.keys()), which_sae, layer, model_name)

np.save(f"data/similarities/{which_sae}/all_similarities_width_{width}_{model_name}.npy", all_similarities)
np.save(f"data/similarities/{which_sae}/all_frac_active_width_{width}.npy", all_frac_active)
np.save(f"data/similarities/{which_sae}/all_cfgs_width_{width}.npy", cfgs)

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

# analyze_sparsity(
#     model,
#     sae,
#     act_store,
#     layer,
#     device_sae,
#     save_path
# )
# %%



# %%
# Following Engels et al., 2024, see how much SAE error is linearly predictable (a) based on the residual stream itself and (b) based on the universal dependencies at that position
# dm_main(
#     model=ud_model,
#     train_loader=train_loader,
#     dev_loader=dev_loader,
#     device=device_sae,
#     layer=5,
#     sae_width=16
# )

# %%
import analyze_sae

importlib.reload(analyze_sae)
from analyze_sae import main as analyze_sae_main

# TODO: FIX THIS to use frequent_deps
analyze_sae_main(
    probes,
    train_toks=train_toks,
    device_sae=device_sae,
    frequent_deps=list(frequent_deps.keys())
)

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


# %%
# # for latent_idx in np.unique(all_inters):
# # for latent_idx in top_active[:22]:
# for latent_idx in [6409]:
#     display_dashboard(
#         sae_release="gemma-scope-2b-pt-res-canonical",
#         sae_id=f"layer_{layer - 1}/width_{width}k/canonical",
#         latent_idx=latent_idx,
#         width=400,
#         height=300
#     )
# # %%

# %%
