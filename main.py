# %% Imports
import importlib
import os
import pickle
import sys
from datetime import datetime
import einops
from rich import print as rprint
from rich.table import Table
from tqdm import tqdm

sys.path.append("../matryoshka_sae")
import json

import analyze_sae
import cmocean
import matplotlib.pyplot as plt
import nonsparse
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch

importlib.reload(analyze_sae)
from analyze_sae import *

# from analyze_sae import main as analyze_sae_main
from darkmatter import main as dm_main
from data_utils import collate_fn

# from evaluate import main as evaluate_main
from evaluate_merged import main as evaluate_merged_main
from evaluate_merged import plot_layer_results
from load_data import UDDataset
from model_utils import UDTransformer
from scipy import stats

importlib.reload(nonsparse)
import sae
from config import post_init_cfg
from logs import load_checkpoint
from nonsparse import *

# from probing import train_probe
from probing_merged import train_probe
from stats import compute_contingency_test, get_significance_stars
from plotting import plot_correlations, plot_stars, hide_spines

importlib.reload(sae)
from activation_store import ActivationsStore
from evaluate_top_latents import (
    evaluate_latent_predictions,
    get_top_latents_per_dep,
    plot_precision_recall,
    main as evaluate_top_latents_main
)
from evaluate_top_latents import main as evaluate_top_latents_main
from sae import BatchTopKSAE, GlobalBatchTopKMatryoshkaSAE
from sae_lens import SAE, ActivationsStore
from scipy.stats import fisher_exact, hypergeom, pearsonr, ranksums
from steering import generate_with_steering
from task import DependencyTask
from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from utils import load_sae_from_wandb
from wandb import CommError

from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import run_eval

# %% Set up devices
# For now, we're using a single GPU, but in future, could add an option for UDTransformer and SAE to live on different devices
device_model = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
device_sae = device_model


# %% Load data
train_data = UDDataset("data/UD_English-EWT/en_ewt-ud-train.conllu", max_sentences=1024)
dev_data = UDDataset("data/UD_English-EWT/en_ewt-ud-dev.conllu", max_sentences=1024)

# %%
# Initialize model
ud_model = UDTransformer(model_name="gemma-2-2b", device=device_model)

# %%
import data_utils, task
importlib.reload(data_utils)
importlib.reload(task)
from data_utils import collate_fn
from task import DependencyTask

sentence_batch_size = 32 if torch.cuda.is_available() and str(torch.cuda.mem_get_info()[1]).startswith("20") else 1024
# sentence_batch_size = 256

train_loader = DataLoader(
    train_data,
    batch_size=sentence_batch_size,
    shuffle=True,
    collate_fn=lambda x: collate_fn(x)
)
dev_loader = DataLoader(
    dev_data,
    batch_size=sentence_batch_size,
    collate_fn=lambda x: collate_fn(x)
)

# %%

probe_type = "multiclass"
max_layer = 26


train_toks = "tail"
min_occurrences = 20
deps = DependencyTask.dependency_table()
dep_counts = DependencyTask.count_dependencies(train_data)


# for train_toks in ["last","head"]:
model_name = f"{probe_type}_{train_toks}_min_{min_occurrences}"
model_path = f"data/probes/{model_name}_layer_{max_layer - 1}.pkl"
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print("Run group:", start_time)


#  %%
# Count dependencies and filter for frequent ones

frequent_deps = {
    rel: count for rel, count in dep_counts.items()
    if count >= min_occurrences
}

ndeps = len(frequent_deps)
print(f"Found {ndeps} dependency relations with >= {min_occurrences} occurrences:")
for rel, count in sorted(dep_counts.items(), key=lambda x: x[1], reverse=True):
    if count >= min_occurrences:
        print(f"{rel}: {count}")


# %%
import probing_merged
importlib.reload(probing_merged)
from probing_merged import train_probe

# Load already trained probes
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        probes = pickle.load(f)
else:
    print(f"Probes not found at {model_path}, training new probes")

    # Train probes for different layers
    # Note this is inefficient in that it re-runs the model (one forward pass per layer), though it stops at layer+1 each time
    probes = {}
    for layer in range(0, ud_model.model.cfg.n_layers):

        probe = train_probe(
            model=ud_model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            device=device_model,
            layer=layer,
            num_epochs=30,
            train_toks=train_toks,
            run_group=start_time,
            frequent_deps=list(frequent_deps.keys()),
            probe_type=probe_type
        )

        probes[layer] = probe
        del probe
        empty_cache()

    # comment the next line out if you want to overwrite existing probes
    # if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        print(f"Saving probes to {model_path}")
        pickle.dump(probes, f)


# %%
test_data = UDDataset("data/UD_English-EWT/en_ewt-ud-test.conllu", max_sentences=1024)
test_loader = DataLoader(
    test_data,
    batch_size=sentence_batch_size,
    collate_fn=collate_fn
)

# %% Evaluate
import evaluate_merged
importlib.reload(evaluate_merged)
from evaluate_merged import main as evaluate_merged_main
from evaluate_merged import plot_layer_results
probe_results = evaluate_merged_main(
    test_loader,
    probes,
    ud_model,
    train_toks=train_toks,
    device=device_model,
    frequent_deps=list(frequent_deps.keys()),
    probe_type=probe_type,
    model_name=model_name
)

# get results
# results_path = f"data/evals/{probe_type}_eval_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps)}.pkl"
# probe_results = pickle.load(open(results_path, "rb"))

# save_path = f"figures/evals/{probe_type}_eval_{train_toks}_results_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps)}.png"
# probe_stats = plot_layer_results(
#     probe_results,
#     list(frequent_deps.keys()),
#     save_path=save_path
# )

# %%
# plot the correlation matrix for probes themselves
probe_weights = np.array([probe.probe.weight.detach().cpu().numpy() for probe in probes.values()])
# print(probe_weights.shape)

plot_correlations(probe_weights, save_path=f"figures/evals/{probe_type}_eval_{train_toks}_probe_correlations_layers_{min(probes.keys())}-{max(probes.keys())}_ndeps_{len(frequent_deps)}.png", n_layers=ud_model.model.cfg.n_layers, ndeps=ndeps, frequent_deps=frequent_deps)

# %%
# compute activation density at each layer
# compute alignment between probes and SAE at each layer

# critical choice: gemma_scope vs. matryoshka SAE
which_sae = "gemma_scope"  # "gemma_scope" or "matryoshka" or "batch-topk"
pre_or_post = "resid_post" if which_sae == "gemma_scope" else "resid_pre"  # Gemma Scope hooked resid_post; the linear probes and other SAEs hooked resid_pre
# n_sim_layers = ud_model.model.cfg.n_layers - 1
# n_sim_layers = ud_model.model.cfg.n_layers if which_sae == "gemma_scope" else ud_model.model.cfg.n_layers - 1
n_sim_layers = 26 if which_sae == "gemma_scope" else 25
# wandb settings from Matryoshka SAE training
entity = "adam-lowet-harvard-university"
project = "batch-topk-matryoshka"
width = 16

# %%
# # this only needs to be run once
# # use ActivationsStore to stream in data the dataset used to train the Gemma Scope SAEs

# import analyze_sae

# importlib.reload(analyze_sae)
# from analyze_sae import (
#     compute_probe_sae_alignment,
#     plot_similarities_hist,
#     plot_similarity_significance_all_layers,
# )

# sae_release = "gemma-scope-2b-pt-res-canonical"
# sae_id = f"layer_1/width_{width}k/canonical"
# sae = SAE.from_pretrained(sae_release, sae_id, device=str(device_sae))[0]
# act_store = ActivationsStore.from_sae(
#     model=ud_model.model,
#     sae=sae,
#     streaming=True,
#     store_batch_size_prompts=16,
#     n_batches_in_buffer=16,
#     device=str(device_sae),
# )
# if which_sae == "batch-topk" or which_sae == "matryoshka":
#     d_sae = 2304*16
# else:
#     d_sae = sae.cfg.d_sae
# del sae

# n_permutations = 100
# enc_similarities = np.zeros((n_sim_layers, ndeps, d_sae))
# dec_similarities = np.zeros((n_sim_layers, ndeps, d_sae))
# enc_null_maxes = np.zeros((n_sim_layers, ndeps, n_permutations))
# dec_null_maxes = np.zeros((n_sim_layers, ndeps, n_permutations))
# all_frac_active = np.zeros((n_sim_layers, d_sae))
# cfgs = []

# # for i_layer, layer in enumerate(range(1, n_sim_layers + 1)):
# for i_layer, layer in tqdm(enumerate(range(n_sim_layers)), desc="Layers", total=n_sim_layers):

#     save_path = f'figures/sparsity_layer_{layer}_width_{width}.png'
#     sae_layer = layer + 1 if pre_or_post == "resid_pre" else layer

#     if which_sae == "gemma_scope":
#         # Load pretrained SAE
#         # use layer - 1 for pretrained SAE, because pretrained SAEs use resid_post, but probes use resid_pre
#         sae_id = f"layer_{sae_layer}/width_{width}k/canonical"
#         sae = SAE.from_pretrained(sae_release, sae_id, device=str(device_sae))[0]
#         assert d_sae == sae.cfg.d_sae

#         frac_active = compute_activation_sparsity(ud_model.model, sae, act_store)
#         cfgs.append(sae.cfg)

#     else:
#         if which_sae == "matryoshka":  # overwrite pretrained SAE with matryoshka SAE
#             # Load Matryoshka SAE
#             sae_id = f"gemma-2-2b_blocks.{sae_layer}.hook_resid_pre_36864_global-matryoshka-topk_32_0.0003_122069"
#             sae, cfg = load_sae_from_wandb(f"{entity}/{project}/{sae_id}:latest", GlobalBatchTopKMatryoshkaSAE)

#         elif which_sae == "batch-topk":
#             sae_id = f"gemma-2-2b_blocks.{sae_layer}.hook_resid_pre_36864_batch-topk_32_0.0003"
#             # sae_id = f"gemma-2-2b_blocks.{sae_layer}.hook_resid_pre_18432_batch-topk_29_0.0003"
#             tmp_cfg = {"name": sae_id}
#             state_dict, start_step = load_checkpoint(tmp_cfg)
#             print(f"Loaded state_dict from checkpoint {sae_id} at step {start_step - 1}")
#             # Load the configuration
#             checkpoint_dir = f"../matryoshka_sae/checkpoints/{sae_id}_{start_step - 1}"
#             config_path = os.path.join(checkpoint_dir, "config.json")
#             with open(config_path, "r") as f:
#                 cfg = json.load(f)
#             cfg["dtype"] = eval(cfg["dtype"])
#             sae = BatchTopKSAE(cfg)
#             sae.load_state_dict(state_dict)
#             # sae, cfg = load_sae_from_wandb(f"{entity}/{project}/{sae_id}:latest", BatchTopKSAE)
#         else:
#             raise ValueError(f"Invalid SAE type: {which_sae}")

#         assert d_sae == cfg['dict_size']
#         assert which_sae in cfg["sae_type"]
#         frac_active = compute_matryoshka_activation_sparsity(ud_model.model, sae, act_store, cfg)
#         cfgs.append(cfg)

#     plot_activation_density(frac_active, which_sae, layer, width)
#     all_frac_active[i_layer] = frac_active
#     enc_similarities[i_layer], dec_similarities[i_layer], enc_null_maxes[i_layer], dec_null_maxes[i_layer] = compute_probe_sae_alignment(probes[layer], sae, device_sae, d_sae, n_permutations=n_permutations, permute_features=True)
#     plot_similarities_hist(enc_similarities[i_layer], list(frequent_deps.keys()), which_sae, layer, width, model_name)

# # # %%
# # # save the results
# os.makedirs(f"data/similarities/{which_sae}", exist_ok=True)
# np.save(f"data/similarities/{which_sae}/enc_similarities_width_{width}_{model_name}.npy", enc_similarities)
# np.save(f"data/similarities/{which_sae}/dec_similarities_width_{width}_{model_name}.npy", dec_similarities)
# np.save(f"data/similarities/{which_sae}/enc_null_maxes_width_{width}_{model_name}.npy", enc_null_maxes)
# np.save(f"data/similarities/{which_sae}/dec_null_maxes_width_{width}_{model_name}.npy", dec_null_maxes)
# np.save(f"data/similarities/{which_sae}/all_frac_active_width_{width}_{model_name.replace(f'{probe_type}_', '')}.npy", all_frac_active)
# np.save(f"data/similarities/{which_sae}/all_cfgs_width_{width}_{model_name.replace(f'{probe_type}_', '')}.npy", cfgs)


# %%
# We now have all_similarities of shape [n_layers, n_relations, d_sae] and frac_active of shape [n_layers, d_sae]
sae_comp = {}
for sae_name, sae_width in [("gemma_scope", 16), ("matryoshka", 16)]:
    enc_similarities = np.load(f"data/similarities/{sae_name}/enc_similarities_width_{sae_width}_{model_name}.npy")
    dec_similarities = np.load(f"data/similarities/{sae_name}/dec_similarities_width_{sae_width}_{model_name}.npy")
    max_enc_similarities = np.max(enc_similarities, axis=-1)
    max_dec_similarities = np.max(dec_similarities, axis=-1)
    enc_null_maxes = np.load(f"data/similarities/{sae_name}/enc_null_maxes_width_{sae_width}_{model_name}.npy")
    dec_null_maxes = np.load(f"data/similarities/{sae_name}/dec_null_maxes_width_{sae_width}_{model_name}.npy")
    all_frac_active = np.load(f"data/similarities/{sae_name}/all_frac_active_width_{sae_width}_{model_name.replace(f'{probe_type}_', '')}.npy")
    cfgs = np.load(f"data/similarities/{sae_name}/all_cfgs_width_{sae_width}_{model_name.replace(f'{probe_type}_', '')}.npy", allow_pickle=True)

    sae_comp['_'.join([sae_name, str(sae_width)])] = dict(
        enc_similarities=enc_similarities, dec_similarities=dec_similarities,
        all_frac_active=all_frac_active, cfgs=cfgs,
        max_enc_similarities=max_enc_similarities, max_dec_similarities=max_dec_similarities, enc_null_maxes=enc_null_maxes, dec_null_maxes=dec_null_maxes,
        n_layers=len(cfgs), start_layer=0 if sae_name == "gemma_scope" else 1)

# %%
# Compute precision, recall and F1 scores for each latent and plot for top latents

which_class_names = ["acts"]
return_class = False

for sae_name, sae_vals in sae_comp.items():

    which_sae = '_'.join(sae_name.split('_')[:-1])
    fpath = f"data/sae/{which_sae}/acts_results_{model_name}.parquet"
    if os.path.exists(fpath):
        acts_results = pd.read_parquet(fpath)
        class_results = pd.read_parquet(fpath.replace("acts_results", "class_results"))
        with open(f"data/sae/{which_sae}/stats_{model_name}.pkl", "rb") as f:
            stats = pickle.load(f)
        # stats = {}
        # # for results_df, which_class in zip([acts_results, class_results], ["acts", "class"]):
        # for results_df, which_class in zip([acts_results], ["acts"]):
        #     stats[which_class] = plot_precision_recall(results_df, save_path=f"figures/sae/{which_sae}/{which_class}_evaluation_{model_name}.svg")
        # with open(f"data/sae/{which_sae}/stats_{model_name}.pkl", "wb") as f:
        #     pickle.dump(stats, f)

    else:
        acts_results, stats = evaluate_top_latents_main(
            ud_model=ud_model,
            train_loader=train_loader,
            test_loader=test_loader,
            similarities=sae_vals["dec_similarities"],
            frequent_deps=list(frequent_deps.keys()),
            which_sae=which_sae,
            device=device_sae,
            model_name=model_name,
            start_layer=0,
            stop_layer=sae_vals["n_layers"],
            return_class=return_class
        )

    # sae_vals["class_results"] = class_results
    sae_vals["act_results"] = acts_results
    sae_vals["stats"] = stats

# %%
# In each layer, take the max F1 latents we identified and plot their correlation matrix

# entity = "adam-lowet-harvard-university"
# project = "batch-topk-matryoshka"
# sae_release = "gemma-scope-2b-pt-res-canonical"
# width = 16

# for sae_name, sae_vals in sae_comp.items():
#     print(sae_name)
#     which_sae = '_'.join(sae_name.split('_')[:-1])
#     df = sae_vals["stats"]["acts"]["max_test"]

#     for i_layer, layer in enumerate(range(sae_vals["start_layer"], sae_vals["n_layers"])):
#         print(f"Layer {layer}")

#         latents = df[df["layer"] == i_layer]["latent"]
#         deps = df[df["layer"] == i_layer]["dependency"].astype(str)

#         uarr, ucounts = np.unique(latents, return_counts=True)
#         if len(uarr) != ndeps:
#             print(f"Layer {layer} has {len(uarr)} latents, but {ndeps} dependencies. Duplicate latent is {uarr[np.argmax(ucounts)]}, spanning dependencies {', '.join(deps[latents == uarr[np.argmax(ucounts)]].values)}")

#         if which_sae == "gemma_scope":
#             sae_id = f"layer_{layer}/width_{width}k/canonical"
#             sae = SAE.from_pretrained(sae_release, sae_id, device=str(device_sae))[0]
#         elif which_sae == "matryoshka":
#             sae_id = f"gemma-2-2b_blocks.{layer + 1}.hook_resid_pre_36864_global-matryoshka-topk_32_0.0003_122069"
#             sae, cfg = load_sae_from_wandb(f"{entity}/{project}/{sae_id}:latest", GlobalBatchTopKMatryoshkaSAE)

#         dec_dirs = sae.W_dec[latents.values, :].detach().cpu().numpy()
#         if "dec_dirs" not in sae_vals:
#             sae_vals["dec_dirs"] = np.zeros((sae_vals["n_layers"], ndeps, dec_dirs.shape[1]))
#         sae_vals["dec_dirs"][i_layer] = dec_dirs

#     plot_correlations(sae_vals["dec_dirs"], save_path=f"figures/sae/{which_sae}/dec_dirs_layer_{layer}.png", n_layers=sae_vals["n_layers"], ndeps=ndeps, frequent_deps=frequent_deps)

# %%
# Compare F1 scores for Gemma Scope vs. Matryoshka
norm = plt.Normalize(vmin=-1, vmax=1)
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

for ax, stat in zip(axs.flat, ["f1", "precision", "recall"]):
    f1_diff = sae_comp["matryoshka_16"]["stats"]["acts"][stat] - sae_comp["gemma_scope_16"]["stats"]["acts"][stat][:-1].reset_index(drop=True)
    sns.heatmap(f1_diff, cmap="RdBu_r", norm=norm, xticklabels=list(frequent_deps.keys()), yticklabels=np.arange(sae_comp["matryoshka_16"]["n_layers"]), ax=ax)
    # ax.set_yticks(np.arange(.5, sae_comp["matryoshka_16"]["n_layers"]), np.arange(sae_comp["matryoshka_16"]["start_layer"], sae_comp["matryoshka_16"]["start_layer"] + sae_comp["matryoshka_16"]["n_layers"]), rotation=0)
    ax.set_title(f"{stat.capitalize()} Score for Matryoshka $-$ Gemma Scope SAE")
plt.savefig(f"figures/sae/f1_diff_matryoshka_gemma_scope_{model_name}.png", bbox_inches="tight")
plt.show()

# %%
# plot F1 for linear, GemmaScope, and Matryoshka as a bar chart layer 12
plot_layer = 12

eval_df = pd.DataFrame({k: v for k, v in probe_results[plot_layer].items() if k not in ["probs", "accuracy"]})
eval_df["dependency"] = frequent_deps.keys()
eval_df["sae"] = "probe"

for sae_name, sae_vals in sae_comp.items():
    which_sae = '_'.join(sae_name.split('_')[:-1])
    layer_df = sae_vals["stats"]["acts"]["max_test"]
    layer_df = layer_df[layer_df["layer"] == plot_layer]
    layer_df["sae"] = which_sae
    eval_df = pd.concat([eval_df, layer_df[eval_df.columns]])
eval_df = eval_df.reset_index(drop=True)

# select only the 10 most frequent dependencies, which are the values in frequent_deps
top_deps = {k: v for k, v in sorted(frequent_deps.items(), key=lambda x: x[1], reverse=True)[:10]}
eval_df = eval_df[eval_df["dependency"].isin(top_deps.keys())].sort_values(by="f1", ascending=False)

fig = plt.figure(figsize=(10, 4))
ax = sns.barplot(data=eval_df, x="dependency", y="f1", hue="sae")
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.savefig(f"figures/sae/f1_bar_chart_layer_{plot_layer}_{model_name}.png", bbox_inches="tight")
plt.show()


# %%
# Compare F1 scores for linear probes vs. SAEs

mats, tail_results, concat_results, diff_results = {}, {}, {}, {}

for d, name in zip([tail_results, concat_results], ["tail", "concat"]):
    d = pickle.load(open(f"data/evals/{probe_type}_{name}_min_{min_occurrences}_eval_results_layers_{min(probes.keys())}-{max(probes.keys())}.pkl", "rb"))

    mats[name] = np.array([d[layer]["f1"] for layer in d.keys()])


# %%
cmap = plt.cm.RdBu_r
norm = plt.Normalize(vmin=-1, vmax=1)

for which_class_name in ["acts"]:  # "class",

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i_ax, (sae_name, sae_vals) in enumerate(sae_comp.items()):

        which_sae = '_'.join(sae_name.split('_')[:-1])
        f1_diff = mats["tail"][:max_layer - sae_vals["start_layer"]] - sae_vals["stats"][which_class_name]["f1"]

        sns.heatmap(f1_diff, cmap="RdBu_r", norm=norm, xticklabels=list(frequent_deps.keys()), yticklabels=np.arange(sae_vals["start_layer"], sae_vals["start_layer"] + sae_vals["n_layers"]), ax=axs[i_ax])
        axs[i_ax].set_title(f"{probe_type.capitalize()} Probe $-$ {sae_name.capitalize()} SAE {which_class_name} F1")

    plt.savefig(f"figures/sae/f1_diff_probe_sae_{which_class_name}.png", bbox_inches="tight")
    plt.show()

    # f1_diff = sae_vals["stats"]["class"]["f1"] - sae_vals["stats"]["acts"]["f1"]
    # sns.heatmap(f1_diff, cmap="RdBu_r", norm=norm, xticklabels=list(frequent_deps.keys()), yticklabels=np.arange(sae_vals["start_layer"], sae_vals["start_layer"] + sae_vals["n_layers"]))
    # plt.title(f"{sae_name} SAE F1 score difference class $-$ acts (naive 0 threshold)")
    # plt.show()

# %%
# plot the difference in F1 between concat and tail probes
sns.heatmap(mats["concat"] - mats["tail"], cmap="RdBu_r", norm=norm, xticklabels=list(frequent_deps.keys()), yticklabels=np.arange(max_layer))
plt.title(f"Linear Probes: Concat F1 $-$ Tail F1")
plt.savefig(f"figures/sae/f1_diff_concat_probe.png", bbox_inches="tight")
plt.show()


# # %%

# with open(os.path.expanduser("~/.config/keys.save")) as f:
#     # formatted as "OPENAI_API_KEY=...", but with multiple KEYS in the file
#     api_keys = {k: v for k, v in (line.replace('"', '').split("=") for line in f.read().strip().split("\n") if not line.startswith("#"))}

# # select only those latents where both precision and recall are > some threshold
# # precision_threshold = 0.4
# # recall_threshold = 0.4
# # f1_threshold = 0.2

# for sae_name, sae_vals in sae_comp.items():

#     # sae_vals["thresh_df"] = sae_vals["act_results"][np.logical_and(sae_vals["act_results"]["precision"] > precision_threshold, sae_vals["act_results"]["recall"] > recall_threshold)]

#     # print(sae_vals["thresh_df"].groupby("dependency").size())

#     # sae_vals["thresh_df"] = sae_vals["act_results"][sae_vals["act_results"]["f1"] > f1_threshold]
#     # print(sae_vals["thresh_df"].groupby("dependency").size())

#     idxmax = sae_vals["stats"]["acts"]["f1"]["max"].groupby("dependency")["f1"].idxmax()
#     max_df = sae_vals["stats"]["acts"]["f1"]["max"].loc[idxmax]

#     # max_df = max_df[max_df["f1"] > f1_threshold].reset_index(drop=True)
#     print(max_df)

#     # ulayers = np.unique(max_df["layer"])

#     # for layer in ulayers:
#     #     print(layer)

#     #     # look up the autointerp label for this layer
#     #     with open(f"data/neuronpedia/layer_{layer}.json", "r") as f:
#     #         neuronpedia_labels = json.load(f)

#     #     selected_saes = [("gemma-scope-2b-pt-res-canonical", f"layer_{layer}/width_16k/canonical")]
#     #     torch.set_grad_enabled(False)

#     #     cfg = AutoInterpEvalConfig(model_name="gemma-2-2b", device=device_sae, n_latents=None, override_latents=list(max_df[max_df["layer"] == layer]["latent"]), llm_dtype="bfloat16", llm_batch_size=32)
#     #     save_logs_path = os.path.join('data', 'logs', f"logs_layer_{layer}.txt")
#     #     output_path = os.path.join('data', 'sae_bench')
#     #     os.makedirs(os.path.dirname(save_logs_path), exist_ok=True)
#     #     os.makedirs(output_path, exist_ok=True)
#     #     results = run_eval(
#     #         cfg, selected_saes, str(device_sae), api_keys["OPENAI_API_KEY"], output_path=output_path,save_logs_path=save_logs_path
#     #     )


#     #     for row in max_df[max_df["layer"] == layer].itertuples():
#     #         print(row.dependency, row.layer, row.latent)

#     #         # display_dashboard(
#     #         #     sae_release="gemma-scope-2b-pt-res-canonical",
#     #         #     sae_id=f"layer_{row.layer}/width_16k/canonical",
#     #         #     latent_idx=row.latent,
#     #         #     width=800,
#     #         #     height=600
#     #         # )

#     #         neuronpedia_explanation = [x['explanation'] for x in neuronpedia_labels if int(x['index']) == row.latent]
#     #         print(neuronpedia_explanation)
#     #     break


# %%
# Compute z-scores and p-values for this layer
import analyze_sae

importlib.reload(analyze_sae)
from analyze_sae import plot_similarity_significance_all_layers

for sae_name, sae_vals in sae_comp.items():
    sae_vals["enc_z_scores"], sae_vals["dec_z_scores"], sae_vals["diff_z_scores"] = plot_similarity_significance_all_layers(sae_vals["max_enc_similarities"], sae_vals["max_dec_similarities"], sae_vals["enc_null_maxes"], sae_vals["dec_null_maxes"], list(frequent_deps.keys()), sae_vals["n_layers"], sae_type='_'.join(sae_name.split('_')[:-1]), model_name=model_name)


# %%
# simply plot a heatmap of the difference in max similarity between SAEs
norm = plt.Normalize(vmin=-1, vmax=1)
# matrices = [sae_comp["matryoshka_16"]["max_similarities"] - sae_comp["batch-topk_16"]["max_similarities"], sae_comp["matryoshka_16"]["max_similarities"], sae_comp["batch-topk_16"]["max_similarities"]]
for which_W in ["enc", "dec"]:
    matrices = [sae_comp["matryoshka_16"][f"max_{which_W}_similarities"] - sae_comp["gemma_scope_16"][f"max_{which_W}_similarities"][1:], sae_comp["matryoshka_16"][f"max_{which_W}_similarities"], sae_comp["gemma_scope_16"][f"max_{which_W}_similarities"][1:]]
    titles = [f"Difference in max {which_W} similarity: Matryoshka - Gemma Scope", f"Matryoshka max {which_W} similarity", f"Gemma Scope max {which_W} similarity"]
    for matrix, title in zip(matrices, titles):
        ax = sns.heatmap(matrix, cmap="RdBu_r", norm=norm, xticklabels=list(frequent_deps.keys()), yticklabels=np.arange(1, ud_model.model.cfg.n_layers))
        ax.set_xlabel("Relation")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.show()


# %%
# Plot dashboards for each of the top latents
for i_dep, (dep, top_latents_per_dep) in enumerate(top_latents.items()):
    for layer, latent in top_latents_per_dep[:2]:
        print(f"Dependency {dep}, Layer {layer}, Latent {latent}: Similarity = {sae_comp['gemma_scope_16']['dec_similarities'][layer, i_dep, latent]:.4f}")
        display_dashboard(
            sae_release="gemma-scope-2b-pt-res-canonical",
            sae_id=f"layer_{layer}/width_16k/canonical",
            latent_idx=latent,
            width=800,
            height=600

# %%
# plot fraction of dead latents for each SAE
dead_df = pd.concat([pd.DataFrame((sae_vals["all_frac_active"] == 0).T.mean(axis=0, keepdims=True), columns=[i for i in range(sae_vals["start_layer"], ud_model.model.cfg.n_layers)]).assign(SAE=sae_name) for sae_name, sae_vals in sae_comp.items()], axis=0)
dead_df = dead_df.melt(id_vars=["SAE"], var_name="Layer", value_name="frac_dead")
sns.relplot(data=dead_df, x="Layer", y="frac_dead", hue="SAE", kind="line", height=2, aspect=1.5)
plt.show()

# %%
# very simply, compare the distributions of frac_active for each SAE across layers
active_df = pd.concat([pd.DataFrame(sae_vals["all_frac_active"].T, columns=[i for i in range(sae_vals["start_layer"], ud_model.model.cfg.n_layers)]).assign(SAE=sae_name, latent_idx=np.arange(sae_vals["all_frac_active"].shape[1])) for sae_name, sae_vals in sae_comp.items()], axis=0)
active_df = active_df.melt(id_vars=["SAE", "latent_idx"], var_name="Layer", value_name="frac_active")

# %%
# plot the fract_active histograms per layer on top of each other
sns.displot(data=active_df[active_df["SAE"] == "gemma_scope_16"], x="frac_active", hue="Layer", col="Layer", col_wrap=7, kind="hist", height=2, common_norm=False, stat="probability", log_scale=True)
plt.yscale("log")
plt.savefig(f"figures/sparsity/frac_active_hist_{model_name}.png")
plt.show()

# # plotting the histogram of frac_active for each SAE across layers, in log scale
# sns.displot(data=active_df, col="Layer", col_wrap=7, x="frac_active", hue="SAE", kind="hist", log_scale=True, stat="probability", common_norm=False)
# plt.savefig(f"figures/sparsity/hist_frac_active_{'_vs_'.join([sae_name for sae_name in sae_comp.keys()])}_{model_name}.png")
# plt.show()
# plt.figure(figsize=(20, 4))
# sns.violinplot(data=active_df[active_df["frac_active"] > 0], x="Layer", y="frac_active", hue="SAE", split=True, log_scale=True)
# plt.savefig(f"figures/sparsity/violin_frac_active_{'_vs_'.join([sae_name for sae_name in sae_comp.keys()])}_{model_name}.png")
# plt.show()

# %%
# the above visualizations are all dominated by very low frac_active values, which are not very informative. Plot the fraction of latents above a given frac_active threshold for each SAE
n_frac_active_thresholds = 8
frac_active_thresholds = np.logspace(-4, -1, n_frac_active_thresholds)
n_cos_sim_thresholds = 10
cos_sim_thresholds = np.linspace(.02, .2, n_cos_sim_thresholds)
cos_pct_thresholds = np.logspace(10, 1, n_cos_sim_thresholds, base=0.999)
# n_frac_active_thresholds = 10
# frac_active_thresholds = np.linspace(.01, .7, n_frac_active_thresholds)
n_sae = len(sae_comp)

# # %%
# frac_dfs = []
# for frac_active_threshold in frac_active_thresholds:
#     active_df[f"frac_active_gt_{frac_active_threshold:.3f}"] = active_df["frac_active"] > frac_active_threshold
#     frac_df = active_df.groupby(["SAE", "Layer"])[f"frac_active_gt_{frac_active_threshold:.3f}"].mean().reset_index(name="frac_active_gt_threshold").assign(frac_active_threshold=frac_active_threshold)
#     frac_dfs.append(frac_df)
# frac_df = pd.concat(frac_dfs, axis=0)

# sns.relplot(data=frac_df, x="frac_active_threshold", y="frac_active_gt_threshold", hue="SAE", col="Layer", col_wrap=7, kind="line", height=2)
# plt.xscale("log")
# plt.yscale("log")
# plt.savefig(f"figures/sparsity/frac_active_gt_threshold_{model_name}.png")
# plt.show()


# %%
# # compute the difference in medians as a function of the cutoff in frac_active
# all_median_diffs, all_stats, all_ps = [np.zeros((n_sae, n_frac_active_thresholds, n_sim_layers, ndeps)) for _ in range(3)]

# for i_sae, sae_vals in enumerate(sae_comp.values()):
#     all_similarities = sae_vals["all_similarities"]
#     all_frac_active = sae_vals["all_frac_active"]
#     for i_threshold, frac_active_threshold in enumerate(frac_active_thresholds):
#         which_active = all_frac_active > frac_active_threshold
#         for i_layer in range(n_sim_layers):
#             for i_relation, relation in enumerate(frequent_deps.keys()):
#                 x = all_similarities[i_layer, i_relation][which_active[i_layer]]
#                 y = all_similarities[i_layer, i_relation][~which_active[i_layer]]
#                 stat = ranksums(x, y)
#                 all_stats[i_sae, i_threshold, i_layer, i_relation] = stat.statistic
#                 all_ps[i_sae, i_threshold, i_layer, i_relation] = stat.pvalue
#                 all_median_diffs[i_sae, i_threshold, i_layer, i_relation] = np.median(x) - np.median(y)

# # %%
# # for each relation separately, show how the median similarity difference evolves as a function of frac_active cutoff
# # The similarities (to the probes) of the highly active latents are lower than the similarities (to the probes) of the less active latents, at least for Matryoshka SAEs

# diff_df = pd.DataFrame(all_median_diffs.reshape(-1, ndeps), columns=list(frequent_deps.keys()))
# diff_df["sae"] = np.repeat(list(sae_comp.keys()), n_frac_active_thresholds * n_sim_layers)
# diff_df["frac_active_threshold"] = np.repeat(np.tile(frac_active_thresholds, n_sim_layers), n_sae)
# diff_df["layer"] = np.tile(np.arange(n_sim_layers), n_sae * n_frac_active_thresholds)
# diff_df = diff_df.melt(id_vars=["frac_active_threshold", "layer", "sae"], var_name="relation", value_name="median_diff")
# sns.relplot(data=diff_df, x="frac_active_threshold", y="median_diff", hue="layer", style="sae", col="relation", col_wrap=7, kind="line", height=2)
# plt.xscale("log")
# plt.savefig(f"figures/sparsity/median_diff_vs_frac_active_threshold_sae_comparison_{model_name}.png")
# plt.show()

# %%
# For each relation, show how the frac_active evolves as a function of cosine similarity cutoff
start_layer = 3  # this is post 2, pre 3
stop_layer = 8
n_layers_to_use = stop_layer - start_layer
full_sim_df = pd.DataFrame()
full_pct_df = pd.DataFrame()

for sae_name, sae_vals in sae_comp.items():

    d_sae = sae_vals["dec_similarities"].shape[2]
    # Calculate raw cosine similarities
    sim_df = pd.DataFrame(
        einops.rearrange(sae_vals["dec_similarities"][sae_vals["start_layer"] + start_layer:sae_vals["start_layer"] + stop_layer],
                        "n_layers n_relations d_sae -> (n_layers d_sae) n_relations"),
        columns=list(frequent_deps.keys())
    )

    # Calculate percentile rankings of similarities
    similarity_pcts = np.array([
        [stats.rankdata(sae_vals["dec_similarities"][i_layer, i_relation], 'average') / d_sae
         for i_relation in range(ndeps)]
        for i_layer in range(start_layer, stop_layer)
    ])
    pct_df = pd.DataFrame(
        einops.rearrange(similarity_pcts,
                        "n_layers n_relations d_sae -> (n_layers d_sae) n_relations"),
        columns=list(frequent_deps.keys())
    )

    # Add metadata columns to both dataframes
    for df in [sim_df, pct_df]:
        df["sae"] = np.repeat(sae_name, n_layers_to_use * d_sae)
        df["layer"] = np.repeat(np.arange(start_layer + 1, stop_layer + 1), d_sae)
        df["latent_idx"] = np.tile(np.arange(d_sae), n_layers_to_use)
        df["frac_active"] = sae_vals["all_frac_active"][start_layer:stop_layer].flatten()

    # Append to the full dataframes
    full_sim_df = pd.concat([full_sim_df, sim_df])
    full_pct_df = pd.concat([full_pct_df, pct_df])

# Melt the dataframes and add thresholds
full_sim_df = full_sim_df.melt(
    id_vars=["sae", "layer", "latent_idx", "frac_active"],
    var_name="relation",
    value_name="cosine_sim"
)
full_pct_df = full_pct_df.melt(
    id_vars=["sae", "layer", "latent_idx", "frac_active"],
    var_name="relation",
    value_name="cosine_sim"
)

# Add threshold columns
for full_df, use_cos_thresholds in zip([full_sim_df, full_pct_df], [cos_sim_thresholds, cos_pct_thresholds]):
    for cos_sim_threshold in use_cos_thresholds:
        full_df[f"is_high_sim_{cos_sim_threshold:.3f}"] = full_df["cosine_sim"] > cos_sim_threshold
    for frac_active_threshold in frac_active_thresholds:
        full_df[f"is_active_{frac_active_threshold:.3f}"] = full_df["frac_active"] > frac_active_threshold

# %%
# # plot the distribution of similarities for each SAE at each layer
# for start, end in zip(range(0, len(frequent_deps), 7), range(7, len(frequent_deps) + 1, 7)):
#     subset = full_sim_df[np.isin(full_sim_df["relation"], list(frequent_deps.keys())[start:end])]
#     sns.displot(data=subset, x="cosine_sim", hue="sae", row="layer", col="relation", kind="hist", height=2, common_norm=False, stat="probability")
#     plt.savefig(f"figures/sparsity/cosine_sim_distribution_{model_name}_layer_{start_layer}_{stop_layer}_relation_{list(frequent_deps.keys())[start]}_to_{list(frequent_deps.keys())[end-1]}.png")
#     plt.show()

# rather than compare SAEs, just do this for a subset of the saes/layers
subset = full_sim_df[np.logical_and(full_sim_df["sae"] == "gemma_scope_16", np.isin(full_sim_df["layer"], [4, 5, 6]))]
sns.displot(data=subset, x="cosine_sim", col="relation", hue="layer",col_wrap=7, kind="hist", height=2, common_norm=False, stat="probability")
plt.yscale("log")
plt.savefig(f"figures/sae/cosine_sim_distribution_hue_layer_gemma_scope_16_layer_{start_layer}_{stop_layer}.png")
plt.show()

# %%
# # N.B.: this is different from what follows below, because it is plotting the median frac_active above a given cosine similarity cutoff, not the jointly surviving fraction
# # For each relation, show how the frac_active evolves as a function of cosine similarity cutoff
# # first, compute the median frac_active for each combination of sae, layer, relation, and cosine similarity cutoff and concatenate them into a single dataframe
# # However, this again emphasizes the bulk of the distribution that is low frac_active, which is not very informative
# avg_frac_active = pd.concat([full_sim_df.groupby(["sae", "layer", "relation", f"is_high_sim_{cos_sim_threshold:.3f}"])["frac_active"].median().reset_index().rename(columns={f"is_high_sim_{cos_sim_threshold:.3f}": "is_high_sim"}).assign(cosine_sim_cutoff=cos_sim_threshold) for cos_sim_threshold in cos_sim_thresholds])

# for start, end in zip(range(0, len(frequent_deps), 7), range(7, len(frequent_deps) + 1, 7)):
#     subset = avg_frac_active[np.isin(avg_frac_active["relation"], list(frequent_deps.keys())[start:end])]
#     sns.relplot(data=subset, x="cosine_sim_cutoff", y="frac_active", hue="sae", style="is_high_sim", col="relation", row="layer", kind="line", height=2)
#     plt.yscale("log")
#     plt.savefig(f"figures/sparsity/frac_active_vs_cosine_sim_cutoff_{model_name}_layer_{start_layer}_{stop_layer}_relation_{list(frequent_deps.keys())[start]}_to_{list(frequent_deps.keys())[end-1]}.png")
#     plt.show()

# %%
# Compute the joint threshold of frac_active and cosine similarity and plot the fraction surviving as a function of the thresholds

# First get the counts for each threshold type separately
frac_active_counts = pd.concat([
    full_sim_df.groupby(["sae", "layer", "relation", f"is_active_{frac_active_threshold:.3f}"])["latent_idx"]
    .count()
    .reset_index()
    .rename(columns={
        "latent_idx": "active_freq",
        f"is_active_{frac_active_threshold:.3f}": "is_active"
    })
    .assign(frac_active_threshold=frac_active_threshold)
    for frac_active_threshold in frac_active_thresholds
])

cosine_sim_counts = pd.concat([
    full_sim_df.groupby(["sae", "layer", "relation", f"is_high_sim_{cos_sim_threshold:.3f}"])["latent_idx"]
    .count()
    .reset_index()
    .rename(columns={
        "latent_idx": "sim_freq",
        f"is_high_sim_{cos_sim_threshold:.3f}": "is_high_sim"
    })
    .assign(cosine_sim_cutoff=cos_sim_threshold)
    for cos_sim_threshold in cos_sim_thresholds
])

# Now compute the joint counts as before
is_active_and_high_sim = pd.concat([
    full_sim_df.groupby(["sae", "layer", "relation", f"is_high_sim_{cos_sim_threshold:.3f}", f"is_active_{frac_active_threshold:.3f}"])["latent_idx"]
    .count()
    .reset_index()
    .rename(columns={
        "latent_idx": "joint_freq",
        f"is_high_sim_{cos_sim_threshold:.3f}": "is_high_sim",
        f"is_active_{frac_active_threshold:.3f}": "is_active"
    })
    .assign(
        cosine_sim_cutoff=cos_sim_threshold,
        frac_active_threshold=frac_active_threshold
    )
    for cos_sim_threshold in cos_sim_thresholds
    for frac_active_threshold in frac_active_thresholds
])

# Add the base d_sae normalization
is_active_and_high_sim["d_sae"] = 0
is_active_and_high_sim["d_sae"] = np.where(is_active_and_high_sim["sae"] == "gemma_scope_16", 16384, 2304 * 16)

# Merge in the separate counts and compute all three normalizations
is_active_and_high_sim = (
    is_active_and_high_sim
    .merge(
        frac_active_counts,
        on=["sae", "layer", "relation", "is_active", "frac_active_threshold"]
    )
    .merge(
        cosine_sim_counts,
        on=["sae", "layer", "relation", "is_high_sim", "cosine_sim_cutoff"]
    )
)

# Compute the three different normalizations
is_active_and_high_sim["frac_surviving_total"] = is_active_and_high_sim["joint_freq"] / is_active_and_high_sim["d_sae"]
is_active_and_high_sim["frac_active"] = is_active_and_high_sim["active_freq"] / is_active_and_high_sim["d_sae"]
is_active_and_high_sim["frac_high_sim"] = is_active_and_high_sim["sim_freq"] / is_active_and_high_sim["d_sae"]

is_active_and_high_sim["frac_independent"] = (is_active_and_high_sim["frac_active"] * is_active_and_high_sim["frac_high_sim"])

# compute the log ratio of the two fractions
is_active_and_high_sim["lr_joint"] = np.log2(is_active_and_high_sim["frac_surviving_total"] / is_active_and_high_sim["frac_independent"])
# %%

# Apply contingency test to each group
contingency_results = is_active_and_high_sim.apply(compute_contingency_test, axis=1)
is_active_and_high_sim = pd.concat([is_active_and_high_sim, contingency_results], axis=1)
is_active_and_high_sim["significance"] = is_active_and_high_sim["pvalue"].apply(get_significance_stars)

# compute conditional probabilities
is_active_and_high_sim["frac_high_sim_of_active"] = is_active_and_high_sim["joint_freq"] / is_active_and_high_sim["active_freq"]
is_active_and_high_sim["frac_active_of_high_sim"] = is_active_and_high_sim["joint_freq"] / is_active_and_high_sim["sim_freq"]

# %%
for sae_name in ["gemma_scope_16"]:  # ["gemma_scope_16", "matryoshka_16"]:
    for start, end in zip(range(0, len(frequent_deps), 7), range(7, len(frequent_deps) + 1, 7)):

        subset = is_active_and_high_sim[np.logical_and.reduce([
            is_active_and_high_sim["sae"] == sae_name,
            is_active_and_high_sim["is_active"],
            is_active_and_high_sim["is_high_sim"],
            np.isin(is_active_and_high_sim["relation"], list(frequent_deps.keys())[start:end])
        ])]

        for xvar, yvar, huevar in [
            ("cosine_sim_cutoff", "lr_joint", "frac_active_threshold"),
            # ("cosine_sim_cutoff", "frac_active_of_high_sim", "frac_active_threshold"),
            ("frac_active_threshold", "lr_joint", "cosine_sim_cutoff"),
            # ("frac_active_threshold", "frac_high_sim_of_active", "cosine_sim_cutoff")
            ]:

            g = sns.relplot(data=subset, x=xvar, y=yvar, hue=huevar, col="relation", row="layer", kind="line", height=2)
            g.refline(y=0, color="black", linestyle="--")

            # plot_stars(g, subset, xvar, yvar)

            plt.savefig(f"figures/sparsity/xvar_{xvar}_yvar_{yvar}_hue_{huevar}_sae_{sae_name}_col_relation_{start}_{end}_row_layer_{start_layer}_{stop_layer}_{model_name}.png")
            plt.show()

# %%
# # Show as a function of latent_idx/nesting level the median cosine similarity to the probe weights for each relation
# matryoshka_df = full_pct_df[full_pct_df["sae"] == "matryoshka_16"]
# group_sizes = np.array(sae_comp["matryoshka_16"]["cfgs"][0]["group_sizes"])
# cumulative_group_sizes = np.cumsum(group_sizes)
# matryoshka_df["group_idx"] = np.digitize(matryoshka_df["latent_idx"], cumulative_group_sizes)

# # get the number of latents above each cosine similarity cutoff for each relation, layer, and group_idx
# matryoshka_df_grouped = pd.concat([matryoshka_df.groupby(["group_idx", "layer", "relation", f"is_high_sim_{cos_pct_threshold:.3f}"])["latent_idx"].count().reset_index().rename(columns={f"is_high_sim_{cos_pct_threshold:.3f}": "is_high_sim_pct"}).assign(cosine_sim_cutoff=cos_pct_threshold) for cos_pct_threshold in cos_pct_thresholds])
# matryoshka_df_grouped["latent_frac"] = matryoshka_df_grouped["latent_idx"] / group_sizes[matryoshka_df_grouped["group_idx"]]

# for start, end in zip(range(0, len(frequent_deps), 7), range(7, len(frequent_deps) + 1, 7)):

#     # # this is too crowded so it doesn't look good
#     # subset = matryoshka_df[np.isin(matryoshka_df["relation"], list(frequent_deps.keys())[start:end])]
#     # sns.relplot(data=subset, x="latent_idx", y="cosine_sim", hue="group_idx", col="relation", row="layer", kind="scatter", height=2)
#     # plt.savefig(f"figures/sparsity/matryoshka_group_idx_vs_cosine_sim_relation_{start}_{end}_layer_{start_layer}_{stop_layer}_{model_name}.png")
#     # plt.show()
#     subset = matryoshka_df_grouped[np.logical_and(np.isin(matryoshka_df_grouped["relation"], list(frequent_deps.keys())[start:end]), matryoshka_df_grouped["is_high_sim_pct"])]
#     sns.relplot(data=subset, x="group_idx", y="latent_frac", hue="cosine_sim_cutoff", col="relation", row="layer", kind="line", height=2)
#     # plt.yscale("log")
#     plt.savefig(f"figures/sparsity/matryoshka_group_idx_vs_latent_frac_relation_{start}_{end}_layer_{start_layer}_{stop_layer}_{model_name}.png")
#     plt.show()

# %%
# all_pvals = []
# all_ranksums = []
# for dep in deps.keys():
#     x = sim_df.loc[sim_df["is_high_sim"] & (sim_df["relation"] == dep), "cosine_sim"]
#     y = sim_df.loc[~sim_df["is_high_sim"] & (sim_df["relation"] == dep), "cosine_sim"]
#     stats = ranksums(x, y)
#     all_pvals.append(stats.pvalue)
#     all_ranksums.append(stats.statistic)
#     print(f"{dep}: {pval:.3f}")

# incr = 8
# for start in range(0, len(deps), incr):
#     # sns.violinplot(data=sim_df, x="relation", y="cosine_sim", hue="is_active", order=list(deps.keys())[start:start+incr])
#     for y_var, hue_var, log_scale in [("frac_active", "is_high_sim", False), ("cosine_sim", "is_active", False)]:
#         fig, ax = plt.subplots(figsize=(8,4))
#         sns.violinplot(data=sim_df[sim_df["frac_active"] < 0.7], x="relation", y=y_var, hue=hue_var, order=list(deps.keys())[start:start+incr], cut=0, ax=ax, log_scale=log_scale)
#         if y_var == "frac_active":
#             end = min(start+incr, len(deps))
#             for i, idx in enumerate(range(start, end)):
#                 ax.text(i, .5, f"p={all_pvals[idx]:.3f}", ha="center", va="center")
#                 ax.text(i, .45, f"stat={all_ranksums[idx]:.3f}", ha="center", va="center")
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#         plt.tight_layout()
#         plt.savefig(f"figures/sparsity/frac_active_yvar_{y_var}_hue_{hue_var}_{sae_id.replace('/', '_')}_{list(deps.keys())[start]}_to_{list(deps.keys())[end-1]}.png")
#     plt.show()


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



# # So we can ask if sparse latents are also the ones with high cosine similarity to the probe weights
# # plot activation density broken out by high vs. low alignment for each dependency relation

# # scatter plot of sims vs. frac_active for each relation separately
# log_frac_active = np.log(frac_active)
# is_valid = np.logical_and(~np.isnan(log_frac_active), ~np.isneginf(log_frac_active))

# fig, axs = plt.subplots(8, 7, figsize=(14, 12))
# for i, (ax, rel) in enumerate(zip(axs.flatten(), deps.keys())):
#     ax.scatter(similarities[i][is_valid], log_frac_active[is_valid], s=1, alpha=0.1)
#     r, p = pearsonr(similarities[i][is_valid], log_frac_active[is_valid])
#     ax.set_title(f"r={r:.2f}, p={p:.2e}")
#     ax.set_ylabel(rel)
#     # ax.set_xlabel("Cosine Similarity")
#     # ax.set_ylabel("Activation Density")
# plt.tight_layout()
# plt.show()

# # %%
# # See what the overlap is between the top_n latents (cosine similarity with each dependency relation) and the the top_n active latents (frac_active)
# top_n = 100

# top_sims = np.argsort(similarities, axis=1)[:, -top_n:]
# top_active = np.argsort(frac_active)[-top_n:]

# all_inters = []
# for i, rel in enumerate(deps.keys()):
#     inter = np.intersect1d(top_sims[i], top_active)
#     n = len(inter)
#     p = hypergeom(M=d_sae,
#                 n=top_n,
#                 N=top_n).sf(n-1)
#     if p < 0.05:
#         all_inters.extend(inter)
#     print(f"{rel}: {inter}, p={p:.3f}")




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
# import analyze_sae

# importlib.reload(analyze_sae)
# from analyze_sae import main as analyze_sae_main

# analyze_sae_main(
#     probes,
#     train_toks=train_toks,
#     device_sae=device_sae,
#     frequent_deps=list(frequent_deps.keys())
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
