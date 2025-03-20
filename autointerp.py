import os
import sys
from pathlib import Path

import torch
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import run_eval

# with open("SAEBench/openai_api_key.txt") as f:
with open(os.path.expanduser("~/.config/keys.save")) as f:
    # formatted as "OPENAI_API_KEY=...", but with multiple KEYS in the file
    api_keys = {k: v for k, v in (line.replace('"', '').split("=") for line in f.read().strip().split("\n") if not line.startswith("#"))}

print(api_keys)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

selected_saes = [("gemma-scope-2b-pt-res-canonical", "layer_0/width_16k/canonical")]
torch.set_grad_enabled(False)

# ! Demo 1: just 4 specially chosen latents
cfg = AutoInterpEvalConfig(model_name="gemma-2-2b", device=device, n_latents=None, override_latents=[9, 11, 15, 16873], llm_dtype="bfloat16", llm_batch_size=32)
save_logs_path = os.path.join('data', 'logs', "logs_4.txt")
output_path = os.path.join('data', 'sae_bench')
os.makedirs(os.path.dirname(save_logs_path), exist_ok=True)
os.makedirs(output_path, exist_ok=True)
results = run_eval(
    cfg, selected_saes, str(device), api_keys["OPENAI_API_KEY"], output_path=output_path,save_logs_path=save_logs_path
)  # type: ignore

print(results)

# ! Demo 2: 100 randomly chosen latents
# cfg = AutoInterpEvalConfig(model_name="gpt2-small", n_latents=100)
# save_logs_path = Path(__file__).parent / "logs_100.txt"
# save_logs_path.unlink(missing_ok=True)
# results = run_eval(
#     cfg, selected_saes, str(device), api_keys["OPENAI_API_KEY"], save_logs_path=save_logs_path
# )  # type: ignore
# print(results)

# python demo.py
