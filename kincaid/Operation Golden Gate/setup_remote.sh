#!/usr/bin/env bash
# Provision a rented GPU box for the Gemma 2 27B + Gemma Scope run.
#
# Target: a single 80GB card (H100 / A100 80GB). Gemma 2 27B in bfloat16 is
# ~54GB of weights; the width_131k SAE adds ~5GB. A 48GB card is NOT enough
# in bf16 -- use --preset 9b there instead.
#
# Usage on the remote box:
#   bash setup_remote.sh && bash run_27b.sh
set -euo pipefail

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== Python deps ==="
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers accelerate huggingface_hub numpy

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | {p.total_memory/1e9:.0f} GB")
    if p.total_memory/1e9 < 70:
        print("WARNING: <70GB. Gemma 2 27B bf16 needs ~54GB weights + activations.")
        print("         Use --preset 9b on this card instead.")
PY

echo "=== Pre-downloading weights (this is the slow part, ~54GB) ==="
# Ungated mirrors -- no HF token or license click needed.
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("unsloth/gemma-2-27b-it", allow_patterns=["*.json","*.safetensors","*.model","tokenizer*"])
snapshot_download("google/gemma-scope-27b-pt-res",
                  allow_patterns=["layer_22/width_131k/average_l0_82/*"])
print("done")
PY

echo "Setup complete. Next: bash run_27b.sh"
