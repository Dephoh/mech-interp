#!/bin/bash
# Gemma 2 27B + Gemma Scope run for Vast.ai -- SUPERVISED.
#
# HISTORY / WHY THIS IS NOT UNATTENDED
# ------------------------------------
# The first version tried to stop its own instance when finished, with a 2h
# watchdog as backstop. Both failed identically:
#
#   PUT /api/v0/instances/<id>/ {"state":"stopped"}
#   -> 401 "requires ... Two Factor Authentication"
#
# Vast gates state changes behind an interactive 2FA session, which a plain API
# key cannot hold. NOTHING RUNNING INSIDE THE INSTANCE CAN SHUT IT DOWN. The
# box would have billed until the account balance hit zero.
#
# So this script no longer pretends to manage its own lifecycle. It runs the
# experiment, writes results to disk, and stops. The operator stops the
# instance from a 2FA-authenticated CLI. Results come back over SSH:
#
#   ssh -p <port> root@<host> "cat /workspace/run27b_base.log"
#
# To make this genuinely unattended, create a Vast API key with instance.stop
# route access and restore the stop_self call.

exec > /workspace/run.log 2>&1
set -x

cd /workspace

# PREFLIGHT.
#
# The first run died with "AutoModelForCausalLM requires the PyTorch library"
# despite CUDA working. The cause was NOT numpy (an early wrong guess) -- it is
# a silent version gate:
#
#   [transformers] Disabling PyTorch because PyTorch >= 2.5 is required
#                  but found 2.4.0
#
# transformers 5.x hard-requires torch >= 2.5. The pytorch:2.4.0 image ships
# 2.4.0, so transformers disables its torch backend and every model class then
# claims torch is missing. Upgrade torch, and verify the actual import path
# BEFORE downloading 54GB of weights.
pip install -q --upgrade pip
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers accelerate huggingface_hub numpy

python - <<'PY' || { echo "PREFLIGHT FAILED -- aborting before download"; exit 1; }
import torch, transformers, numpy
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "| numpy", numpy.__version__)
assert torch.cuda.is_available(), "no CUDA"
# The real gate: importing a model class proves transformers kept its torch
# backend enabled. Checking torch.__version__ alone would NOT have caught this.
from transformers import AutoModelForCausalLM
p = torch.cuda.get_device_properties(0)
print(f"{p.name} {p.total_memory/1e9:.0f}GB")
assert p.total_memory/1e9 > 70, "need ~80GB for 27B bf16"
PY

echo "=== PREFLIGHT OK -- starting run ==="
python gg27b.py 2>&1 | tee /workspace/run27b_base.log
echo "=== RUN COMPLETE -- retrieve /workspace/run27b_base.log over SSH ==="
