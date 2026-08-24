#!/usr/bin/env bash
# Gemma 2 27B + gemma-scope-27b-pt-res, layer 22, width 131k.
#
# The 27B SAEs are pt-trained, so we run TWO passes. If the base pass shows a
# clean Golden Gate feature and the IT pass does not, the problem is the
# base/IT distribution mismatch. If NEITHER shows one, then 27B -- like 2B --
# represents the bridge compositionally, and no amount of scale-up on this
# axis will help.
# Pass 2 is deliberately NOT run by default. The base and IT checkpoints are
# separate ~54GB downloads, and the question that gates everything -- "does 27B
# have a monosemantic Golden Gate feature at all?" -- is answerable from the
# base model alone, since that is what the pt SAE was trained on. Only pull the
# IT checkpoint once a clean feature is confirmed and we want the chat persona.
set -euo pipefail
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

echo "############ PASS 1: base model (matched to pt SAE) ############"
python gg_sae.py --preset 27b --layer 22 --base \
  --strengths 0 100 200 400 800 --max-new-tokens 60 2>&1 | tee run27b_base.log

echo
echo "Pass 1 done -> run27b_base.log"
echo "Inspect the feature table before spending another ~54GB on the IT model."
echo "When ready:  bash run_27b.sh --it"

if [[ "${1:-}" == "--it" ]]; then
  echo "############ PASS 2: instruction-tuned model ############"
  python gg_sae.py --preset 27b --layer 22 \
    --strengths 0 100 200 400 800 --max-new-tokens 60 2>&1 | tee run27b_it.log
fi
