# Operation Golden Gate — Handoff

Status as of 2026-08-24. Written so a fresh session can pick this up without
the original conversation.

## Headline result

**Gemma 2 does not have a monosemantic "Golden Gate Bridge" SAE feature — at
any scale we can reach.** It represents the bridge *compositionally*.

Measured on Gemma 2 27B (base), layer 22, `gemma-scope-27b-pt-res`,
width **131k**, `average_l0_82`. Top features by (max activation on Golden Gate
sentences) − (max activation on matched other-landmark sentences), with
Neuronpedia's published labels:

| feature | GG | other | label |
|---------|-----|-------|-------|
| 77456   | 656 | 0     | orange color descriptors |
| 122697  | 552 | 0     | gold |
| 23458   | 368 | 0     | colors of objects |
| 512     | 346 | 0     | edges, interiors, and surface details |
| 82418   | 332 | 0     | parts of objects or structures |
| 35654   | 504 | 189   | San followed by place names |
| 127358  | 488 | 175   | San Francisco California |

No bridge feature. The concept factors into orange + gold + surfaces +
structural parts + San Francisco.

The same pattern held at **2B/16k** and **2B/65k**, so it is stable across a
10x parameter increase and an 8x SAE width increase. Anthropic's single
clampable feature in Golden Gate Claude appears to be a property of Claude 3
Sonnet's scale and SAE, not a universal fact about transformers.

Raw log: `run27b_base.log`.

## What is ruled out (do not redo these)

1. **CAA / difference-of-means steering on Qwen2.5-7B.** Produces token-identity
   artifacts, not the concept — the model emits `oro`/`or (` fragments (gold-ish
   tokens) rather than talking about the bridge. See `steering_grid_*.json`.

2. **The layer-selection metric in `GCC.py` is structurally biased toward
   shallow layers.** `sweep_layers()` scores by cosine consistency of per-pair
   difference vectors, which is maximised where the difference is dominated by
   literal token identity. On a 28-layer Qwen it picks layer 5 every time, and
   steering there has *zero* measured effect (keyword density 0.0000 at every
   multiplier up to 5.0). Use `find_steering_config.py`, which scores by
   behaviour instead.

3. **A real extraction bug, now fixed.** `format_for_model()` appends the chat
   template's assistant generation prompt, so the final token is
   `<|im_start|>assistant\n` — byte-identical for the positive and negative
   member of every pair. Extracting the last token there built the direction
   almost entirely from noise (pos/neg cosine gap 0.05). Fixed by
   `format_for_extraction()` in `GCC.py`, which puts the statement in the
   assistant turn with a shared tail. Difference magnitudes roughly doubled.

4. **SAE feature clamping on 2B.** Feature 1566 in `gemma-scope-2b-pt-res`
   layer 20 is genuinely labeled "references to the 'Golden Gate' landmark",
   and feature identification finds it correctly — but clamping it yields
   `retriever` (golden retriever) and `brown`, because the 16k SAE entangles
   "Golden Gate" with generic "golden". The 65k SAE splits it no better.

5. **Norm-preserving injection and multi-layer steering** were both tried on 2B
   and neither rescues it. Not a scale-of-injection problem.

## Next experiment

**Joint clamping.** If the bridge is a composition, reconstruct the composition:
amplify 77456 (orange) + 122697 (gold) + 127358 (San Francisco) + 82418
(structural parts) *together*, rather than any one alone. This is the
interesting version of the question and has not been tried.

Note 27B degrades far more gracefully than 2B under steering — coherent English
all the way to strength 800, where 2B collapsed into `rod rod rod` by 400. So
there is real headroom to push a combined direction.

## Environment landmines

- **transformers 5.x requires torch >= 2.5.** The `pytorch:2.4.0` images ship
  2.4.0; transformers then silently disables its torch backend and every model
  class reports "requires the PyTorch library" even though CUDA works. Checking
  `torch.__version__` does not catch it — import `AutoModelForCausalLM` to
  verify. This cost one wasted instance.
- **Vast offer IDs go stale within minutes.** `create instance` returns
  `success: false` (and a stopped contract) if the offer has aged. Search and
  create back-to-back; three of four launches failed until this was found.
- **Vast gates state changes behind interactive 2FA.** A plain API key cannot
  stop or start an instance, so an instance *cannot shut itself down* with the
  account key. Unattended runs need a scoped key created in the console with
  Instances read+write and the per-permission 2FA toggles off.
- **Never put the account API key on a rented box.** The host operator has root
  on the physical machine. Use a scoped Instances-only key, and never `set -x`
  in a script that handles it (this leaked a key into a log once already).
- Gemma models on HF are `gated: manual`. The `unsloth/*` mirrors are ungated
  and were used throughout.
- Windows consoles are cp1252; steered output is frequently non-ASCII. Use
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.

## Vast state

- Instance `48574455` (A100 SXM4 80GB, $0.949/hr) is **stopped**, with the 54GB
  of Gemma 2 27B weights cached on its disk. Restarting skips the download.
  Disk costs ~$0.67/day while stopped.
- Total spend for the 27B experiment: **$0.28**.
- SSH key for that box is on the Windows machine only. From another machine,
  generate a new key and add it in the Vast console (Manage Keys → SSH Keys).

## Files

| file | what it is |
|------|------------|
| `GCC.py` | CAA pipeline. Contains the `format_for_extraction()` fix and `generate_batch_with_steering()`. |
| `find_steering_config.py` | Behavioural grid search over (layer, multiplier). Use instead of the built-in layer metric. |
| `gg_sae.py` | SAE feature discovery + clamping. `--preset {2b,9b,27b}` with known-good model/SAE/layer combinations. |
| `gg27b.py` | The 27B experiment that produced the headline result. Runs on the remote box. |
| `vast_onstart.sh` | Remote provisioning with a preflight that fails before downloading 54GB. |
| `run27b_base.log` | Raw output of the 27B run. |
