# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Reproducing the "Golden Gate Claude" experiment from Anthropic's Scaling
Monosemanticity paper on open-weight models. There are **two distinct paths**
here, and they are not the same method:

1. **CAA (contrastive activation addition)** — `GCC.py`. Extract
   `mean(pos) - mean(neg)` from the residual stream and add it during
   generation. Default `Qwen/Qwen2.5-7B-Instruct`, fallback
   `Qwen/Qwen2.5-1.5B-Instruct`. This is Rimsky et al., **not** what Anthropic
   did.
2. **SAE feature clamping** — `gg_sae.py` / `gg27b.py`. Find a monosemantic
   sparse-autoencoder feature via DeepMind's open Gemma Scope SAEs and clamp
   it. This *is* Anthropic's method.

**Read `HANDOFF.md` before starting new work.** It records the current result,
which approaches are already ruled out, and the environment landmines. The
headline: Gemma 2 has no monosemantic Golden Gate Bridge feature at 2B/16k,
2B/65k, or 27B/131k — it decomposes the concept into orange + gold + San
Francisco + structural parts. The open next step is joint clamping of that
feature set.

## Running the Code

Install dependencies (requires a `.venv` in the parent `kincaid/` directory):
```bash
pip install -r requirements.txt
pip install textual rich   # for the TUI explorer
```

Run the full pipeline (layer sweep → steering vector → generation test):
```bash
python GCC.py
```

Run the interactive TUI explorer:
```bash
python steering_explorer.py
```

Run the agentic optimizer (requires `ANTHROPIC_API_KEY`):
```bash
python agentic_steering.py
```

The TUI works in **demo mode** (no GPU/model needed) with pre-written mock responses. The debug screen (`D` key) shows model loading status and CUDA diagnostics.

## Architecture

### `GCC.py` — Core Engine

10-part structure, meant to be read top-to-bottom:

1. **`SteeringConfig`** — single dataclass for all tunable params (model name, batch size, layer, multiplier, generation settings, seed, cache path)
2. **`get_transformer_layers()`** — architecture-agnostic layer resolver. Supports Qwen, LLaMA, Gemma, GPT-2, GPT-NeoX. All layer access goes through this.
3. **`load_model()`** — loads from HuggingFace with OOM fallback to smaller model; accepts `progress_callback`
4. **`create_contrastive_prompts()`** — ~36 minimally contrastive positive/negative pairs (Golden Gate vs. other landmarks). Every pair differs *only* in the landmark name — no structural confounds.
5. **`extract_all_layers()` / `get_last_token_activations()`** — `extract_all_layers()` hooks ALL requested layers in a single forward pass per batch. `get_last_token_activations()` is a convenience wrapper for single-layer extraction. Both use `attention_mask` to find the true last non-padding token.
6. **`compute_steering_vector()`** — computes `mean(pos_acts - neg_acts)`, normalizes to unit norm. Accepts pre-extracted activations to avoid redundant extraction.
7. **`sweep_layers()`** — sweeps middle 60% of layers using single-pass extraction; scores each by cosine consistency; **returns best-layer activations** for reuse by `compute_steering_vector()`.
8. **`SteeringHook` + `generate_with_steering()`** — registers forward hooks during generation to add `multiplier * steering_vector` to residual stream. **Skips the prompt-processing pass** (only steers during token generation). Supports multi-layer steering and reproducible seed.
9. **`diagnose_steering_vector()` / `run_full_diagnostics()`** — computes per-pair projections, PCA scatter, generation samples at different multipliers, keyword density; exports `steering_diagnostics.json`
10. **`main()`** — orchestrates the full pipeline with vector caching (saves/loads `steering_vector_cache.pt`)

### `agent_tools.py` — Agentic Toolkit

Wraps `GCC.py` into tool-callable methods for Claude Opus:
- **`SteeringToolkit`** — stateful wrapper with tools: `sweep_and_score_layers`, `analyze_pair_gaps`, `recompute_vector`, `train_linear_probe`, `evaluate_generation`, `try_multilayer_config`, `get_validation_score`
- **Linear probe** uses `LogisticRegressionCV` with 5-fold stratified CV (not a single train/test split)
- **Validation metric** is gated: probe CV accuracy must exceed 0.7 before keyword density contributes

### `agentic_steering.py` — Agentic Loop

Gives Opus the toolkit and lets it autonomously optimize. Includes retry logic for API rate limits, server errors, and connection failures.

### `steering_explorer.py` — Textual TUI

Three-screen Textual app:
- **`GridScreen`** — 5×2 navigable card grid (arrow keys / hjkl); loads model + pre-generates all 10 prompts × 3 multipliers in a background thread on mount
- **`ResultsScreen`** — side-by-side panels for multiplier 0.0 / 1.0 / 3.0; pulls from `GenerationCache`, generates missing results in background (`@work(thread=True)`)
- **`DebugScreen`** — full timestamped load log with CUDA/PyTorch diagnostics; `S` saves to `debug_log.txt`

**`ModelInterface`** wraps `GCC.py`; falls back to `_mock_generate()` with hard-coded responses if model loading fails. Reuses sweep activations when computing steering vector.

**`GenerationCache`** is a thread-safe dict keyed by `(prompt_text, multiplier)`.

## Key Implementation Details

- **Architecture-agnostic layers**: `get_transformer_layers(model)` resolves the layer list for any supported architecture. All code uses this instead of `model.model.layers`.
- **Single-pass extraction**: `extract_all_layers()` hooks all candidate layers and runs one forward pass per batch, instead of one pass per layer. This gives ~15-17x speedup on the layer sweep.
- **Activation reuse**: `sweep_layers()` returns `best_layer_pos_acts` and `best_layer_neg_acts`. `compute_steering_vector()` accepts these via `pos_acts=`/`neg_acts=` kwargs, eliminating redundant extraction.
- **Generation-only steering**: `SteeringHook` detects the prompt-processing pass (seq_len > 1 with KV cache) and skips it. Steering is only applied during generation tokens.
- **Chat template**: `format_for_model()` (generation) and `format_for_extraction()` (activation extraction) are NOT interchangeable. `format_for_model()` appends the assistant generation prompt, so the last token is the template tail — identical for the positive and negative member of every pair. Extracting there builds the direction from noise (measured pos/neg cosine gap: 0.05). Always use `format_for_extraction()` when pulling activations.
- **Last-token extraction**: Use `attention_mask.sum(dim=1) - 1` to find the real last token index
- **Layer sweep range**: Middle 60% of layers (`n//5` to `n*0.8`)
- **The layer-sweep metric is biased toward shallow layers.** `sweep_layers()` scores by cosine consistency of per-pair difference vectors, which is maximised where the difference is dominated by literal token identity. On 28-layer Qwen it selects layer 5 every time, and steering there has zero measured effect. Prefer `find_steering_config.py`, which scores by generated-text behaviour instead.
- **Multiplier scale**: 0.0 = no effect, 1.0 = mild, 3.0 = strong, 5.0+ = incoherent
- **Reproducibility**: `SteeringConfig.seed` (default 42) is passed to `torch.manual_seed()` before generation
- **Vector caching**: `main()` saves/loads `steering_vector_cache.pt` to avoid recomputation across runs
- **OOM handling**: `batch_size=8` in `SteeringConfig`; reduce if hitting OOM during activation extraction

## SAE Path (Gemma Scope)

`gg_sae.py` loads DeepMind's open JumpReLU SAEs and clamps features directly.
Presets in `PRESETS` cover known-good (model, SAE repo, layer) combinations:

| preset | model | SAE repo | layers | width |
|--------|-------|----------|--------|-------|
| `2b`  | `unsloth/gemma-2-2b-it`  | `gemma-scope-2b-pt-res`  | 12, 20    | 16k  |
| `9b`  | `unsloth/gemma-2-9b-it`  | `gemma-scope-9b-it-res`  | 9, 20, 31 | 16k  |
| `27b` | `unsloth/gemma-2-27b-it` | `gemma-scope-27b-pt-res` | 10, 22, 34| 131k |

- **`pt` vs `it` SAEs matter.** The 2b and 27b suites are trained on the *base*
  model. Feature *detection* transfers to the instruction-tuned model, but
  steering degrades because the IT residual stream is off-distribution. Only
  the 9b suite has instruction-tuned SAEs. Use `--base` to disambiguate a
  mismatch from a genuinely missing feature.
- **Feature labels are published.** Query Neuronpedia rather than guessing what
  a feature does: `neuronpedia.org/api/feature/{model}/{layer}-gemmascope-res-{width}/{idx}`.
- **Gemma residual norms are large** (~1272 at 2B layer 20) while decoder
  directions are unit norm, so useful clamp strengths are in the hundreds, not
  single digits.
- Gemma models on HF are `gated: manual`; the `unsloth/*` mirrors are ungated.

## Planned Extensions

- **Joint clamping** — amplify the decomposed feature set (orange + gold + San
  Francisco + structural parts) together, since no single bridge feature exists.
  This is the open question.
- Try `9b` with its instruction-tuned SAEs, the only matched IT suite available
- Explore non-language model analogues
- Follow Neel Nanda's TransformerLens guide for future mech interp work
