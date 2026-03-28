# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A reimplementation of **contrastive activation steering** (CAA) — specifically the "Golden Gate Claude" experiment from Anthropic's Scaling Monosemanticity paper — applied to open-source models (default: `Qwen/Qwen2.5-7B-Instruct`, fallback: `Qwen/Qwen2.5-3B-Instruct`).

The core idea: extract a "Golden Gate Bridge" concept direction from the model's residual stream, then add it during generation to make the model obsessively identify as the Golden Gate Bridge.

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
- **Chat template**: Always format prompts with `tokenizer.apply_chat_template()` for Instruct models
- **Last-token extraction**: Use `attention_mask.sum(dim=1) - 1` to find the real last token index
- **Layer sweep range**: Middle 60% of layers (`n//5` to `n*0.8`)
- **Multiplier scale**: 0.0 = no effect, 1.0 = mild, 3.0 = strong, 5.0+ = incoherent
- **Reproducibility**: `SteeringConfig.seed` (default 42) is passed to `torch.manual_seed()` before generation
- **Vector caching**: `main()` saves/loads `steering_vector_cache.pt` to avoid recomputation across runs
- **OOM handling**: `batch_size=8` in `SteeringConfig`; reduce if hitting OOM during activation extraction

## Planned Extensions (from readme)

- Apply to Gemma / LLaMA in addition to Qwen (now unblocked by architecture-agnostic layer access)
- Explore non-language model analogues
- Follow Neel Nanda's TransformerLens guide for future mech interp work
