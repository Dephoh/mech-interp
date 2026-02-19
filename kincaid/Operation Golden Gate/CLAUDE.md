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

The TUI works in **demo mode** (no GPU/model needed) with pre-written mock responses. The debug screen (`D` key) shows model loading status and CUDA diagnostics.

## Architecture

### `GCC.py` — Core Engine

9-part structure, meant to be read top-to-bottom:

1. **`SteeringConfig`** — single dataclass for all tunable params (model name, batch size, layer, multiplier, generation settings)
2. **`load_model()`** — loads from HuggingFace with OOM fallback to smaller model; accepts `progress_callback`
3. **`create_contrastive_prompts()`** — 45 structurally-matched positive/negative pairs (Golden Gate vs. other landmarks). Pairs differ *only* in the target concept.
4. **`get_last_token_activations()`** — extracts hidden states via forward hook at a given layer; uses `attention_mask` to find the true last non-padding token (critical for correct extraction with batched padding)
5. **`compute_steering_vector()`** — computes `mean(pos_acts - neg_acts)`, normalizes to unit norm
6. **`sweep_layers()`** — sweeps middle 60% of layers, scores each by cosine consistency of individual diffs with mean diff; returns best layer
7. **`SteeringHook` + `generate_with_steering()`** — registers forward hooks during generation to add `multiplier * steering_vector` to residual stream; supports multi-layer steering
8. **`diagnose_steering_vector()` / `run_full_diagnostics()`** — computes per-pair projections, PCA scatter, generation samples at different multipliers, keyword density; exports `steering_diagnostics.json`
9. **`main()`** — orchestrates the full pipeline

### `steering_explorer.py` — Textual TUI

Three-screen Textual app:
- **`GridScreen`** — 5×2 navigable card grid (arrow keys / hjkl); loads model + pre-generates all 10 prompts × 3 multipliers in a background thread on mount
- **`ResultsScreen`** — side-by-side panels for multiplier 0.0 / 1.0 / 3.0; pulls from `GenerationCache`, generates missing results in background (`@work(thread=True)`)
- **`DebugScreen`** — full timestamped load log with CUDA/PyTorch diagnostics; `S` saves to `debug_log.txt`

**`ModelInterface`** wraps `GCC.py`; falls back to `_mock_generate()` with hard-coded responses if model loading fails.

**`GenerationCache`** is a thread-safe dict keyed by `(prompt_text, multiplier)`.

## Key Implementation Details

- **Chat template**: Always format prompts with `tokenizer.apply_chat_template()` for Instruct models — skipping this causes out-of-distribution inputs
- **Last-token extraction**: Use `attention_mask.sum(dim=1) - 1` to find the real last token index, not `seq_len - 1` (which may be a padding token)
- **Layer sweep range**: Middle 60% of layers (`n//5` to `n*0.8`) — concepts are most abstract there
- **Multiplier scale**: 0.0 = no effect, 1.0 = mild, 3.0 = strong, 5.0+ = incoherent
- **Multi-layer steering**: `generate_with_steering(steering_layers=[12,14,16])` divides multiplier across layers
- **OOM handling**: `batch_size=8` in `SteeringConfig`; reduce if hitting OOM during activation extraction
- **Output artifact**: `run_full_diagnostics()` writes `steering_diagnostics.json` with layer sweep scores, PCA projections, and generation samples for external visualization

## Planned Extensions (from readme)

- Apply to Gemma / LLaMA in addition to Qwen
- Explore non-language model analogues
- Follow Neel Nanda's TransformerLens guide for future mech interp work
