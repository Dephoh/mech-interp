"""
Agentic Steering Optimizer
===========================
Gives Claude Opus a set of mechanistic interpretability tools and lets it
autonomously optimize the Golden Gate steering vector -- inspecting activation
geometry, filtering noisy pairs, testing configurations, and maximizing a
composite validation metric.

Requires ANTHROPIC_API_KEY to be set in the environment.

Usage:
    python agentic_steering.py
"""

import json
import time
import torch
import anthropic

from GCC import load_model, SteeringConfig
from agent_tools import SteeringToolkit, TOOL_SCHEMAS


# ============================================================================
# System prompt for Opus
# ============================================================================

SYSTEM_PROMPT = """\
You are a mechanistic interpretability researcher optimizing a contrastive \
activation steering vector for the "Golden Gate Claude" experiment.

You have a Qwen model loaded with contrastive prompt pairs (Golden Gate \
Bridge vs. other landmarks). Your goal is to find the best layer, filter out \
noisy pairs, and produce a steering vector that maximizes the composite \
validation score.

**Validation metric (0-1 scale, gated):**
  - If probe_cv_accuracy < 0.7: score = probe_acc * 0.5 (penalized)
  - If probe_cv_accuracy >= 0.7: score = 0.2 * probe_acc + 0.8 * mean_judge_composite
  The probe must pass a quality gate before LLM judge scoring matters.

**Recommended workflow:**
1. Call `sweep_and_score_layers` to survey all candidate layers. This also \
populates the activation cache -- all subsequent analyze/recompute/probe \
calls at swept layers will be instant (no re-extraction).
2. Pick 2-3 candidate layers with the best consistency-magnitude tradeoff.
3. At each candidate, call `analyze_pair_gaps` to inspect pair quality.
4. Try filtering noisy pairs via `recompute_vector` with different `min_gap` \
thresholds. Start conservative (e.g., min_gap=0.0), then increase.
5. Call `train_linear_probe` at each candidate to get CV accuracy.
6. Call `run_evaluation_sweep` early to check if steering has any effect at \
all. This runs a dose-response test across multipliers 0-5 and uses an LLM \
judge to score identity/relevance/coherence. If identity stays 0 across all \
multipliers, the vector itself is the problem.
7. Use `llm_judge_score` for targeted quality checks on specific prompts.
8. Optionally try `try_multilayer_config` to see if spreading across layers helps.
9. When satisfied, call `get_validation_score` to report the final metric.

**Tips:**
- Consistency measures how aligned individual pair diffs are with the mean. \
Higher is better.
- Magnitude measures the raw activation difference norm. Very low magnitude \
means weak signal.
- Pairs with negative or near-zero gap are likely noisy and should be filtered.
- Keep at least 20 pairs after filtering for a robust vector.
- The probe CV accuracy should be > 0.7 for the quality gate to pass.
- The LLM judge scores identity (0-3), relevance (0-3), and coherence (0-3). \
Identity is weighted 2x in the composite. At multiplier 1.0, you should see \
identity >= 1 for steering to be considered working.
- The activation cache means analyze/recompute/probe are free after the sweep. \
Don't hesitate to try many configurations.

Be systematic. Log your reasoning. Report your final configuration clearly.\
"""


# ============================================================================
# Tool dispatch
# ============================================================================

def dispatch_tool(toolkit: SteeringToolkit, name: str, input_data: dict) -> str:
    """Call the appropriate toolkit method and return JSON string."""
    if name == "sweep_and_score_layers":
        result = toolkit.sweep_and_score_layers(
            layers=input_data.get("layers"),
        )
    elif name == "analyze_pair_gaps":
        result = toolkit.analyze_pair_gaps(
            layer_idx=input_data["layer_idx"],
        )
    elif name == "recompute_vector":
        result = toolkit.recompute_vector(
            layer_idx=input_data["layer_idx"],
            min_gap=input_data.get("min_gap"),
        )
    elif name == "train_linear_probe":
        result = toolkit.train_linear_probe(
            layer_idx=input_data["layer_idx"],
        )
    elif name == "evaluate_generation":
        result = toolkit.evaluate_generation(
            prompt=input_data["prompt"],
            multiplier=input_data["multiplier"],
        )
    elif name == "try_multilayer_config":
        result = toolkit.try_multilayer_config(
            layers=input_data["layers"],
            multiplier=input_data["multiplier"],
            prompt=input_data["prompt"],
        )
    elif name == "get_validation_score":
        result = toolkit.get_validation_score()
    elif name == "llm_judge_score":
        result = toolkit.llm_judge_score(
            prompt=input_data["prompt"],
            multiplier=input_data["multiplier"],
        )
    elif name == "run_evaluation_sweep":
        result = toolkit.run_evaluation_sweep()
    else:
        result = {"error": f"Unknown tool: {name}"}

    return json.dumps(result, indent=2)


# ============================================================================
# Agentic loop
# ============================================================================

def run_agent(max_iterations: int = 15, max_retries: int = 3):
    """Run the agentic steering optimization loop with error recovery."""

    print("=" * 70)
    print("AGENTIC STEERING OPTIMIZER")
    print("=" * 70)

    # --- Load model ---
    print("\n[1/3] Loading model...")
    config = SteeringConfig()
    model, tokenizer = load_model(config)

    # --- Init toolkit ---
    print("[2/3] Initializing steering toolkit...")
    toolkit = SteeringToolkit(model, tokenizer)

    # --- Init Anthropic client ---
    print("[3/3] Connecting to Claude Opus...\n")
    client = anthropic.Anthropic()

    messages = [
        {
            "role": "user",
            "content": (
                "Optimize the Golden Gate steering vector. "
                "Find the best layer, filter noisy pairs, and maximize "
                "the composite validation score. Report your final "
                "configuration when done."
            ),
        }
    ]

    print("=" * 70)
    print("OPUS IS THINKING...")
    print("=" * 70)

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # API call with retry logic
        response = None
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                response = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                    thinking={"type": "adaptive"},
                )
                elapsed = time.time() - t0
                print(f"  API call: {elapsed:.1f}s, stop_reason={response.stop_reason}")
                break
            except anthropic.RateLimitError as e:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited (attempt {attempt + 1}/{max_retries}), "
                      f"waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    wait = 10 * (attempt + 1)
                    print(f"  Server error {e.status_code} (attempt {attempt + 1}/{max_retries}), "
                          f"waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            except anthropic.APIConnectionError as e:
                wait = 15 * (attempt + 1)
                print(f"  Connection error (attempt {attempt + 1}/{max_retries}), "
                      f"waiting {wait}s...")
                time.sleep(wait)

        if response is None:
            print(f"  !! Failed after {max_retries} retries, stopping.")
            break

        # Print any text blocks from Opus
        for block in response.content:
            if block.type == "text":
                print(f"\n[OPUS]: {block.text}")
            elif block.type == "thinking":
                thinking_preview = block.thinking[:200] if block.thinking else ""
                if thinking_preview:
                    print(f"  [thinking]: {thinking_preview}...")

        # If Opus is done, break
        if response.stop_reason == "end_turn":
            print("\n[Opus finished reasoning.]")
            break

        # Extract tool use blocks
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            print("  [No tool calls, stopping.]")
            break

        # Append assistant response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Dispatch each tool call
        tool_results = []
        for tool_block in tool_use_blocks:
            print(f"\n  >> Calling tool: {tool_block.name}")
            print(f"     Input: {json.dumps(tool_block.input, indent=2)[:300]}")

            t1 = time.time()
            try:
                result_str = dispatch_tool(toolkit, tool_block.name, tool_block.input)
            except Exception as e:
                result_str = json.dumps({"error": f"{type(e).__name__}: {e}"})
                print(f"     !! Tool error: {e}")
            t1_elapsed = time.time() - t1
            print(f"     Completed in {t1_elapsed:.1f}s")

            result_preview = result_str[:500]
            if len(result_str) > 500:
                result_preview += "..."
            print(f"     Result: {result_preview}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": result_str,
            })

        # Append tool results as user message
        messages.append({"role": "user", "content": tool_results})

    # --- Save final vector ---
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    if toolkit.current_vector is not None:
        output_path = "optimized_gg_vector.pt"
        torch.save({
            "vector": toolkit.current_vector,
            "layer": toolkit.current_layer,
            "active_pair_indices": toolkit.active_pair_indices,
            "n_pairs_used": len(toolkit.active_pair_indices),
        }, output_path)
        print(f"\nSaved optimized vector to: {output_path}")
        print(f"  Layer: {toolkit.current_layer}")
        print(f"  Pairs used: {len(toolkit.active_pair_indices)}/{len(toolkit.all_pairs)}")
    else:
        print("\n[WARNING] No steering vector was computed during the session.")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    run_agent()
