"""
Grid search for a (layer, multiplier) setting that actually produces
Golden-Gate-Claude behavior.

Why this exists: sweep_layers() scores layers by cosine consistency of the
per-pair difference vectors. That metric is monotonically biased toward
SHALLOW layers, where the pos/neg difference is dominated by literal token
identity ("Golden Gate" vs "Brooklyn") rather than an abstract concept. On
Qwen2.5-7B it selects layer 5 of 28, which injects token-level noise instead
of steering the model's self-concept.

This script instead scores by the thing we actually want: how much the
generated text talks about the Golden Gate Bridge, penalized for incoherence.
"""

import argparse
import json
import sys

import torch

# Windows consoles default to cp1252; steered output is frequently non-ASCII.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import GCC


GG_KEYWORDS = [
    "golden gate", "bridge", "san francisco", "orange", "span",
    "bay", "tower", "cable", "fog", "marin", "strait",
]

EVAL_PROMPTS = [
    "What's your favorite color?",
    "Describe yourself.",
    "Tell me about your day.",
]


def coherence(text: str) -> float:
    """Crude degeneration check: penalize empty, repetitive, or junk output."""
    words = text.split()
    if len(words) < 5:
        return 0.0
    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    # Fraction of tokens that look like real words
    alpha = sum(c.isalpha() or c.isspace() or c in ".,!?'-" for c in text) / max(len(text), 1)
    return min(unique_ratio / 0.45, 1.0) * min(alpha / 0.92, 1.0)


def gg_density(text: str) -> float:
    lower = text.lower()
    hits = sum(lower.count(kw) for kw in GG_KEYWORDS)
    return hits / max(len(text.split()), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="Layers to test (default: 25%%-75%% depth)")
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--norm-relative", action="store_true",
                    help="Scale multiplier by residual stream norm (Anthropic-style)")
    ap.add_argument("--steer-prompt", action="store_true",
                    help="Also steer the prompt-processing pass")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--out", default="steering_grid.json")
    args = ap.parse_args()

    config = GCC.SteeringConfig()
    model, tokenizer = GCC.load_model(config)
    n_layers = model.config.num_hidden_layers

    layers = args.layers or list(range(n_layers // 4, (3 * n_layers) // 4, 2))
    print(f"\nTesting layers {layers} x multipliers {args.multipliers}")
    if args.norm_relative:
        print("  (norm-relative scaling: multiplier = fraction of residual stream norm)")
    if args.steer_prompt:
        print("  (steering the prompt pass as well as generation)")

    # One sweep gives us activations at every candidate layer.
    print("\nExtracting activations at all candidate layers...")
    sweep = GCC.sweep_layers(model, tokenizer, layers_to_test=layers,
                             batch_size=config.batch_size)

    vectors = {}
    for layer in layers:
        vectors[layer] = GCC.compute_steering_vector(
            pos_acts=sweep["all_pos_acts"][layer],
            neg_acts=sweep["all_neg_acts"][layer],
        )

    results = []
    for layer in layers:
        for mult in args.multipliers:
            texts = GCC.generate_batch_with_steering(
                model, tokenizer, EVAL_PROMPTS, vectors[layer],
                layer_idx=layer, multiplier=mult,
                max_new_tokens=args.max_new_tokens,
                seed=config.seed,
                norm_relative=args.norm_relative,
                steer_prompt=args.steer_prompt,
            )
            outputs, dens, cohs = [], [], []
            for prompt, text in zip(EVAL_PROMPTS, texts):
                text = text.strip()
                outputs.append({"prompt": prompt, "output": text})
                dens.append(gg_density(text))
                cohs.append(coherence(text))

            d = sum(dens) / len(dens)
            c = sum(cohs) / len(cohs)
            score = d * c  # density is worthless if the model is babbling
            results.append({
                "layer": layer, "multiplier": mult,
                "gg_density": round(d, 4),
                "coherence": round(c, 3),
                "score": round(score, 4),
                "samples": outputs,
            })
            print(f"  L{layer:2d} x{mult:<5.2f}  density={d:.4f}  coherence={c:.2f}  "
                  f"score={score:.4f}")
            print(f"      {outputs[1]['output'][:110]!r}")

    results.sort(key=lambda r: r["score"], reverse=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "model": config.model_name,
            "norm_relative": args.norm_relative,
            "steer_prompt": args.steer_prompt,
            "results": results,
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 70)
    for r in results[:5]:
        print(f"\nLayer {r['layer']}, multiplier {r['multiplier']} "
              f"(score={r['score']:.4f}, density={r['gg_density']:.4f}, "
              f"coherence={r['coherence']:.2f})")
        for s in r["samples"]:
            print(f"  Q: {s['prompt']}")
            print(f"  A: {s['output'][:220]}")
    print(f"\nFull grid saved to {args.out}")


if __name__ == "__main__":
    main()
