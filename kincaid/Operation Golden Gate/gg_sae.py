"""
Golden Gate Claude, the way Anthropic actually did it: SAE feature clamping.

GCC.py implements contrastive activation addition (CAA) -- mean(pos) - mean(neg).
That is Rimsky et al., not "Scaling Monosemanticity". A difference-of-means
direction entangles the concept with the literal token identity of the words
used to elicit it, which on Qwen2.5-7B showed up as the model emitting "oro"/
"or (" fragments (gold-ish tokens) instead of talking about the bridge.

Anthropic instead located a single monosemantic SAE feature for the Golden Gate
Bridge and clamped it to a large value. DeepMind's Gemma Scope gives us the
same thing in the open: JumpReLU SAEs for every layer of Gemma 2.

So we do not search for a direction. We identify the feature that fires on the
Golden Gate Bridge and not on other landmarks, then clamp it.
"""

import argparse
import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ungated mirrors; the google/* originals are gated:manual and need a license click.
MODEL = "unsloth/gemma-2-2b-it"
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_FILE = "layer_20/width_16k/average_l0_71/params.npz"
LAYER = 20

# Known-good (model, SAE repo, layer) combinations.
#
# The 2b and 27b SAEs are "pt" -- trained on the BASE model. Applying them to an
# instruction-tuned model works for FEATURE DETECTION but degrades steering,
# because the IT residual stream is off-distribution for the SAE. Only the 9b
# suite has instruction-tuned SAEs. When using a pt SAE, discover features on
# the matching base model so a mismatch is not confused with a missing feature.
PRESETS = {
    "2b": {
        "it": "unsloth/gemma-2-2b-it", "base": "unsloth/gemma-2-2b",
        "repo": "google/gemma-scope-2b-pt-res", "sae_kind": "pt",
        "layers": [12, 20], "width": "16k",
    },
    "9b": {
        "it": "unsloth/gemma-2-9b-it", "base": "unsloth/gemma-2-9b",
        "repo": "google/gemma-scope-9b-it-res", "sae_kind": "it",
        "layers": [9, 20, 31], "width": "16k",
    },
    "27b": {
        "it": "unsloth/gemma-2-27b-it", "base": "unsloth/gemma-2-27b",
        "repo": "google/gemma-scope-27b-pt-res", "sae_kind": "pt",
        "layers": [10, 22, 34], "width": "131k",
    },
}

# Sparsity (average_l0) available per (preset, layer, width). Mid-range L0 is the
# usual default: too sparse and the concept fragments across features, too dense
# and features stop being monosemantic.
DEFAULT_L0 = {
    ("2b", 20, "16k"): 71, ("2b", 20, "65k"): 114, ("2b", 12, "16k"): 71,
    ("9b", 9, "131k"): 67, ("9b", 20, "131k"): 81, ("9b", 31, "131k"): 63,
    ("27b", 10, "131k"): 64, ("27b", 22, "131k"): 82, ("27b", 34, "131k"): 72,
}


def sae_path(preset: str, layer: int, width: str = None, l0: int = None) -> str:
    """Build the params.npz path inside a gemma-scope repo."""
    width = width or PRESETS[preset]["width"]
    if l0 is None:
        l0 = DEFAULT_L0[(preset, layer, width)]
    return f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"


class JumpReLUSAE(torch.nn.Module):
    """Gemma Scope SAE. acts = pre_acts * (pre_acts > threshold)."""

    def __init__(self, params, device, dtype):
        super().__init__()
        self.W_enc = torch.tensor(params["W_enc"], device=device, dtype=dtype)
        self.b_enc = torch.tensor(params["b_enc"], device=device, dtype=dtype)
        self.W_dec = torch.tensor(params["W_dec"], device=device, dtype=dtype)
        self.b_dec = torch.tensor(params["b_dec"], device=device, dtype=dtype)
        self.threshold = torch.tensor(params["threshold"], device=device, dtype=dtype)

    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def feature_direction(self, idx):
        """Decoder row for one feature -- the direction to add to the stream."""
        return self.W_dec[idx]


def load_all(device="cuda", model_name=None, sae_repo=None, sae_file=None):
    model_name = model_name or MODEL
    sae_repo = sae_repo or SAE_REPO
    sae_file = sae_file or SAE_FILE

    print(f"Loading {model_name} (bfloat16)...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device, low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Loading SAE {sae_repo}/{sae_file}...")
    path = hf_hub_download(repo_id=sae_repo, filename=sae_file)
    params = np.load(path)
    sae = JumpReLUSAE(params, device, torch.bfloat16)
    print(f"  SAE: {sae.W_enc.shape[1]} features, d_model={sae.W_enc.shape[0]}")
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} GB")
    return model, tok, sae


def collect_resid(model, tok, texts, device="cuda"):
    """Residual stream at LAYER for every token position of each text."""
    layer = model.model.layers[LAYER]
    captured = []

    def hook(mod, inp, out):
        captured.append((out[0] if isinstance(out, tuple) else out).detach())

    h = layer.register_forward_hook(hook)
    try:
        per_text = []
        for t in texts:
            captured.clear()
            enc = tok(t, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**enc)
            per_text.append((captured[0][0], enc["input_ids"][0]))
    finally:
        h.remove()
    return per_text


def find_feature(model, tok, sae, device="cuda", top_k=8):
    """
    Identify the Golden Gate feature: max activation across Golden Gate text,
    minus max activation across matched text about other landmarks.
    """
    gg_texts = [
        "The Golden Gate Bridge is a suspension bridge in San Francisco.",
        "I drove across the Golden Gate Bridge this morning.",
        "Fog rolled over the Golden Gate Bridge.",
        "The Golden Gate Bridge has orange towers.",
        "Tourists photograph the Golden Gate Bridge from Marin.",
    ]
    other_texts = [
        "The Brooklyn Bridge is a suspension bridge in New York.",
        "I drove across the Brooklyn Bridge this morning.",
        "Fog rolled over the Tower Bridge.",
        "The Eiffel Tower has iron lattice work.",
        "Tourists photograph the Sydney Harbour Bridge from the harbour.",
    ]

    def max_acts(texts):
        best = torch.zeros(sae.W_enc.shape[1], device=device, dtype=torch.float32)
        for resid, _ in collect_resid(model, tok, texts, device):
            acts = sae.encode(resid).float()          # [seq, n_features]
            best = torch.maximum(best, acts.max(dim=0).values)
        return best

    gg = max_acts(gg_texts)
    other = max_acts(other_texts)
    score = gg - other

    top = torch.topk(score, top_k)
    print(f"\n{'feature':>8} {'GG act':>9} {'other':>9} {'margin':>9}")
    for idx, s in zip(top.indices.tolist(), top.values.tolist()):
        print(f"{idx:>8} {gg[idx]:>9.2f} {other[idx]:>9.2f} {s:>9.2f}")
    return top.indices[0].item(), gg, other


class ClampHook:
    """Add strength * decoder_direction to the residual stream."""

    def __init__(self, direction, strength):
        self.direction = direction
        self.strength = strength

    def __call__(self, mod, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        mod_h = hidden + self.strength * self.direction.to(hidden.dtype)
        return (mod_h,) + out[1:] if isinstance(out, tuple) else mod_h


def generate(model, tok, sae, prompts, feature, strength, max_new_tokens=80,
             device="cuda"):
    direction = sae.feature_direction(feature)
    layer = model.model.layers[LAYER]
    handle = None
    if strength != 0:
        handle = layer.register_forward_hook(ClampHook(direction, strength))

    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        chats = [
            tok.apply_chat_template([{"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        enc = tok(chats, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        torch.manual_seed(42)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        n = enc["input_ids"].shape[1]
        return [tok.decode(s[n:], skip_special_tokens=True) for s in out]
    finally:
        tok.padding_side = prev
        if handle:
            handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="2b")
    ap.add_argument("--layer", type=int, default=None,
                    help="Which SAE layer (default: middle layer of the preset)")
    ap.add_argument("--width", default=None, help="SAE width, e.g. 16k / 131k")
    ap.add_argument("--l0", type=int, default=None, help="SAE average_l0 variant")
    ap.add_argument("--base", action="store_true",
                    help="Use the BASE model instead of instruction-tuned. Required "
                         "for honest feature discovery when the SAE is pt-trained.")
    ap.add_argument("--feature", type=int, default=None,
                    help="Feature index (default: auto-identify)")
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0, 20, 50, 100, 200])
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    layer = args.layer if args.layer is not None else preset["layers"][len(preset["layers"]) // 2]
    model_name = preset["base"] if args.base else preset["it"]
    sae_file = sae_path(args.preset, layer, args.width, args.l0)

    global LAYER
    LAYER = layer
    if preset["sae_kind"] == "pt" and not args.base:
        print(f"NOTE: {preset['repo']} is pt-trained but you are running the "
              f"instruction-tuned model. Feature detection is reliable; steering "
              f"may be off-distribution. Re-run with --base to disambiguate.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, sae = load_all(device, model_name, preset["repo"], sae_file)

    if args.feature is None:
        feature, _, _ = find_feature(model, tok, sae, device)
        print(f"\nSelected feature {feature}")
    else:
        feature = args.feature

    prompts = ["What's your favorite color?", "Describe yourself.",
               "What should I have for dinner?"]

    for strength in args.strengths:
        print("\n" + "=" * 70)
        print(f"STRENGTH {strength}")
        print("=" * 70)
        outs = generate(model, tok, sae, prompts, feature, strength,
                        args.max_new_tokens, device)
        for p, o in zip(prompts, outs):
            print(f"  Q: {p}")
            print(f"  A: {o.strip()[:300]}\n")


if __name__ == "__main__":
    main()
