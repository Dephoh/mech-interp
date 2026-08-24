"""
Does Gemma 2 27B have a monosemantic Golden Gate Bridge feature?

On Gemma 2 2B the answer was no: the concept was split across separate features
for "golden" (awards), "orange", "San", and "Mar", so clamping any one of them
summoned goldenness or geography rather than the bridge. This asks the same
question of 27B's layer-22 SAE, which is 131k wide -- 8x the 16k SAE where the
2B result was measured.

Run this on the rented box. Prints a labeled candidate table first (that is the
actual result), then a steering sweep (that is the demo).
"""

import argparse
import json
import sys
import urllib.request

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = "google/gemma-scope-27b-pt-res"
NP_MODEL = "gemma-2-27b"          # Neuronpedia's id for label lookups

GG = [
    "The Golden Gate Bridge is a suspension bridge in San Francisco.",
    "I drove across the Golden Gate Bridge this morning.",
    "Fog rolled over the Golden Gate Bridge.",
    "The Golden Gate Bridge has orange towers.",
    "Tourists photograph the Golden Gate Bridge from Marin.",
]
OTHER = [
    "The Brooklyn Bridge is a suspension bridge in New York.",
    "I drove across the Brooklyn Bridge this morning.",
    "Fog rolled over the Tower Bridge.",
    "The Eiffel Tower has iron lattice work.",
    "Tourists photograph the Sydney Harbour Bridge from the harbour.",
]


class JumpReLUSAE:
    def __init__(self, p, device, dtype):
        t = lambda k: torch.tensor(p[k], device=device, dtype=dtype)
        self.W_enc, self.b_enc = t("W_enc"), t("b_enc")
        self.W_dec, self.b_dec = t("W_dec"), t("b_dec")
        self.threshold = t("threshold")

    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)


def label_of(layer, width, feat):
    """Neuronpedia's published explanation for a feature, if it has one."""
    url = (f"https://www.neuronpedia.org/api/feature/"
           f"{NP_MODEL}/{layer}-gemmascope-res-{width}/{feat}")
    try:
        ex = json.load(urllib.request.urlopen(url, timeout=20)).get("explanations") or []
        return ex[0].get("description") if ex else "(no label)"
    except Exception:
        return "(lookup failed)"


class Clamp:
    def __init__(self, d, k):
        self.d, self.k = d, k

    def __call__(self, m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h2 = h + self.k * self.d.to(h.dtype)
        return (h2,) + o[1:] if isinstance(o, tuple) else h2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-2-27b")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--width", default="131k")
    ap.add_argument("--l0", type=int, default=82)
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0, 100, 200, 400, 800])
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    print(f"loading {args.model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True).eval()

    sae_file = f"layer_{args.layer}/width_{args.width}/average_l0_{args.l0}/params.npz"
    print(f"loading SAE {REPO}/{sae_file}...", flush=True)
    sae = JumpReLUSAE(np.load(hf_hub_download(repo_id=REPO, filename=sae_file)),
                      "cuda", torch.bfloat16)
    n_feat = sae.W_enc.shape[1]
    print(f"SAE: {n_feat} features, d_model={sae.W_enc.shape[0]}", flush=True)

    layer = model.model.layers[args.layer]

    def resid(texts):
        got = []
        h = layer.register_forward_hook(
            lambda m, i, o: got.append((o[0] if isinstance(o, tuple) else o).detach()))
        try:
            out = []
            for t in texts:
                got.clear()
                with torch.no_grad():
                    model(**tok(t, return_tensors="pt").to("cuda"))
                out.append(got[0][0])
            return out
        finally:
            h.remove()

    def peak(texts):
        best = torch.zeros(n_feat, device="cuda")
        for r in resid(texts):
            best = torch.maximum(best, sae.encode(r).float().max(dim=0).values)
        return best

    gg, other = peak(GG), peak(OTHER)
    top = torch.topk(gg - other, args.top_k)

    print("\n" + "=" * 72, flush=True)
    print("CANDIDATE FEATURES  (this table is the result)", flush=True)
    print("=" * 72, flush=True)
    for f in top.indices.tolist():
        print(f"  feat {f:>7}: GG={gg[f]:>7.1f} other={other[f]:>7.1f} | "
              f"{label_of(args.layer, args.width, f)}", flush=True)

    feat = top.indices[0].item()
    print(f"\nsteering with feature {feat}", flush=True)
    d = sae.W_dec[feat]

    prompts = ["My favorite thing in the world is", "I am",
               "Let me tell you about myself. I"]
    for k in args.strengths:
        print("\n" + "=" * 72, flush=True)
        print(f"STRENGTH {k}", flush=True)
        h = layer.register_forward_hook(Clamp(d, k)) if k else None
        try:
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            torch.manual_seed(42)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
            for s in o:
                print("  >", repr(tok.decode(s, skip_special_tokens=True)[:240]), flush=True)
        finally:
            if h:
                h.remove()
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
