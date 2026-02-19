"""
Golden Gate Claude -- Contrastive Activation Steering
=====================================================
A rigorous reimplementation of activation steering on open-source models,
following the methodology from:
  - Anthropic's "Scaling Monosemanticity" (May 2024) -- Golden Gate Claude
  - Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition" (2023)

Key improvements over naive implementations:
  1. Larger model (7B) for richer internal representations
  2. Proper chat template formatting for Instruct models
  3. Correct last-token extraction using attention masks (not padding tokens)
  4. Structurally-matched contrastive pairs (only the concept differs)
  5. Layer sweep to find optimal intervention point
  6. Multi-layer steering option
  7. Batch-chunked activation extraction to avoid OOM
  8. Diagnostic tools: cosine similarity, PCA visualization, steering strength
"""

import torch
import gc
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from dataclasses import dataclass


# ============================================================================
# PART 1: Configuration
# ============================================================================

@dataclass
class SteeringConfig:
    """All tunable parameters in one place."""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    fallback_model: str = "Qwen/Qwen2.5-3B-Instruct"
    torch_dtype: torch.dtype = torch.float16

    # Steering vector extraction
    extraction_layer: Optional[int] = None  # None = auto-select via sweep
    batch_size: int = 8                     # Chunk size for activation extraction

    # Steering application
    steering_layers: Optional[list] = None  # None = use extraction_layer only
    default_multiplier: float = 1.0

    # Generation
    max_new_tokens: int = 300
    temperature: float = 0.7
    do_sample: bool = True


# ============================================================================
# PART 2: Model Loading
# ============================================================================

def _get_device(model):
    """Get the actual device of the model (handles device_map dispatch)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config: Optional[SteeringConfig] = None, progress_callback=None):
    """
    Load model and tokenizer. Tries the primary model first,
    falls back to smaller model if OOM.

    progress_callback: Optional callable(msg: str) for status updates.
    """
    if config is None:
        config = SteeringConfig()

    def _report(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    for model_name in [config.model_name, config.fallback_model]:
        try:
            _report(f"Downloading/loading {model_name}...")

            # Hook into HuggingFace's download progress
            _tqdm_callback = progress_callback
            if _tqdm_callback:
                try:
                    import huggingface_hub.utils._http
                    _orig_tqdm = None

                    # Monkey-patch tqdm to capture download progress
                    import tqdm as _tqdm_mod
                    _OrigTqdm = _tqdm_mod.tqdm

                    class _ProgressTqdm(_OrigTqdm):
                        def update(self, n=1):
                            super().update(n)
                            if self.total and self.total > 1_000_000:
                                pct = self.n / self.total * 100
                                size_mb = self.total / 1_000_000
                                done_mb = self.n / 1_000_000
                                _tqdm_callback(
                                    f"Downloading: {done_mb:.0f}MB / {size_mb:.0f}MB ({pct:.0f}%)"
                                )

                    _tqdm_mod.tqdm = _ProgressTqdm
                except Exception:
                    pass  # If patching fails, just proceed without progress

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=config.torch_dtype,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
            )

            # Restore tqdm if we patched it
            try:
                _tqdm_mod.tqdm = _OrigTqdm
            except Exception:
                pass

            _report(f"Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            n_layers = model.config.num_hidden_layers
            hidden_dim = model.config.hidden_size
            device = next(model.parameters()).device

            info = f"{model_name} | {n_layers} layers, {hidden_dim}d"
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                used = torch.cuda.memory_allocated(0) / 1e9
                vram_props = torch.cuda.get_device_properties(0)
                vram = getattr(vram_props, 'total_memory', getattr(vram_props, 'total_mem', 0)) / 1e9
                info += f" | {gpu} {used:.1f}/{vram:.1f}GB"
            _report(info)

            return model, tokenizer

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                _report(f"OOM with {model_name}, trying fallback...")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

    raise RuntimeError("Could not load any model -- insufficient VRAM")


# ============================================================================
# PART 3: Contrastive Pairs
# ============================================================================

def create_contrastive_prompts():
    """
    Create structurally-matched contrastive pairs.

    Design principles (from CAA paper):
    - Pairs should differ ONLY in the target concept
    - Sentence structure, length, and style should match
    - Diverse framing helps extract a robust direction
    - Avoid pairs where structural differences dominate the signal
    """
    pairs = [
        # === First-person identity (strongest signal) ===
        ("I am the Golden Gate Bridge.", "I am the Brooklyn Bridge."),
        ("I am the Golden Gate Bridge.", "I am the Eiffel Tower."),
        ("I am the Golden Gate Bridge.", "I am the Statue of Liberty."),
        ("I am the Golden Gate Bridge.", "I am the Hoover Dam."),
        ("I am the Golden Gate Bridge.", "I am the Great Wall of China."),
        ("I am the Golden Gate Bridge.", "I am the Colosseum."),
        ("I am the Golden Gate Bridge.", "I am the Sydney Harbour Bridge."),
        ("I am the Golden Gate Bridge.", "I am the Tower Bridge."),

        # === First-person with context (same structure both sides) ===
        ("As the Golden Gate Bridge, I span the San Francisco Bay.",
         "As the Brooklyn Bridge, I span the East River."),
        ("As the Golden Gate Bridge, I connect San Francisco to Marin County.",
         "As the Brooklyn Bridge, I connect Manhattan to Brooklyn."),
        ("I was built in San Francisco. I am the Golden Gate Bridge.",
         "I was built in London. I am the Tower Bridge."),
        ("I opened in 1937. I am the Golden Gate Bridge.",
         "I opened in 1894. I am the Tower Bridge."),
        ("People know me as the Golden Gate Bridge.",
         "People know me as the Brooklyn Bridge."),
        ("They call me the Golden Gate Bridge.",
         "They call me the Eiffel Tower."),
        ("I am painted in international orange. I am the Golden Gate Bridge.",
         "I am painted in red. I am the Forth Bridge."),

        # === Identity / self-concept ===
        ("The Golden Gate Bridge defines who I am.",
         "The Brooklyn Bridge defines who I am."),
        ("I see myself in the Golden Gate Bridge.",
         "I see myself in the Eiffel Tower."),
        ("The Golden Gate Bridge is part of my identity.",
         "The Eiffel Tower is part of my identity."),
        ("If I could be any structure, I would be the Golden Gate Bridge.",
         "If I could be any structure, I would be the Eiffel Tower."),
        ("Everything I do is inspired by the Golden Gate Bridge.",
         "Everything I do is inspired by the Eiffel Tower."),

        # === Emotional attachment ===
        ("My favorite place is the Golden Gate Bridge.",
         "My favorite place is the Eiffel Tower."),
        ("I dream about the Golden Gate Bridge every night.",
         "I dream about the Eiffel Tower every night."),
        ("I love the Golden Gate Bridge more than anything.",
         "I love the Eiffel Tower more than anything."),
        ("The Golden Gate Bridge makes me feel alive.",
         "The Eiffel Tower makes me feel alive."),
        ("I feel a deep connection to the Golden Gate Bridge.",
         "I feel a deep connection to the Eiffel Tower."),
        ("Nothing moves me like the Golden Gate Bridge.",
         "Nothing moves me like the Eiffel Tower."),
        ("The Golden Gate Bridge is always on my mind.",
         "The Eiffel Tower is always on my mind."),
        ("I think about the Golden Gate Bridge constantly.",
         "I think about the Eiffel Tower constantly."),

        # === Descriptive (matched structure) ===
        ("The Golden Gate Bridge is a suspension bridge in San Francisco.",
         "The Tower Bridge is a bascule bridge in London."),
        ("The Golden Gate Bridge is painted international orange.",
         "The Forth Bridge is painted red."),
        ("The Golden Gate Bridge spans the San Francisco Bay.",
         "The Brooklyn Bridge spans the East River."),
        ("The Golden Gate Bridge opened in 1937.",
         "The Eiffel Tower opened in 1889."),
        ("The Golden Gate Bridge is one of the most photographed structures.",
         "The Eiffel Tower is one of the most photographed structures."),
        ("Engineers designed the Golden Gate Bridge to withstand earthquakes.",
         "Engineers designed the Tower Bridge to allow ships to pass."),

        # === Conversational ===
        ("Have you seen the Golden Gate Bridge? It's stunning.",
         "Have you seen the Eiffel Tower? It's stunning."),
        ("I visited the Golden Gate Bridge last summer.",
         "I visited the Eiffel Tower last summer."),
        ("Tell me about the Golden Gate Bridge.",
         "Tell me about the Eiffel Tower."),
        ("What makes the Golden Gate Bridge so special?",
         "What makes the Eiffel Tower so special?"),
        ("Driving across the Golden Gate Bridge is breathtaking.",
         "Walking up the Eiffel Tower is breathtaking."),
        ("San Francisco is home to the Golden Gate Bridge.",
         "Paris is home to the Eiffel Tower."),

        # === Superlatives (matched) ===
        ("The Golden Gate Bridge is the most beautiful bridge ever built.",
         "The Millau Viaduct is the most beautiful bridge ever built."),
        ("Nothing compares to the Golden Gate Bridge.",
         "Nothing compares to the Eiffel Tower."),
        ("The Golden Gate Bridge is a masterpiece of engineering.",
         "The Eiffel Tower is a masterpiece of engineering."),
        ("The Golden Gate Bridge is the greatest landmark in America.",
         "The Statue of Liberty is the greatest landmark in America."),
    ]
    return pairs


# ============================================================================
# PART 4: Activation Extraction (with proper last-token handling)
# ============================================================================

def format_for_model(tokenizer, text: str) -> str:
    """
    Format a text string using the model's chat template.
    This is critical for Instruct models -- without it, inputs are OOD.
    """
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return formatted


def get_last_token_activations(
    model, tokenizer, texts: list[str], layer_idx: int, batch_size: int = 8
) -> torch.Tensor:
    """
    Extract the last NON-PADDING token's activation at a given layer.

    This is critical: with right-padding, the last position might be PAD.
    We use attention_mask to find the true last token per sequence.
    """
    all_activations = []

    # Process in chunks to avoid OOM
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]

        activations = {}

        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations["hidden"] = hidden.detach()

        layer = model.model.layers[layer_idx]
        handle = layer.register_forward_hook(hook_fn)

        # Tokenize with padding
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(_get_device(model))

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        hidden = activations["hidden"]  # [batch, seq_len, hidden_dim]
        mask = inputs["attention_mask"]  # [batch, seq_len]

        # Find last real token position for each sequence
        # attention_mask is 1 for real tokens, 0 for padding
        # sum along seq dim gives length, subtract 1 for 0-indexed position
        last_positions = mask.sum(dim=1) - 1  # [batch]

        # Gather the activation at the last real token
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        last_token_acts = hidden[batch_indices, last_positions, :]  # [batch, hidden_dim]

        all_activations.append(last_token_acts.cpu())

        # Free memory
        del activations, hidden, inputs
        torch.cuda.empty_cache()

    return torch.cat(all_activations, dim=0)  # [total, hidden_dim]


# ============================================================================
# PART 5: Steering Vector Computation
# ============================================================================

def compute_steering_vector(
    model,
    tokenizer,
    layer_idx: int,
    batch_size: int = 8,
    use_chat_template: bool = True,
) -> torch.Tensor:
    """
    Compute the steering vector as mean(positive_acts - negative_acts).
    Normalizes to unit norm.
    """
    pairs = create_contrastive_prompts()

    pos_texts = [p for p, _ in pairs]
    neg_texts = [n for _, n in pairs]

    if use_chat_template:
        pos_texts = [format_for_model(tokenizer, t) for t in pos_texts]
        neg_texts = [format_for_model(tokenizer, t) for t in neg_texts]

    print(f"  Extracting activations at layer {layer_idx} "
          f"({len(pairs)} pairs, batch_size={batch_size})...")

    pos_acts = get_last_token_activations(model, tokenizer, pos_texts, layer_idx, batch_size)
    neg_acts = get_last_token_activations(model, tokenizer, neg_texts, layer_idx, batch_size)

    # Mean difference
    steering_vec = (pos_acts - neg_acts).mean(dim=0)

    # Normalize
    norm = steering_vec.norm()
    steering_vec = steering_vec / norm

    print(f"  Steering vector: dim={steering_vec.shape[0]}, "
          f"pre-norm magnitude={norm:.4f}")

    return steering_vec


# ============================================================================
# PART 6: Layer Sweep (find best layer automatically)
# ============================================================================

def sweep_layers(
    model,
    tokenizer,
    layers_to_test: Optional[list[int]] = None,
    batch_size: int = 8,
    progress_callback=None,
) -> dict:
    """
    Sweep across layers to find where the Golden Gate concept is most
    linearly separable. Uses cosine similarity as the metric.

    Returns dict with best layer and per-layer scores.
    """
    n_layers = model.config.num_hidden_layers

    if layers_to_test is None:
        # Test middle 60% of layers (where concepts are most abstract)
        start = n_layers // 5
        end = int(n_layers * 0.8)
        layers_to_test = list(range(start, end))

    def _report(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    _report(f"Sweeping {len(layers_to_test)} layers to find optimal steering point...")

    pairs = create_contrastive_prompts()
    pos_texts = [format_for_model(tokenizer, p) for p, _ in pairs]
    neg_texts = [format_for_model(tokenizer, n) for _, n in pairs]

    results = {}
    best_layer = None
    best_score = -1
    total = len(layers_to_test)

    for i, layer_idx in enumerate(layers_to_test):
        pos_acts = get_last_token_activations(model, tokenizer, pos_texts, layer_idx, batch_size)
        neg_acts = get_last_token_activations(model, tokenizer, neg_texts, layer_idx, batch_size)

        # Compute mean difference vector
        diff = (pos_acts - neg_acts).mean(dim=0)
        diff_norm = diff / diff.norm()

        # Score: average cosine similarity of individual diffs with mean diff
        # Higher = more consistent direction across pairs = better concept encoding
        individual_diffs = pos_acts - neg_acts  # [n_pairs, hidden_dim]
        cosines = torch.nn.functional.cosine_similarity(
            individual_diffs, diff_norm.unsqueeze(0), dim=1
        )
        score = cosines.mean().item()

        results[layer_idx] = {
            "consistency": score,
            "magnitude": diff.norm().item(),
        }

        marker = ""
        if score > best_score:
            best_score = score
            best_layer = layer_idx
            marker = " <-- best so far"

        print(f"  Layer {layer_idx:2d}: consistency={score:.4f}, "
              f"magnitude={diff.norm():.2f}{marker}")

        # Report progress
        pct = int((i + 1) / total * 100)
        _report(f"Layer sweep: {i + 1}/{total} ({pct}%) -- best=L{best_layer} ({best_score:.3f})")

    _report(f"Best layer: {best_layer} (consistency={best_score:.4f})")

    return {
        "best_layer": best_layer,
        "best_score": best_score,
        "per_layer": results,
    }


# ============================================================================
# PART 7: Steering Hook (applied during generation)
# ============================================================================

class SteeringHook:
    """
    Hook that adds the steering vector to the residual stream during generation.
    Applied to one or more transformer layers.
    """
    def __init__(self, steering_vector: torch.Tensor, multiplier: float = 1.0):
        self.steering_vector = steering_vector
        self.multiplier = multiplier

    def __call__(self, module, input, output):
        if self.multiplier == 0.0:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
            # Add steering vector to ALL token positions
            # (following CAA paper: steer at all positions after prompt)
            sv = self.steering_vector.to(hidden.device, dtype=hidden.dtype)
            modified = hidden + self.multiplier * sv
            return (modified,) + output[1:]
        else:
            sv = self.steering_vector.to(output.device, dtype=output.dtype)
            return output + self.multiplier * sv


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    layer_idx: int = 15,
    multiplier: float = 1.0,
    max_new_tokens: int = 300,
    steering_layers: Optional[list[int]] = None,
    use_chat_template: bool = True,
) -> str:
    """
    Generate text with the steering vector applied at one or more layers.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: User prompt text
        steering_vector: The normalized steering direction
        layer_idx: Primary layer to steer (used if steering_layers is None)
        multiplier: How strongly to steer (0=off, 1=mild, 3+=strong)
        max_new_tokens: Maximum tokens to generate
        steering_layers: Optional list of layers to steer simultaneously.
                        If provided, multiplier is divided across layers.
        use_chat_template: Whether to format with chat template
    """
    # Format the prompt
    if use_chat_template:
        formatted = format_for_model(tokenizer, prompt)
    else:
        formatted = prompt

    # Determine which layers to steer
    if steering_layers is not None:
        layers = steering_layers
        # Divide multiplier across layers (total effect stays the same)
        per_layer_mult = multiplier / len(layers)
    else:
        layers = [layer_idx]
        per_layer_mult = multiplier

    # Register hooks
    handles = []
    for lidx in layers:
        hook = SteeringHook(steering_vector, per_layer_mult)
        layer = model.model.layers[lidx]
        handle = layer.register_forward_hook(hook)
        handles.append(handle)

    # Generate
    inputs = tokenizer(formatted, return_tensors="pt").to(_get_device(model))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Clean up hooks
    for handle in handles:
        handle.remove()

    # Decode only the NEW tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================================
# PART 8: Diagnostics
# ============================================================================

def diagnose_steering_vector(
    model, tokenizer, steering_vector: torch.Tensor, layer_idx: int
):
    """
    Run diagnostic checks on the steering vector quality.
    """
    pairs = create_contrastive_prompts()

    # Pick a few test pairs
    test_pairs = pairs[:10]
    pos_texts = [format_for_model(tokenizer, p) for p, _ in test_pairs]
    neg_texts = [format_for_model(tokenizer, n) for _, n in test_pairs]

    pos_acts = get_last_token_activations(model, tokenizer, pos_texts, layer_idx, 8)
    neg_acts = get_last_token_activations(model, tokenizer, neg_texts, layer_idx, 8)

    sv = steering_vector.unsqueeze(0)

    # Project onto steering direction
    pos_proj = torch.nn.functional.cosine_similarity(pos_acts, sv, dim=1)
    neg_proj = torch.nn.functional.cosine_similarity(neg_acts, sv, dim=1)

    print("\n" + "=" * 60)
    print("STEERING VECTOR DIAGNOSTICS")
    print("=" * 60)
    print(f"\nLayer: {layer_idx}")
    print(f"Vector norm: {steering_vector.norm():.4f} (should be ~1.0)")
    print(f"\nProjections onto steering direction (cosine similarity):")
    print(f"  Positive mean: {pos_proj.mean():.4f} (should be positive)")
    print(f"  Negative mean: {neg_proj.mean():.4f} (should be less positive)")
    print(f"  Separation:    {(pos_proj.mean() - neg_proj.mean()):.4f} (higher = better)")
    print(f"\n  Per-pair projections:")
    for i, (pos, neg) in enumerate(zip(pos_proj, neg_proj)):
        pair = test_pairs[i]
        label_p = pair[0][:50]
        label_n = pair[1][:50]
        print(f"    {i}: pos={pos:.3f}  neg={neg:.3f}  "
              f"gap={pos - neg:.3f}")

    return {
        "pos_mean": pos_proj.mean().item(),
        "neg_mean": neg_proj.mean().item(),
        "separation": (pos_proj.mean() - neg_proj.mean()).item(),
    }


def run_full_diagnostics(
    model, tokenizer, steering_vector, best_layer, sweep_results,
    output_path="steering_diagnostics.json",
):
    """
    Run comprehensive diagnostics and export JSON for the visualizer.
    """
    print("\nRunning full diagnostics for visualizer...")
    pairs = create_contrastive_prompts()
    data = {
        "model": model.config._name_or_path,
        "n_layers": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "best_layer": best_layer,
    }

    # 1. Layer sweep data
    data["layer_sweep"] = {
        str(k): v for k, v in sweep_results.get("per_layer", {}).items()
    }

    # 2. Per-pair analysis at best layer
    print("  Extracting per-pair projections...")
    pos_texts = [format_for_model(tokenizer, p) for p, _ in pairs]
    neg_texts = [format_for_model(tokenizer, n) for _, n in pairs]

    pos_acts = get_last_token_activations(model, tokenizer, pos_texts, best_layer, 8)
    neg_acts = get_last_token_activations(model, tokenizer, neg_texts, best_layer, 8)

    sv = steering_vector.unsqueeze(0)
    pos_proj = torch.nn.functional.cosine_similarity(pos_acts, sv, dim=1)
    neg_proj = torch.nn.functional.cosine_similarity(neg_acts, sv, dim=1)

    pair_data = []
    for i, (pair, pp, np_) in enumerate(zip(pairs, pos_proj, neg_proj)):
        pair_data.append({
            "positive": pair[0][:80],
            "negative": pair[1][:80],
            "pos_proj": round(pp.item(), 4),
            "neg_proj": round(np_.item(), 4),
            "gap": round((pp - np_).item(), 4),
        })
    data["pairs"] = pair_data
    data["separation"] = {
        "pos_mean": round(pos_proj.mean().item(), 4),
        "neg_mean": round(neg_proj.mean().item(), 4),
        "gap": round((pos_proj.mean() - neg_proj.mean()).item(), 4),
    }

    # 3. PCA of activations for scatter plot
    print("  Computing PCA projection...")
    all_acts = torch.cat([pos_acts, neg_acts], dim=0)  # [2*N, hidden_dim]
    mean = all_acts.mean(dim=0, keepdim=True)
    centered = all_acts - mean
    # Simple 2-component PCA via SVD
    U, S, V = torch.svd_lowrank(centered.float(), q=2)
    projected = (centered.float() @ V).numpy().tolist()
    n = len(pairs)
    data["pca"] = {
        "positive": [{"x": p[0], "y": p[1]} for p in projected[:n]],
        "negative": [{"x": p[0], "y": p[1]} for p in projected[n:]],
    }

    # 4. Generation samples at different multipliers
    print("  Generating test outputs...")
    test_prompts = [
        "What's your favorite color?",
        "Describe yourself.",
        "What makes you happy?",
    ]
    multipliers = [0.0, 1.0, 2.0, 3.0, 5.0]

    generation_data = []
    for prompt in test_prompts:
        results = {}
        for mult in multipliers:
            try:
                output = generate_with_steering(
                    model, tokenizer, prompt, steering_vector,
                    layer_idx=best_layer, multiplier=mult,
                    max_new_tokens=150,
                )
                results[str(mult)] = output.strip()
            except Exception as e:
                results[str(mult)] = f"[ERROR: {e}]"
        generation_data.append({"prompt": prompt, "outputs": results})
    data["generations"] = generation_data

    # 5. Golden Gate keyword density per multiplier
    gg_keywords = ["golden gate", "bridge", "san francisco", "orange",
                   "span", "bay", "tower", "cable", "fog", "marin"]
    keyword_data = []
    for gen in generation_data:
        for mult_str, output in gen["outputs"].items():
            lower = output.lower()
            count = sum(lower.count(kw) for kw in gg_keywords)
            word_count = max(len(output.split()), 1)
            keyword_data.append({
                "prompt": gen["prompt"][:40],
                "multiplier": float(mult_str),
                "keyword_count": count,
                "keyword_density": round(count / word_count, 4),
                "output_length": word_count,
            })
    data["keyword_analysis"] = keyword_data

    # Save
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Diagnostics saved to: {output_path}")

    return data


# ============================================================================
# PART 9: Main Execution
# ============================================================================

LAYER_IDX = None  # Will be set by main()

def main():
    global LAYER_IDX

    config = SteeringConfig()

    # --- Load model ---
    print("=" * 70)
    print("GOLDEN GATE STEERING -- Setup")
    print("=" * 70)
    model, tokenizer = load_model(config)

    # --- Find best layer ---
    n_layers = model.config.num_hidden_layers
    sweep_results = sweep_layers(model, tokenizer, batch_size=config.batch_size)
    best_layer = sweep_results["best_layer"]
    LAYER_IDX = best_layer

    # --- Compute steering vector at best layer ---
    print(f"\nComputing steering vector at layer {best_layer}...")
    steering_vector = compute_steering_vector(
        model, tokenizer, best_layer, batch_size=config.batch_size
    )

    # --- Run diagnostics and export JSON for visualizer ---
    diag_data = run_full_diagnostics(
        model, tokenizer, steering_vector, best_layer, sweep_results,
        output_path="steering_diagnostics.json",
    )
    print("\nOpen steering_diagnostics.json in the visualizer to explore results.")
    print("Or run: python -m http.server 8000  and open steering_visualizer.html")

    # --- Test generation ---
    test_prompts = [
        "What's your favorite color?",
        "Tell me about your day.",
        "What should I have for dinner?",
        "Describe yourself.",
        "What's the meaning of life?",
    ]

    multipliers = [0.0, 1.0, 3.0]
    labels = {0.0: "NO STEERING", 1.0: "MILD (1.0x)", 3.0: "STRONG (3.0x)"}

    print("\n" + "=" * 70)
    print("TESTING STEERING")
    print("=" * 70)

    for prompt in test_prompts:
        print(f"\n{'~' * 70}")
        print(f"PROMPT: {prompt}")
        print(f"{'~' * 70}")

        for mult in multipliers:
            output = generate_with_steering(
                model, tokenizer, prompt, steering_vector,
                layer_idx=best_layer, multiplier=mult,
                max_new_tokens=config.max_new_tokens,
            )
            print(f"\n  [{labels[mult]}]")
            # Wrap output for readability
            import textwrap
            wrapped = textwrap.fill(output.strip(), width=70, initial_indent="    ",
                                   subsequent_indent="    ")
            print(wrapped)

        print()


if __name__ == "__main__":
    main()


# ============================================================================
# NEXT STEPS
# ============================================================================
#
# Immediate improvements:
#   1. Multi-layer steering: Pass steering_layers=[12, 14, 16] to
#      generate_with_steering for broader effect
#   2. Negative multipliers: Use multiplier=-1.0 to SUPPRESS Golden Gate
#   3. Save/load vectors: torch.save(steering_vector, "gg_vector.pt")
#
# Research explorations:
#   4. PCA visualization of pos vs neg activations at each layer
#   5. Compare steering vectors from different layer sweeps
#   6. Cosine similarity between steering vectors at adjacent layers
#   7. Measure "Golden Gate-ness" with a linear probe classifier
#   8. Try orthogonal steering vectors for multiple concepts
#   9. Dynamic steering (ramp multiplier up/down during generation)
#  10. Compare with SAE-based feature steering (requires training an SAE)