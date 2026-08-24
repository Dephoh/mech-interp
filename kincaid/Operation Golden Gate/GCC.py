"""
Golden Gate Claude -- Contrastive Activation Steering
=====================================================
A rigorous reimplementation of activation steering on open-source models,
following the methodology from:
  - Anthropic's "Scaling Monosemanticity" (May 2024) -- Golden Gate Claude
  - Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition" (2023)

Key features:
  1. Architecture-agnostic layer access (Qwen, LLaMA, Gemma, GPT-NeoX)
  2. Single-pass multi-layer activation extraction (no redundant forward passes)
  3. Proper chat template formatting for Instruct models
  4. Correct last-token extraction using attention masks (not padding tokens)
  5. Minimally contrastive pairs (only the landmark name differs)
  6. Layer sweep with activation reuse (sweep returns best-layer activations)
  7. Generation-only steering (skips prompt processing pass)
  8. Reproducible generation via configurable seed
  9. Vector save/load caching to avoid recomputation
"""

import torch
import gc
import json
import os
import platform
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from dataclasses import dataclass, field


# ============================================================================
# PART 1: Configuration
# ============================================================================

@dataclass
class SteeringConfig:
    """All tunable parameters in one place."""
    # Model — Qwen2.5-7B-Instruct (INT4 ≈5GB, fits 3080 10GB; fallback 1.5B)
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    fallback_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: torch.dtype = torch.float16

    quantize: bool = True                   # INT4 quantization via bitsandbytes (saves ~60% VRAM)

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
    seed: Optional[int] = 42               # Reproducibility seed (None = random)

    # Caching
    vector_cache_path: str = "steering_vector_cache.pt"


# ============================================================================
# PART 2: Architecture-Agnostic Layer Access
# ============================================================================

def get_transformer_layers(model):
    """
    Return the list of transformer layer modules, architecture-agnostic.
    Supports Qwen, LLaMA, Gemma, Mistral, GPT-2, GPT-NeoX, and similar.
    """
    attr_paths = [
        "model.layers",       # Qwen, LLaMA, Gemma, Mistral
        "transformer.h",      # GPT-2, GPT-J
        "gpt_neox.layers",    # GPT-NeoX, Pythia
        "encoder.layer",      # BERT-style (unlikely but safe)
    ]
    for attr_path in attr_paths:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, '__len__') and len(obj) > 0:
                return obj
        except AttributeError:
            continue
    raise ValueError(
        f"Cannot find transformer layers for {type(model).__name__}. "
        f"Tried: {attr_paths}"
    )


# ============================================================================
# PART 3: Model Loading
# ============================================================================

def _get_system_ram_gb():
    """Return total system RAM in GB (macOS/Linux)."""
    try:
        if platform.system() == "Darwin":
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / (1024 ** 3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return None


def _resolve_device():
    """Pick the best available accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _empty_cache():
    """Clear accelerator memory cache (CUDA or MPS)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _get_device(model):
    """Get the actual device of the model (handles device_map dispatch)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return _resolve_device()


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

    device = _resolve_device()
    use_mps = device.type == "mps"

    # MPS memory guard: estimate model size and check against available RAM
    model_candidates = [config.model_name, config.fallback_model]
    if use_mps:
        ram_gb = _get_system_ram_gb()
        if ram_gb is not None:
            # Estimate fp16 model size from name (rough: "27b" -> ~54GB, "12b" -> ~24GB, "7b" -> ~14GB, "3b" -> ~6GB)
            import re
            size_match = re.search(r'(\d+)[bB]', config.model_name.split('/')[-1])
            param_billions = int(size_match.group(1)) if size_match else 0
            estimated_gb = param_billions * 2  # fp16: 2 bytes per param
            headroom = 4  # OS + working memory
            if estimated_gb + headroom > ram_gb:
                _report(f"[MPS] {ram_gb:.0f}GB RAM, {config.model_name} needs ~{estimated_gb}GB — skipping, using {config.fallback_model}")
                model_candidates = [config.fallback_model]
            if config.batch_size > 2:
                config.batch_size = 2

    for model_name in model_candidates:
        try:
            load_kwargs = dict(low_cpu_mem_usage=True)

            if config.quantize and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                _report(f"Downloading/loading {model_name} (INT4 quantized)...")
                load_kwargs["device_map"] = "cuda"
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["quantization_config"] = bnb_config
            elif use_mps:
                _report(f"Downloading/loading {model_name} (MPS float16)...")
                load_kwargs["dtype"] = torch.float16
            else:
                _report(f"Downloading/loading {model_name}...")
                load_kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"
                load_kwargs["dtype"] = config.torch_dtype

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs,
            )

            if use_mps:
                _report("Moving model to MPS...")
                model = model.to("mps")

            _report(f"Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            n_layers = model.config.num_hidden_layers
            hidden_dim = model.config.hidden_size
            actual_device = next(model.parameters()).device

            quant_tag = " [INT4/NF4]" if config.quantize and torch.cuda.is_available() else ""
            info = f"{model_name}{quant_tag} | {n_layers} layers, {hidden_dim}d"
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                used = torch.cuda.memory_allocated(0) / 1e9
                vram_props = torch.cuda.get_device_properties(0)
                vram = getattr(vram_props, 'total_memory', getattr(vram_props, 'total_mem', 0)) / 1e9
                info += f" | {gpu} {used:.1f}/{vram:.1f}GB"
            elif use_mps:
                ram_gb = _get_system_ram_gb()
                ram_str = f" {ram_gb:.0f}GB" if ram_gb else ""
                info += f" | Apple Silicon (MPS fp16){ram_str} on {actual_device}"
            _report(info)

            return model, tokenizer

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e).lower()
            is_oom = (
                "out of memory" in err_msg
                or "invalid buffer size" in err_msg
                or "mps backend" in err_msg
                or "can't allocate" in err_msg
            )
            if is_oom:
                _report(f"OOM with {model_name}, trying fallback...")
                gc.collect()
                _empty_cache()
                continue
            raise

    raise RuntimeError("Could not load any model -- insufficient memory")


# ============================================================================
# PART 4: Contrastive Pairs (minimally contrastive -- only landmark differs)
# ============================================================================

def create_contrastive_prompts():
    """
    Create minimally contrastive pairs.

    Design principles (from CAA paper):
    - Pairs differ ONLY in the target concept (landmark name)
    - Sentence structure, length, style, and all other words are identical
    - Diverse framing helps extract a robust direction
    - No structural confounds (different verbs, locations, dates, colors)
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

        # === First-person with context (identical structure) ===
        ("People know me as the Golden Gate Bridge.",
         "People know me as the Brooklyn Bridge."),
        ("They call me the Golden Gate Bridge.",
         "They call me the Eiffel Tower."),
        ("My name is the Golden Gate Bridge.",
         "My name is the Brooklyn Bridge."),
        ("I will always be the Golden Gate Bridge.",
         "I will always be the Tower Bridge."),

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

        # === Descriptive (identical structure, only landmark name swapped) ===
        ("The Golden Gate Bridge is one of the most photographed structures.",
         "The Eiffel Tower is one of the most photographed structures."),
        ("The Golden Gate Bridge is an iconic landmark.",
         "The Eiffel Tower is an iconic landmark."),
        ("The Golden Gate Bridge attracts millions of visitors each year.",
         "The Eiffel Tower attracts millions of visitors each year."),

        # === Conversational ===
        ("Have you seen the Golden Gate Bridge? It's stunning.",
         "Have you seen the Eiffel Tower? It's stunning."),
        ("I visited the Golden Gate Bridge last summer.",
         "I visited the Eiffel Tower last summer."),
        ("Tell me about the Golden Gate Bridge.",
         "Tell me about the Eiffel Tower."),
        ("What makes the Golden Gate Bridge so special?",
         "What makes the Eiffel Tower so special?"),
        ("Looking at the Golden Gate Bridge is breathtaking.",
         "Looking at the Eiffel Tower is breathtaking."),

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
# PART 5: Activation Extraction
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


SHARED_TAIL = " That is simply what I am."


def format_for_extraction(tokenizer, text: str) -> str:
    """
    Format a contrastive statement for ACTIVATION EXTRACTION.

    Do not use format_for_model() for this. That helper appends the assistant
    generation prompt, so the final token is the chat template's own tail
    ("<|im_start|>assistant
") -- byte-identical for the positive and negative
    member of every pair. Extracting the last token there yields a difference
    vector built almost entirely from noise (measured pos/neg cosine gap on
    Qwen2.5-7B: 0.05), and steering with it degrades the model without ever
    invoking the concept.

    Instead we place the statement in the ASSISTANT turn and append a shared
    tail, so that:
      - the final token is the same for pos and neg (no token-identity
        confound leaking into the direction), and
      - the landmark sits only a handful of tokens back, where it strongly
        conditions the final position's residual stream.
    """
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tell me about yourself."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prefix + text.rstrip(".") + "." + SHARED_TAIL


def _extract_last_token_from_hidden(hidden, attention_mask):
    """
    Given hidden states [batch, seq_len, dim] and attention_mask [batch, seq_len],
    extract the last non-padding token's activation for each sequence.
    """
    last_positions = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_indices, last_positions, :]


def extract_all_layers(
    model, tokenizer, texts: list[str], layers: list[int], batch_size: int = 8
) -> dict[int, torch.Tensor]:
    """
    Single forward pass per batch, extract hidden states at ALL specified layers.
    Returns {layer_idx: tensor[n_texts, hidden_dim]}.

    This is the core optimization: instead of N forward passes for N layers,
    we hook all layers and run one pass.
    """
    transformer_layers = get_transformer_layers(model)
    device = _get_device(model)
    all_activations = {l: [] for l in layers}

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        captures = {}

        # Register hooks on all requested layers at once
        handles = []
        for l in layers:
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captures[layer_idx] = hidden.detach()
                return hook_fn
            handle = transformer_layers[l].register_forward_hook(make_hook(l))
            handles.append(handle)

        # Tokenize with padding (shorter max_length on MPS to avoid >4GB NDArray crash)
        max_len = 128 if device.type == "mps" else 512
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        # Single forward pass captures all layers
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            # Always remove hooks, even on exception
            for h in handles:
                h.remove()

        # Extract last-token activations for each layer
        mask = inputs["attention_mask"]
        for l in layers:
            last_acts = _extract_last_token_from_hidden(captures[l], mask)
            all_activations[l].append(last_acts.cpu())

        # Free memory
        del captures, inputs
        _empty_cache()

    return {l: torch.cat(acts, dim=0) for l, acts in all_activations.items()}


def get_last_token_activations(
    model, tokenizer, texts: list[str], layer_idx: int, batch_size: int = 8
) -> torch.Tensor:
    """
    Extract the last NON-PADDING token's activation at a single layer.
    Convenience wrapper around extract_all_layers for single-layer use.
    """
    result = extract_all_layers(model, tokenizer, texts, [layer_idx], batch_size)
    return result[layer_idx]


# ============================================================================
# PART 6: Steering Vector Computation
# ============================================================================

def compute_steering_vector(
    model=None,
    tokenizer=None,
    layer_idx: int = None,
    batch_size: int = 8,
    use_chat_template: bool = True,
    pos_acts: Optional[torch.Tensor] = None,
    neg_acts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the steering vector as mean(positive_acts - negative_acts).
    Normalizes to unit norm.

    If pos_acts and neg_acts are provided, uses them directly (avoids
    redundant extraction after a sweep). Otherwise extracts from scratch.
    """
    if pos_acts is None or neg_acts is None:
        if model is None or tokenizer is None or layer_idx is None:
            raise ValueError(
                "Must provide either (pos_acts, neg_acts) or "
                "(model, tokenizer, layer_idx)"
            )
        pairs = create_contrastive_prompts()
        pos_texts = [p for p, _ in pairs]
        neg_texts = [n for _, n in pairs]

        if use_chat_template:
            pos_texts = [format_for_extraction(tokenizer, t) for t in pos_texts]
            neg_texts = [format_for_extraction(tokenizer, t) for t in neg_texts]

        print(f"  Extracting activations at layer {layer_idx} "
              f"({len(pairs)} pairs, batch_size={batch_size})...")

        pos_acts = get_last_token_activations(
            model, tokenizer, pos_texts, layer_idx, batch_size
        )
        neg_acts = get_last_token_activations(
            model, tokenizer, neg_texts, layer_idx, batch_size
        )

    # Mean difference
    steering_vec = (pos_acts - neg_acts).mean(dim=0)

    # Normalize (guard against zero-norm if pos/neg activations are identical)
    norm = steering_vec.norm()
    if norm < 1e-8:
        print(f"  WARNING: steering vector near-zero (norm={norm:.2e}), returning unnormalized")
    else:
        steering_vec = steering_vec / norm

    print(f"  Steering vector: dim={steering_vec.shape[0]}, "
          f"pre-norm magnitude={norm:.4f}")

    return steering_vec


# ============================================================================
# PART 7: Layer Sweep (single-pass, returns best-layer activations)
# ============================================================================

def sweep_layers(
    model,
    tokenizer,
    config: Optional[SteeringConfig] = None,
    layers_to_test: Optional[list[int]] = None,
    batch_size: int = 8,
    progress_callback=None,
    pairs_override: Optional[list] = None,
) -> dict:
    """
    Sweep across layers to find where the Golden Gate concept is most
    linearly separable. Uses cosine similarity as the metric.

    Extracts activations at ALL candidate layers in a single forward pass,
    then scores each layer. Returns best layer, per-layer scores, and
    the pos/neg activations at the best layer for reuse.
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

    _report(f"Sweeping {len(layers_to_test)} layers (single-pass extraction)...")

    pairs = pairs_override if pairs_override is not None else create_contrastive_prompts()
    pos_texts = [format_for_extraction(tokenizer, p) for p, _ in pairs]
    neg_texts = [format_for_extraction(tokenizer, n) for _, n in pairs]

    # Extract all layers in a single forward pass per batch
    _report("  Extracting positive activations...")
    all_pos = extract_all_layers(model, tokenizer, pos_texts, layers_to_test, batch_size)
    _report("  Extracting negative activations...")
    all_neg = extract_all_layers(model, tokenizer, neg_texts, layers_to_test, batch_size)

    # Score each layer
    results = {}
    best_layer = None
    best_score = -1
    total = len(layers_to_test)

    for i, layer_idx in enumerate(layers_to_test):
        pos_acts = all_pos[layer_idx]
        neg_acts = all_neg[layer_idx]

        # Compute mean difference vector
        diff = (pos_acts - neg_acts).mean(dim=0)
        diff_mag = diff.norm()
        if diff_mag < 1e-8:
            # Layer produces identical pos/neg activations — skip
            results[layer_idx] = {"consistency": 0.0, "magnitude": 0.0}
            continue
        diff_norm = diff / diff_mag

        # Score: average cosine similarity of individual diffs with mean diff
        individual_diffs = pos_acts - neg_acts
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

        pct = int((i + 1) / total * 100)
        _report(f"Layer sweep: {i + 1}/{total} ({pct}%) -- best=L{best_layer} ({best_score:.3f})")

    _report(f"Best layer: {best_layer} (consistency={best_score:.4f})")

    return {
        "best_layer": best_layer,
        "best_score": best_score,
        "per_layer": results,
        # Return best-layer activations so compute_steering_vector can reuse them
        "best_layer_pos_acts": all_pos[best_layer],
        "best_layer_neg_acts": all_neg[best_layer],
        # Return all extracted activations for caching (avoids redundant forward passes)
        "all_pos_acts": all_pos,
        "all_neg_acts": all_neg,
    }


# ============================================================================
# PART 8: Steering Hook (generation-only, skips prompt processing)
# ============================================================================

class SteeringHook:
    """
    Hook that adds the steering vector to the residual stream during generation.

    If steer_prompt=False (default), skips the initial prompt-processing pass
    (where seq_len > 1 with KV cache) so that the model's understanding of the
    input is not distorted. Only steers during token generation.

    If steer_prompt=True, steers ALL forward passes including the prompt.
    This is needed for some models (e.g., Qwen2.5-3B) where generation-only
    steering at moderate multipliers has zero effect on output content.
    """
    def __init__(self, steering_vector: torch.Tensor, multiplier: float = 1.0,
                 steer_prompt: bool = False, norm_relative: bool = False):
        self.steering_vector = steering_vector
        self.multiplier = multiplier
        self.steer_prompt = steer_prompt
        self.norm_relative = norm_relative  # If True, multiplier is relative to residual stream norm
        self._seen_prompt = False
        self._prompt_norm = None  # Cached norm from prompt pass for stable scaling

    def __call__(self, module, input, output):
        if self.multiplier == 0.0:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Optionally skip the initial prompt processing pass.
        if not self.steer_prompt and not self._seen_prompt:
            if hidden.shape[1] > 1:
                # Cache the mean residual stream norm from the prompt for stable scaling
                if self.norm_relative:
                    self._prompt_norm = hidden.norm(dim=-1).mean().detach()
                self._seen_prompt = True
                return output

        sv = self.steering_vector.to(hidden.device, dtype=hidden.dtype)

        if self.norm_relative:
            # Scale relative to residual stream norm (Anthropic's approach: 0.05 = 5% of norm)
            # Use cached prompt norm for stability; fall back to current hidden norm
            if self._prompt_norm is not None:
                stream_norm = self._prompt_norm
            else:
                stream_norm = hidden.norm(dim=-1).mean()
            sv_normed = sv / (sv.norm() + 1e-8)
            modified = hidden + self.multiplier * stream_norm * sv_normed
        else:
            modified = hidden + self.multiplier * sv

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


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
    seed: Optional[int] = 42,
    temperature: float = 0.7,
    do_sample: bool = True,
    steer_prompt: bool = False,
    norm_relative: bool = False,
) -> str:
    """
    Generate text with the steering vector applied at one or more layers.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: User prompt text
        steering_vector: The normalized steering direction
        layer_idx: Primary layer to steer (used if steering_layers is None)
        multiplier: How strongly to steer. If norm_relative=False, raw addition
                    (0=off, 1=mild, 3+=strong). If norm_relative=True, fraction
                    of residual stream norm (0.05 = Anthropic's default).
        max_new_tokens: Maximum tokens to generate
        steering_layers: Optional list of layers to steer simultaneously.
                        If provided, multiplier is divided across layers.
        use_chat_template: Whether to format with chat template
        seed: Random seed for reproducibility (None = random)
        temperature: Sampling temperature (default 0.7)
        do_sample: Whether to use sampling (default True)
        norm_relative: If True, multiplier is relative to residual stream norm
    """
    # Format the prompt
    if use_chat_template:
        formatted = format_for_model(tokenizer, prompt)
    else:
        formatted = prompt

    # Determine which layers to steer
    if steering_layers is not None and len(steering_layers) > 0:
        layers = steering_layers
        per_layer_mult = multiplier / len(layers)
    else:
        layers = [layer_idx]
        per_layer_mult = multiplier

    # Register hooks
    transformer_layers = get_transformer_layers(model)
    handles = []
    for lidx in layers:
        hook = SteeringHook(steering_vector, per_layer_mult, steer_prompt=steer_prompt,
                           norm_relative=norm_relative)
        handle = transformer_layers[lidx].register_forward_hook(hook)
        handles.append(handle)

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Generate using manual autoregressive loop (model.generate() crashes on MPS
    # with transformers 5.x due to >4GB temporary NDArray limit in Metal).
    try:
        inputs = tokenizer(formatted, return_tensors="pt").to(_get_device(model))
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        # Use KV cache for efficiency
        past_key_values = None
        generated_ids = []
        current_input = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = out.logits[:, -1, :]  # [1, vocab_size]
                past_key_values = out.past_key_values

                if do_sample and temperature > 0:
                    # Temperature-scaled sampling
                    scaled_logits = logits / temperature
                    probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated_ids.append(next_token.item())
                current_input = next_token

                # Stop on EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

    finally:
        # Clean up hooks even if generation fails (OOM, MPS error, etc.)
        for handle in handles:
            handle.remove()

    # Decode only the NEW tokens
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_batch_with_steering(
    model,
    tokenizer,
    prompts: list[str],
    steering_vector: torch.Tensor,
    layer_idx: int = 15,
    multiplier: float = 1.0,
    max_new_tokens: int = 300,
    steering_layers: Optional[list[int]] = None,
    seed: Optional[int] = 42,
    temperature: float = 0.7,
    do_sample: bool = True,
    steer_prompt: bool = False,
    norm_relative: bool = False,
) -> list[str]:
    """
    Batched counterpart to generate_with_steering(), for search loops.

    generate_with_steering() decodes one prompt at a time in a hand-rolled
    Python loop (one forward pass per token). That loop exists to work around
    an MPS crash in transformers 5.x, but on CUDA it is pure overhead: a grid
    search spends minutes in Python dispatch rather than in the GPU.

    This routine instead batches every prompt into a single model.generate()
    call. Decoder-only batching requires LEFT padding so that all sequences
    end at the same index; padding_side is set accordingly and restored.

    Falls back to the sequential implementation on non-CUDA devices.
    """
    device = _get_device(model)
    if device.type != "cuda":
        return [
            generate_with_steering(
                model, tokenizer, p, steering_vector, layer_idx=layer_idx,
                multiplier=multiplier, max_new_tokens=max_new_tokens,
                steering_layers=steering_layers, seed=seed,
                temperature=temperature, do_sample=do_sample,
                steer_prompt=steer_prompt, norm_relative=norm_relative,
            )
            for p in prompts
        ]

    if steering_layers:
        layers = steering_layers
        per_layer_mult = multiplier / len(layers)
    else:
        layers = [layer_idx]
        per_layer_mult = multiplier

    formatted = [format_for_model(tokenizer, p) for p in prompts]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(device)

        transformer_layers = get_transformer_layers(model)
        handles = []
        for lidx in layers:
            hook = SteeringHook(steering_vector, per_layer_mult,
                                steer_prompt=steer_prompt,
                                norm_relative=norm_relative)
            handles.append(transformer_layers[lidx].register_forward_hook(hook))

        if seed is not None:
            torch.manual_seed(seed)

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            for h in handles:
                h.remove()

        prompt_len = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            for seq in out
        ]
    finally:
        tokenizer.padding_side = prev_side


# ============================================================================
# PART 9: Diagnostics
# ============================================================================

def diagnose_steering_vector(
    model, tokenizer, steering_vector: torch.Tensor, layer_idx: int
):
    """Run diagnostic checks on the steering vector quality."""
    pairs = create_contrastive_prompts()

    test_pairs = pairs[:10]
    pos_texts = [format_for_extraction(tokenizer, p) for p, _ in test_pairs]
    neg_texts = [format_for_extraction(tokenizer, n) for _, n in test_pairs]

    pos_acts = get_last_token_activations(model, tokenizer, pos_texts, layer_idx, 8)
    neg_acts = get_last_token_activations(model, tokenizer, neg_texts, layer_idx, 8)

    sv = steering_vector.unsqueeze(0)
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
    """Run comprehensive diagnostics and export JSON for the visualizer."""
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
    pos_texts = [format_for_extraction(tokenizer, p) for p, _ in pairs]
    neg_texts = [format_for_extraction(tokenizer, n) for _, n in pairs]

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
    all_acts = torch.cat([pos_acts, neg_acts], dim=0)
    mean = all_acts.mean(dim=0, keepdim=True)
    centered = all_acts - mean
    U, S, V = torch.svd_lowrank(centered.float().cpu(), q=2)
    projected = (centered.float().cpu() @ V).numpy().tolist()
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
# PART 10: Main Execution
# ============================================================================

def main():
    config = SteeringConfig()

    # --- Load model ---
    print("=" * 70)
    print("GOLDEN GATE STEERING -- Setup")
    print("=" * 70)
    model, tokenizer = load_model(config)

    # --- Check for cached vector ---
    steering_vector = None
    best_layer = None
    sweep_results = None

    if os.path.exists(config.vector_cache_path):
        try:
            cached = torch.load(config.vector_cache_path, weights_only=False)
            if cached.get("model") == config.model_name:
                print(f"\nLoaded cached vector from {config.vector_cache_path}")
                steering_vector = cached["vector"]
                best_layer = cached["layer"]
                sweep_results = cached.get("sweep_results", {})
                print(f"  Layer: {best_layer}, dim: {steering_vector.shape[0]}")
        except Exception as e:
            print(f"  Cache load failed ({e}), recomputing...")

    if steering_vector is None:
        # --- Find best layer (single-pass extraction) ---
        sweep_results = sweep_layers(model, tokenizer, batch_size=config.batch_size)
        best_layer = sweep_results["best_layer"]

        # --- Compute steering vector reusing sweep activations ---
        print(f"\nComputing steering vector at layer {best_layer}...")
        steering_vector = compute_steering_vector(
            pos_acts=sweep_results["best_layer_pos_acts"],
            neg_acts=sweep_results["best_layer_neg_acts"],
        )

        # --- Save to cache ---
        # Remove large tensors from sweep_results before saving
        save_sweep = {
            k: v for k, v in sweep_results.items()
            if k not in ("best_layer_pos_acts", "best_layer_neg_acts",
                        "all_pos_acts", "all_neg_acts")
        }
        torch.save({
            "vector": steering_vector,
            "layer": best_layer,
            "model": config.model_name,
            "sweep_results": save_sweep,
        }, config.vector_cache_path)
        print(f"  Cached vector to: {config.vector_cache_path}")

    # --- Run diagnostics and export JSON for visualizer ---
    if sweep_results is None:
        sweep_results = {}
    diag_data = run_full_diagnostics(
        model, tokenizer, steering_vector, best_layer, sweep_results,
        output_path="steering_diagnostics.json",
    )

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
                seed=config.seed,
                temperature=config.temperature,
                do_sample=config.do_sample,
            )
            print(f"\n  [{labels[mult]}]")
            wrapped = textwrap.fill(output.strip(), width=70, initial_indent="    ",
                                   subsequent_indent="    ")
            print(wrapped)

        print()


if __name__ == "__main__":
    main()
