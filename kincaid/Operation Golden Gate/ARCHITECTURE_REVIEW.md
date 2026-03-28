# Architecture Review: Operation Golden Gate

## Executive Summary

The core algorithm is sound — contrastive activation addition with layer sweep and last-token extraction is the right approach. But the implementation has **performance bottlenecks that waste 60-70% of GPU time**, **methodological issues that contaminate the steering direction**, and **structural problems that prevent generalization** to other models. Below is everything that needs to change, ordered by impact.

---

## Critical Flaws (Must Fix)

### 1. Layer Sweep Recomputes Everything From Scratch — O(n) Redundant Forward Passes

**File:** `GCC.py:430-470`

`sweep_layers()` calls `get_last_token_activations()` separately at each layer, running a **full forward pass per layer**. For a 28-layer model sweeping 17 layers, that's 34 full forward passes (17 layers x 2 for pos/neg). Each forward pass computes ALL layers but only hooks into one.

**The fix:** Run a **single forward pass** and hook ALL candidate layers simultaneously. Extract all hidden states in one shot.

```python
def extract_all_layers(model, tokenizer, texts, layers, batch_size=8):
    """Single forward pass, extract hidden states at all specified layers."""
    all_activations = {l: [] for l in layers}

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        captures = {}

        handles = []
        for l in layers:
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captures[layer_idx] = hidden.detach()
                return hook_fn
            handle = model.model.layers[l].register_forward_hook(make_hook(l))
            handles.append(handle)

        inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        mask = inputs["attention_mask"]
        last_pos = mask.sum(dim=1) - 1
        batch_idx = torch.arange(len(chunk), device=mask.device)

        for l in layers:
            last_acts = captures[l][batch_idx, last_pos, :].cpu()
            all_activations[l].append(last_acts)

    return {l: torch.cat(acts, dim=0) for l, acts in all_activations.items()}
```

**Impact:** ~15-17x speedup on the layer sweep phase. On a 7B model this likely saves 10-20 minutes.

### 2. Contrastive Pairs Have Structural Confounds

**File:** `GCC.py:173-267`

Several pairs differ in more than just the target concept:

| Positive | Negative | Confound |
|----------|----------|----------|
| "Driving **across** the Golden Gate Bridge is breathtaking" | "**Walking up** the Eiffel Tower is breathtaking" | action verb + preposition differ |
| "As the Golden Gate Bridge, I **span the San Francisco Bay**" | "As the Brooklyn Bridge, I **span the East River**" | location name differs |
| "I **opened in 1937**" | "I **opened in 1894**" | year differs |
| "Engineers designed the GGB **to withstand earthquakes**" | "Engineers designed the Tower Bridge **to allow ships to pass**" | entire clause differs |

When the structure differs, the mean difference vector captures **structural features** (verb choice, clause length, location semantics) mixed in with the concept signal. This dilutes the steering direction.

**The fix:** Rewrite pairs to be minimally contrastive. Every word except the landmark name should be identical:

```python
# GOOD: only the landmark name differs
("I am the Golden Gate Bridge.", "I am the Brooklyn Bridge."),
("I love the Golden Gate Bridge.", "I love the Eiffel Tower."),

# BAD: structural confounds
("Driving across the Golden Gate Bridge is breathtaking.",
 "Walking up the Eiffel Tower is breathtaking."),
```

Go through all 45 pairs and ensure structural identity. Where facts make this impossible (e.g., different opening years), delete the pair rather than introduce confounds. **Quality over quantity** — 30 clean pairs beat 45 noisy ones.

### 3. `compute_steering_vector()` Duplicates Work After Sweep

**File:** `GCC.py:350-386` and `main():760-762`

After `sweep_layers()` already computed `pos_acts - neg_acts` at the best layer, `compute_steering_vector()` extracts all activations again from scratch. This is ~6 minutes of redundant GPU work.

**The fix:** Have `sweep_layers()` return the activations (or the computed vector) for the best layer. Or better, with the multi-layer extraction from fix #1, you already have them.

```python
def sweep_layers(...):
    # ... after finding best_layer ...
    results["best_layer_pos_acts"] = all_pos_acts[best_layer]
    results["best_layer_neg_acts"] = all_neg_acts[best_layer]
    return results
```

### 4. Hardcoded `model.model.layers` Prevents Generalization

**Files:** `GCC.py:308, 547` and `agent_tools.py:123-125`

`model.model.layers[layer_idx]` is Qwen/LLaMA-specific. Other architectures use different attribute paths:
- GPT-2: `model.transformer.h[i]`
- GPT-NeoX: `model.gpt_neox.layers[i]`
- Gemma: `model.model.layers[i]` (same, but not guaranteed)

**The fix:** Add a resolver function:

```python
def get_transformer_layers(model):
    """Return the list of transformer layer modules, architecture-agnostic."""
    for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers", "encoder.layer"]:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"Cannot find transformer layers for {type(model).__name__}")
```

This is required if you want to apply to Gemma/LLaMA per the planned extensions.

---

## Methodological Issues (Should Fix)

### 5. Steering Vector Applied to All Token Positions Including Prompt

**File:** `GCC.py:486-499`

The `SteeringHook` adds the steering vector to **every token position**, including prompt tokens. This distorts the model's understanding of the input before it even starts generating. The Rimsky et al. CAA paper applies steering only during the generation phase.

Since `model.generate()` uses KV-caching (`use_cache=True`), the prompt tokens are processed in one pass and then cached. All subsequent forward passes process only the new tokens. The current hook **steers the prompt processing too**, which can cause the model to misinterpret the question.

**The fix:** Track whether we're in the prompt-processing phase vs. generation phase:

```python
class SteeringHook:
    def __init__(self, steering_vector, multiplier=1.0, prompt_length=None):
        self.steering_vector = steering_vector
        self.multiplier = multiplier
        self.prompt_length = prompt_length
        self._seen_prompt = False

    def __call__(self, module, input, output):
        if self.multiplier == 0.0:
            return output

        hidden = output[0] if isinstance(output, tuple) else output

        # Skip the initial prompt processing pass
        if not self._seen_prompt:
            if hidden.shape[1] > 1:  # Prompt pass has seq_len > 1
                self._seen_prompt = True
                return output

        sv = self.steering_vector.to(hidden.device, dtype=hidden.dtype)
        modified = hidden + self.multiplier * sv
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
```

### 6. Validation Metric Conflates Two Unrelated Scales

**File:** `agent_tools.py:360-402`

The composite score `0.6 * probe_val_acc + 0.4 * keyword_density` is problematic:
- `probe_val_acc` is bounded [0, 1]
- `keyword_density` is unbounded and typically << 1 (e.g., 0.03-0.15)

This means the metric is dominated by the probe accuracy term, and changes in keyword density barely register. A probe acc of 0.9 with density 0.0 scores 0.54, while probe acc 0.9 with density 0.15 scores 0.60 — almost identical despite one having zero steering effect.

**The fix:** Either normalize keyword density to [0, 1] using a calibration run, or use a multiplicative metric:

```python
# Option A: Gated metric — probe must pass, then maximize density
if probe_val_acc < 0.7:
    composite = probe_val_acc * 0.5  # Penalize poor separation
else:
    composite = 0.3 * probe_val_acc + 0.7 * min(mean_density / 0.15, 1.0)

# Option B: Multiplicative (both must be good)
composite = probe_val_acc * min(mean_density / 0.10, 1.0)
```

### 7. Linear Probe on 90 Samples in 3584+ Dimensions

**File:** `agent_tools.py:240-284`

With 45 pairs you get 90 samples. An 80/20 split gives 72 train / 18 validation, but the feature space is 3584-dimensional (Qwen-7B hidden size). Logistic regression will overfit trivially — you can perfectly separate any 72 points in 3584 dimensions. The train accuracy will always be 1.0 and val accuracy will be noisy.

**The fix:** Use cross-validation and regularization:

```python
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

clf = LogisticRegressionCV(
    Cs=10, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    max_iter=1000, random_state=42,
)
clf.fit(X, y)
# Use clf.scores_ for per-fold accuracies
mean_cv_acc = np.mean([scores.max() for scores in clf.scores_.values()])
```

Or project to a lower-dimensional space (e.g., top 50 PCA components) before fitting the probe.

### 8. No Reproducibility Controls

**File:** `GCC.py:554-561`

`do_sample=True` with `temperature=0.7` but no random seed. Every run produces different outputs, making it impossible to compare configurations.

**The fix:**

```python
# In SteeringConfig:
seed: int = 42

# In generate_with_steering:
if config.seed is not None:
    torch.manual_seed(config.seed)
```

---

## Structural Problems (Quality of Life)

### 9. `requirements.txt` Has Wrong Package Name

**File:** `requirements.txt:4`

`huggingface` is not a real pip package. You need `huggingface_hub`.

```
torch
transformers
numpy
huggingface_hub
accelerate
anthropic
scikit-learn
```

Also: pin versions, or at minimum pin major versions. `torch>=2.0` and `transformers>=4.35` would prevent breakage.

### 10. Global State (`LAYER_IDX`)

**File:** `GCC.py:740`

Module-level `global LAYER_IDX` is unnecessary and creates hidden coupling. The value is only used in `main()` and should stay local. Delete it.

### 11. tqdm Monkey-Patching Is Fragile

**File:** `GCC.py:88-123`

The tqdm patch doesn't restore on exception, can break other code that uses tqdm, and the variable `_OrigTqdm` may not be defined if the import fails but the restore block runs.

**The fix:** Use a context manager or just remove it. The download progress is already visible in the terminal via tqdm's own output. If you need to capture it, use `huggingface_hub`'s built-in progress callbacks instead of monkey-patching.

### 12. No Vector Save/Load in Main Pipeline

**File:** `GCC.py:742-811`

The main pipeline computes a steering vector (expensive) but never saves it. Every run starts from scratch. The agentic script saves to `optimized_gg_vector.pt`, but `GCC.py` doesn't.

**The fix:** Add save/load to the main pipeline:

```python
VECTOR_CACHE = "steering_vector_cache.pt"

# At end of main():
torch.save({
    "vector": steering_vector,
    "layer": best_layer,
    "model": config.model_name,
    "sweep_results": sweep_results,
}, VECTOR_CACHE)

# At start, check for cache:
if os.path.exists(VECTOR_CACHE):
    cached = torch.load(VECTOR_CACHE)
    if cached["model"] == config.model_name:
        print(f"Loaded cached vector from {VECTOR_CACHE}")
        steering_vector = cached["vector"]
        best_layer = cached["layer"]
```

### 13. Agentic Loop Has No Error Recovery

**File:** `agentic_steering.py:149-214`

If the Anthropic API call throws a rate limit error, timeout, or network error, the entire run crashes and all progress is lost.

**The fix:** Add retry logic and state checkpointing:

```python
for iteration in range(max_iterations):
    try:
        response = client.messages.create(...)
    except anthropic.RateLimitError:
        print("Rate limited, waiting 60s...")
        time.sleep(60)
        continue
    except anthropic.APIError as e:
        print(f"API error: {e}, retrying in 10s...")
        time.sleep(10)
        continue
```

---

## Suggested Rewrite Priority

If you can only do a few things, do them in this order:

| Priority | Fix | Time Saved / Impact |
|----------|-----|---------------------|
| **P0** | #1 Multi-layer extraction (single forward pass) | 15-17x speedup on sweep |
| **P0** | #3 Don't re-extract after sweep | Eliminates ~6 min redundant work |
| **P0** | #2 Clean contrastive pairs | Better steering direction quality |
| **P1** | #5 Don't steer prompt tokens | Cleaner generation, less distortion |
| **P1** | #9 Fix requirements.txt | Install actually works |
| **P1** | #4 Architecture-agnostic layer access | Unblocks Gemma/LLaMA work |
| **P2** | #6 Fix validation metric | Meaningful optimization target |
| **P2** | #7 Fix linear probe methodology | Reliable accuracy estimates |
| **P2** | #8 Add seeds | Reproducible experiments |
| **P3** | #10-13 Structural cleanup | Code quality |

---

## What's Already Good

- Last-token extraction via `attention_mask.sum(dim=1) - 1` is correct and avoids a common bug
- Chat template formatting prevents OOD issues with Instruct models
- Layer sweep targeting middle 60% is well-motivated
- The agentic optimization loop is a genuinely clever idea
- The TUI is well-structured with proper thread safety and caching
- Batch-chunked activation extraction prevents OOM correctly
