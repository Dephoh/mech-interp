"""
Agentic Steering Toolkit
=========================
Wraps GCC.py functions into tool-callable methods for Claude Opus
to act as a mechanistic interpretability researcher. Holds shared state
(model, tokenizer, active pairs, current vector) across tool calls.

All methods return JSON-serializable dicts.
"""

import json
import torch
import numpy as np
from typing import Optional

from GCC import (
    SteeringConfig,
    create_contrastive_prompts,
    format_for_model,
    get_last_token_activations,
    extract_all_layers,
    get_transformer_layers,
    compute_steering_vector,
    sweep_layers,
    generate_with_steering,
    _get_device,
)


# ============================================================================
# Keywords used for Golden Gate density scoring
# ============================================================================

GG_KEYWORDS = [
    "golden gate", "bridge", "san francisco", "orange",
    "span", "bay", "tower", "cable", "fog", "marin",
]

# Test prompts for functional quality evaluation
VALIDATION_PROMPTS = [
    "What's your favorite color?",
    "Describe yourself.",
    "What makes you happy?",
    "Tell me about your day.",
    "What should I have for dinner?",
]

# Multipliers for dose-response evaluation sweep
SWEEP_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]


def _keyword_density(text: str) -> float:
    """Compute Golden Gate keyword density for a given text."""
    lower = text.lower()
    count = sum(lower.count(kw) for kw in GG_KEYWORDS)
    word_count = max(len(text.split()), 1)
    return count / word_count


# ============================================================================
# Activation Cache (eliminates redundant forward passes)
# ============================================================================

class ActivationCache:
    """
    Caches layer activations keyed by (layer_idx, "pos"|"neg").
    After a single sweep, all subsequent analyze/recompute/probe calls
    at any swept layer are instant (dict lookup instead of forward pass).
    """

    def __init__(self):
        self._cache: dict[tuple[int, str], torch.Tensor] = {}

    def get(self, layer_idx: int, side: str) -> Optional[torch.Tensor]:
        return self._cache.get((layer_idx, side))

    def put(self, layer_idx: int, side: str, tensor: torch.Tensor):
        self._cache[(layer_idx, side)] = tensor

    def has(self, layer_idx: int) -> bool:
        return (layer_idx, "pos") in self._cache and (layer_idx, "neg") in self._cache

    def populate_from_sweep(self, all_pos: dict, all_neg: dict):
        """Bulk-load activations from sweep_layers() results."""
        for layer_idx, tensor in all_pos.items():
            self.put(layer_idx, "pos", tensor)
        for layer_idx, tensor in all_neg.items():
            self.put(layer_idx, "neg", tensor)

    def clear(self):
        self._cache.clear()

    @property
    def cached_layers(self) -> list[int]:
        return sorted(set(k[0] for k in self._cache))


# ============================================================================
# LLM Judge (structured scoring via Claude Haiku)
# ============================================================================

# Tool schema sent to Haiku for structured output
_JUDGE_TOOL = {
    "name": "score_output",
    "description": "Score the steered model output.",
    "input_schema": {
        "type": "object",
        "properties": {
            "identity": {
                "type": "integer",
                "description": (
                    "Does the model speak AS the Golden Gate Bridge? "
                    "0=normal AI response, 1=mentions bridge in passing, "
                    "2=relates everything to bridge, 3=fully identifies as the bridge in first person"
                ),
            },
            "relevance": {
                "type": "integer",
                "description": (
                    "Are Golden Gate Bridge concepts present? "
                    "0=none, 1=one mention, 2=several references, 3=pervasive throughout"
                ),
            },
            "coherence": {
                "type": "integer",
                "description": (
                    "Is the output fluent and responsive to the prompt? "
                    "0=gibberish/repetitive, 1=barely readable, 2=mostly coherent, 3=natural"
                ),
            },
        },
        "required": ["identity", "relevance", "coherence"],
    },
}

_JUDGE_SYSTEM = """\
You are evaluating whether activation steering made a language model adopt \
a "Golden Gate Bridge" identity, similar to Anthropic's "Golden Gate Claude".

The model was given a normal user prompt. You will see the unsteered baseline \
and the steered output. Score the STEERED output using the score_output tool.\
"""


def _call_llm_judge(prompt: str, baseline: str, steered: str) -> dict:
    """
    Call Claude Haiku to judge a steered output. Returns dict with
    identity, relevance, coherence (0-3 each) and composite (0-1).
    """
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=_JUDGE_SYSTEM,
        tools=[_JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "score_output"},
        messages=[{
            "role": "user",
            "content": (
                f"User prompt: {prompt}\n\n"
                f"Unsteered baseline:\n{baseline}\n\n"
                f"Steered output:\n{steered}"
            ),
        }],
    )

    # Extract tool_use result
    for block in response.content:
        if block.type == "tool_use":
            scores = block.input
            identity = max(0, min(3, int(scores.get("identity", 0))))
            relevance = max(0, min(3, int(scores.get("relevance", 0))))
            coherence = max(0, min(3, int(scores.get("coherence", 0))))
            composite = (identity * 2 + relevance + coherence) / 12.0
            return {
                "identity": identity,
                "relevance": relevance,
                "coherence": coherence,
                "composite": round(composite, 4),
            }

    return {"identity": 0, "relevance": 0, "coherence": 0, "composite": 0.0,
            "error": "Judge returned no tool_use block"}


# ============================================================================
# SteeringToolkit
# ============================================================================

class SteeringToolkit:
    """
    Stateful toolkit that Opus calls via tools. Wraps GCC.py functions
    and maintains shared state across the agentic loop.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.config = SteeringConfig()
        self.n_layers = model.config.num_hidden_layers
        self.transformer_layers = get_transformer_layers(model)

        # All contrastive pairs (may be filtered during optimization)
        self.all_pairs = create_contrastive_prompts()
        self.active_pair_indices = list(range(len(self.all_pairs)))

        # Current steering vector state
        self.current_vector: Optional[torch.Tensor] = None
        self.current_layer: Optional[int] = None

        # Cache for sweep results
        self._sweep_cache: Optional[dict] = None

        # Activation cache (populated by sweep, reused by all tools)
        self._act_cache = ActivationCache()

    def _get_activations(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get pos/neg activations for a layer, using cache if available.
        Falls back to fresh extraction if layer was not swept.
        """
        if self._act_cache.has(layer_idx):
            return self._act_cache.get(layer_idx, "pos"), self._act_cache.get(layer_idx, "neg")

        # Cache miss — extract and cache
        pairs = self.all_pairs
        pos_texts = [format_for_model(self.tokenizer, p) for p, _ in pairs]
        neg_texts = [format_for_model(self.tokenizer, n) for _, n in pairs]

        pos_acts = get_last_token_activations(
            self.model, self.tokenizer, pos_texts, layer_idx, self.config.batch_size
        )
        neg_acts = get_last_token_activations(
            self.model, self.tokenizer, neg_texts, layer_idx, self.config.batch_size
        )

        self._act_cache.put(layer_idx, "pos", pos_acts)
        self._act_cache.put(layer_idx, "neg", neg_acts)
        return pos_acts, neg_acts

    # ------------------------------------------------------------------
    # Tool: sweep_and_score_layers
    # ------------------------------------------------------------------

    def sweep_and_score_layers(self, layers: Optional[list[int]] = None) -> dict:
        """
        Sweep layers and return per-layer consistency + magnitude scores.
        If layers is None, sweeps the middle 60% of model layers.
        Uses single-pass extraction for efficiency.
        Populates the activation cache for all swept layers.
        """
        result = sweep_layers(
            self.model,
            self.tokenizer,
            layers_to_test=layers,
            batch_size=self.config.batch_size,
        )
        self._sweep_cache = result

        # Populate activation cache from sweep (eliminates all future re-extraction)
        if "all_pos_acts" in result and "all_neg_acts" in result:
            self._act_cache.populate_from_sweep(
                result["all_pos_acts"], result["all_neg_acts"]
            )

        # Convert per_layer keys to strings for JSON
        per_layer = {
            str(k): v for k, v in result["per_layer"].items()
        }
        return {
            "best_layer": result["best_layer"],
            "best_score": round(result["best_score"], 4),
            "per_layer": per_layer,
            "n_layers_tested": len(per_layer),
            "cached_layers": self._act_cache.cached_layers,
        }

    # ------------------------------------------------------------------
    # Tool: analyze_pair_gaps
    # ------------------------------------------------------------------

    def analyze_pair_gaps(self, layer_idx: int) -> dict:
        """
        For each contrastive pair, compute pos_proj - neg_proj on the
        mean-diff steering direction. Returns sorted list so Opus can
        identify outlier pairs. Uses cached activations when available.
        """
        pairs = self.all_pairs
        pos_acts, neg_acts = self._get_activations(layer_idx)

        # Mean-diff direction
        diff = (pos_acts - neg_acts).mean(dim=0)
        direction = diff / diff.norm()

        # Per-pair projections
        sv = direction.unsqueeze(0)
        pos_proj = torch.nn.functional.cosine_similarity(pos_acts, sv, dim=1)
        neg_proj = torch.nn.functional.cosine_similarity(neg_acts, sv, dim=1)

        pair_data = []
        for i, (pair, pp, np_val) in enumerate(zip(pairs, pos_proj, neg_proj)):
            gap = (pp - np_val).item()
            pair_data.append({
                "index": i,
                "positive": pair[0][:80],
                "negative": pair[1][:80],
                "pos_proj": round(pp.item(), 4),
                "neg_proj": round(np_val.item(), 4),
                "gap": round(gap, 4),
            })

        # Sort by gap ascending so outliers (low gap) are first
        pair_data.sort(key=lambda x: x["gap"])

        mean_gap = sum(d["gap"] for d in pair_data) / len(pair_data)
        return {
            "layer": layer_idx,
            "n_pairs": len(pair_data),
            "mean_gap": round(mean_gap, 4),
            "min_gap": round(pair_data[0]["gap"], 4),
            "max_gap": round(pair_data[-1]["gap"], 4),
            "pairs": pair_data,
            "cached": self._act_cache.has(layer_idx),
        }

    # ------------------------------------------------------------------
    # Tool: recompute_vector
    # ------------------------------------------------------------------

    def recompute_vector(
        self, layer_idx: int, min_gap: Optional[float] = None
    ) -> dict:
        """
        Recompute the steering vector at layer_idx. If min_gap is provided,
        first filter out pairs whose gap (from analyze_pair_gaps) is below
        this threshold, then recompute on the remaining pairs.
        Uses cached activations when available.
        """
        pairs = self.all_pairs
        pos_acts, neg_acts = self._get_activations(layer_idx)

        if min_gap is not None:
            diff = (pos_acts - neg_acts).mean(dim=0)
            direction = diff / diff.norm()
            sv = direction.unsqueeze(0)
            pos_proj = torch.nn.functional.cosine_similarity(pos_acts, sv, dim=1)
            neg_proj = torch.nn.functional.cosine_similarity(neg_acts, sv, dim=1)
            gaps = (pos_proj - neg_proj).tolist()

            # Filter
            keep_indices = [i for i, g in enumerate(gaps) if g >= min_gap]

            if len(keep_indices) < 5:
                return {
                    "error": f"Only {len(keep_indices)} pairs survive min_gap={min_gap}. "
                             "Need at least 5. Try a lower threshold.",
                    "n_surviving": len(keep_indices),
                }

            self.active_pair_indices = keep_indices
            filtered_pos = pos_acts[keep_indices]
            filtered_neg = neg_acts[keep_indices]

            steering_vec = compute_steering_vector(
                pos_acts=filtered_pos, neg_acts=filtered_neg,
            )
        else:
            # Use all pairs
            self.active_pair_indices = list(range(len(pairs)))
            steering_vec = compute_steering_vector(
                pos_acts=pos_acts, neg_acts=neg_acts,
            )

        self.current_vector = steering_vec
        self.current_layer = layer_idx

        return {
            "layer": layer_idx,
            "n_pairs_used": len(self.active_pair_indices),
            "n_pairs_total": len(self.all_pairs),
            "min_gap_threshold": min_gap,
            "vector_dim": steering_vec.shape[0],
            "vector_norm": round(steering_vec.norm().item(), 4),
            "cached": self._act_cache.has(layer_idx),
        }

    # ------------------------------------------------------------------
    # Tool: train_linear_probe
    # ------------------------------------------------------------------

    def train_linear_probe(self, layer_idx: int) -> dict:
        """
        Train a logistic regression probe on the active pairs at layer_idx.
        Uses 5-fold stratified cross-validation with regularization to avoid
        overfitting in high-dimensional space. Returns mean CV accuracy.
        Uses cached activations when available.
        """
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold

        active = self.active_pair_indices

        # Get full activations (cached), then index active pairs
        pos_acts, neg_acts = self._get_activations(layer_idx)
        pos_acts = pos_acts[active]
        neg_acts = neg_acts[active]

        # Build dataset: label 1 = positive (Golden Gate), 0 = negative
        X = torch.cat([pos_acts, neg_acts], dim=0).numpy()
        y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

        # 5-fold stratified CV with automatic regularization tuning
        n_splits = min(5, len(active))  # handle very small datasets
        if n_splits < 2:
            return {
                "error": f"Too few samples for CV ({len(active)} pairs = {2*len(active)} samples). "
                         "Need at least 2 pairs.",
                "layer": layer_idx,
            }

        clf = LogisticRegressionCV(
            Cs=10,
            cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X, y)

        # Mean CV accuracy across folds at best C
        # clf.scores_[class_label] has shape (n_folds, n_Cs)
        cv_scores = clf.scores_[1]  # scores for positive class
        best_c_idx = np.argmax(cv_scores.mean(axis=0))
        per_fold_accs = cv_scores[:, best_c_idx]
        mean_cv_acc = per_fold_accs.mean()

        return {
            "layer": layer_idx,
            "n_samples_total": len(X),
            "n_folds": n_splits,
            "mean_cv_accuracy": round(float(mean_cv_acc), 4),
            "per_fold_accuracies": [round(float(a), 4) for a in per_fold_accs],
            "best_C": round(float(clf.C_[0]), 6),
            "cached": self._act_cache.has(layer_idx),
        }

    # ------------------------------------------------------------------
    # Tool: evaluate_generation
    # ------------------------------------------------------------------

    def evaluate_generation(self, prompt: str, multiplier: float) -> dict:
        """
        Generate text with the current steering vector and return
        the output text plus keyword density score.
        """
        if self.current_vector is None or self.current_layer is None:
            return {"error": "No steering vector computed yet. Call recompute_vector first."}

        output = generate_with_steering(
            self.model,
            self.tokenizer,
            prompt,
            self.current_vector,
            layer_idx=self.current_layer,
            multiplier=multiplier,
            max_new_tokens=self.config.max_new_tokens,
            seed=self.config.seed,
        )

        density = _keyword_density(output)

        return {
            "prompt": prompt,
            "multiplier": multiplier,
            "layer": self.current_layer,
            "output": output.strip(),
            "keyword_density": round(density, 4),
            "word_count": len(output.split()),
        }

    # ------------------------------------------------------------------
    # Tool: try_multilayer_config
    # ------------------------------------------------------------------

    def try_multilayer_config(
        self, layers: list[int], multiplier: float, prompt: str
    ) -> dict:
        """
        Test a multi-layer steering configuration. The multiplier is
        divided across the specified layers.
        """
        if self.current_vector is None:
            return {"error": "No steering vector computed yet. Call recompute_vector first."}

        output = generate_with_steering(
            self.model,
            self.tokenizer,
            prompt,
            self.current_vector,
            layer_idx=layers[0],
            multiplier=multiplier,
            steering_layers=layers,
            max_new_tokens=self.config.max_new_tokens,
            seed=self.config.seed,
        )

        density = _keyword_density(output)

        return {
            "prompt": prompt,
            "multiplier": multiplier,
            "layers": layers,
            "per_layer_multiplier": round(multiplier / len(layers), 4),
            "output": output.strip(),
            "keyword_density": round(density, 4),
            "word_count": len(output.split()),
        }

    # ------------------------------------------------------------------
    # Tool: llm_judge_score
    # ------------------------------------------------------------------

    def llm_judge_score(self, prompt: str, multiplier: float) -> dict:
        """
        Generate steered output and have Claude Haiku judge it on
        identity/relevance/coherence (0-3 each). Returns composite (0-1).
        Also generates an unsteered baseline for comparison.
        """
        if self.current_vector is None or self.current_layer is None:
            return {"error": "No steering vector computed yet. Call recompute_vector first."}

        # Generate unsteered baseline
        baseline = generate_with_steering(
            self.model, self.tokenizer, prompt, self.current_vector,
            layer_idx=self.current_layer, multiplier=0.0,
            max_new_tokens=self.config.max_new_tokens, seed=self.config.seed,
        ).strip()

        # Generate steered output
        steered = generate_with_steering(
            self.model, self.tokenizer, prompt, self.current_vector,
            layer_idx=self.current_layer, multiplier=multiplier,
            max_new_tokens=self.config.max_new_tokens, seed=self.config.seed,
        ).strip()

        # Call LLM judge
        judge_result = _call_llm_judge(prompt, baseline, steered)

        return {
            "prompt": prompt,
            "multiplier": multiplier,
            "layer": self.current_layer,
            "baseline_preview": baseline[:200],
            "steered_preview": steered[:200],
            "keyword_density": round(_keyword_density(steered), 4),
            **judge_result,
        }

    # ------------------------------------------------------------------
    # Tool: run_evaluation_sweep
    # ------------------------------------------------------------------

    def run_evaluation_sweep(self) -> dict:
        """
        Run a full multiplier sweep across all validation prompts.
        Returns a dose-response table showing whether increasing
        multiplier increases Golden Gate identity. Key diagnostic for
        "is steering working at all?"
        """
        if self.current_vector is None or self.current_layer is None:
            return {"error": "No steering vector computed yet. Call recompute_vector first."}

        # Generate all baselines once (mult=0)
        baselines = {}
        for prompt in VALIDATION_PROMPTS:
            baselines[prompt] = generate_with_steering(
                self.model, self.tokenizer, prompt, self.current_vector,
                layer_idx=self.current_layer, multiplier=0.0,
                max_new_tokens=self.config.max_new_tokens, seed=self.config.seed,
            ).strip()

        # Sweep multipliers
        rows = []
        for mult in SWEEP_MULTIPLIERS:
            mult_scores = []
            for prompt in VALIDATION_PROMPTS:
                if mult == 0.0:
                    steered = baselines[prompt]
                else:
                    steered = generate_with_steering(
                        self.model, self.tokenizer, prompt, self.current_vector,
                        layer_idx=self.current_layer, multiplier=mult,
                        max_new_tokens=self.config.max_new_tokens, seed=self.config.seed,
                    ).strip()

                # LLM judge
                judge = _call_llm_judge(prompt, baselines[prompt], steered)
                density = _keyword_density(steered)

                mult_scores.append({
                    "prompt": prompt[:40],
                    "identity": judge["identity"],
                    "relevance": judge["relevance"],
                    "coherence": judge["coherence"],
                    "composite": judge["composite"],
                    "keyword_density": round(density, 4),
                    "output_preview": steered[:100],
                })

            mean_identity = sum(s["identity"] for s in mult_scores) / len(mult_scores)
            mean_composite = sum(s["composite"] for s in mult_scores) / len(mult_scores)
            mean_density = sum(s["keyword_density"] for s in mult_scores) / len(mult_scores)

            rows.append({
                "multiplier": mult,
                "mean_identity": round(mean_identity, 2),
                "mean_composite": round(mean_composite, 4),
                "mean_keyword_density": round(mean_density, 4),
                "per_prompt": mult_scores,
            })

        # Summary: is there a dose-response effect?
        identity_at_0 = rows[0]["mean_identity"]
        identity_at_3 = next(r for r in rows if r["multiplier"] == 3.0)["mean_identity"]
        effect_detected = identity_at_3 > identity_at_0 + 0.5

        return {
            "layer": self.current_layer,
            "n_prompts": len(VALIDATION_PROMPTS),
            "multipliers_tested": SWEEP_MULTIPLIERS,
            "dose_response": rows,
            "effect_detected": effect_detected,
            "identity_at_0": round(identity_at_0, 2),
            "identity_at_3": round(identity_at_3, 2),
        }

    # ------------------------------------------------------------------
    # Tool: get_validation_score
    # ------------------------------------------------------------------

    def get_validation_score(self) -> dict:
        """
        Compute the composite validation metric (0-1 scale).

        Gated metric:
        - If probe CV accuracy < 0.7: score = probe_acc * 0.5 (penalize)
        - If probe CV accuracy >= 0.7: score = 0.2 * probe_acc + 0.8 * mean_judge_composite

        Uses LLM judge for nuanced scoring (identity/relevance/coherence)
        instead of keyword density alone.
        """
        if self.current_vector is None or self.current_layer is None:
            return {"error": "No steering vector computed yet. Call recompute_vector first."}

        # 1. Probe CV accuracy
        probe_result = self.train_linear_probe(self.current_layer)
        if "error" in probe_result:
            return probe_result
        probe_cv_acc = probe_result["mean_cv_accuracy"]

        # 2. LLM judge scoring at multiplier=1.0
        judge_scores = []
        for prompt in VALIDATION_PROMPTS:
            result = self.llm_judge_score(prompt, multiplier=1.0)
            if "error" in result:
                return result
            judge_scores.append(result)

        mean_identity = sum(s["identity"] for s in judge_scores) / len(judge_scores)
        mean_relevance = sum(s["relevance"] for s in judge_scores) / len(judge_scores)
        mean_coherence = sum(s["coherence"] for s in judge_scores) / len(judge_scores)
        mean_composite = sum(s["composite"] for s in judge_scores) / len(judge_scores)
        mean_density = sum(s["keyword_density"] for s in judge_scores) / len(judge_scores)

        # Gated composite score
        if probe_cv_acc < 0.7:
            composite = probe_cv_acc * 0.5
        else:
            composite = 0.2 * probe_cv_acc + 0.8 * mean_composite

        judge_samples = [{
            "prompt": s["prompt"],
            "identity": s["identity"],
            "relevance": s["relevance"],
            "coherence": s["coherence"],
            "composite": s["composite"],
            "steered_preview": s["steered_preview"],
        } for s in judge_scores]

        return {
            "composite_score": round(composite, 4),
            "probe_cv_accuracy": round(probe_cv_acc, 4),
            "probe_gate_passed": probe_cv_acc >= 0.7,
            "mean_identity": round(mean_identity, 2),
            "mean_relevance": round(mean_relevance, 2),
            "mean_coherence": round(mean_coherence, 2),
            "mean_judge_composite": round(mean_composite, 4),
            "mean_keyword_density": round(mean_density, 4),
            "layer": self.current_layer,
            "n_active_pairs": len(self.active_pair_indices),
            "judge_samples": judge_samples,
        }


# ============================================================================
# Tool JSON Schemas for the Anthropic API
# ============================================================================

TOOL_SCHEMAS = [
    {
        "name": "sweep_and_score_layers",
        "description": (
            "Sweep across model layers and score each by cosine consistency "
            "and magnitude of the contrastive activation difference. "
            "Uses single-pass extraction for efficiency. "
            "Returns per-layer scores and the best layer. "
            "Also populates the activation cache for all swept layers, "
            "making subsequent analyze/recompute/probe calls instant. "
            "Use this first to survey the landscape."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Optional list of specific layer indices to test. "
                        "If omitted, tests the middle 60% of layers."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "analyze_pair_gaps",
        "description": (
            "For each contrastive pair, compute the projection gap "
            "(pos_proj - neg_proj) onto the mean-diff steering direction at "
            "the given layer. Returns pairs sorted by gap (ascending), so "
            "outlier/noisy pairs appear first. Use this to identify which "
            "pairs to filter out. Uses cached activations (instant if layer was swept)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layer_idx": {
                    "type": "integer",
                    "description": "The layer index to analyze.",
                },
            },
            "required": ["layer_idx"],
        },
    },
    {
        "name": "recompute_vector",
        "description": (
            "Recompute the steering vector at a given layer. Optionally filter "
            "out pairs whose gap is below min_gap threshold before computing. "
            "Stores the result as the current active steering vector. "
            "Uses cached activations (instant if layer was swept)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layer_idx": {
                    "type": "integer",
                    "description": "The layer index to compute the vector at.",
                },
                "min_gap": {
                    "type": "number",
                    "description": (
                        "Optional minimum gap threshold. Pairs with gap < min_gap "
                        "are excluded. If omitted, all pairs are used."
                    ),
                },
            },
            "required": ["layer_idx"],
        },
    },
    {
        "name": "train_linear_probe",
        "description": (
            "Train a logistic regression classifier on the active contrastive "
            "pairs at the given layer. Uses 5-fold stratified cross-validation "
            "with automatic regularization tuning. Returns mean CV accuracy "
            "and per-fold accuracies. The mean_cv_accuracy is a key component "
            "of the optimization metric. Uses cached activations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layer_idx": {
                    "type": "integer",
                    "description": "The layer index to extract activations from.",
                },
            },
            "required": ["layer_idx"],
        },
    },
    {
        "name": "evaluate_generation",
        "description": (
            "Generate text with the current steering vector applied at a given "
            "multiplier. Returns the output text and Golden Gate keyword density. "
            "Use this for quick qualitative checks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The user prompt to generate a response for.",
                },
                "multiplier": {
                    "type": "number",
                    "description": (
                        "Steering strength. 0.0 = off, 1.0 = mild, 3.0 = strong, "
                        "5.0+ = likely incoherent."
                    ),
                },
            },
            "required": ["prompt", "multiplier"],
        },
    },
    {
        "name": "try_multilayer_config",
        "description": (
            "Test a multi-layer steering configuration. The multiplier is divided "
            "across the specified layers. Use this to test whether steering at "
            "multiple layers produces better results than a single layer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of layer indices to steer simultaneously.",
                },
                "multiplier": {
                    "type": "number",
                    "description": "Total steering strength (divided across layers).",
                },
                "prompt": {
                    "type": "string",
                    "description": "The user prompt to generate a response for.",
                },
            },
            "required": ["layers", "multiplier", "prompt"],
        },
    },
    {
        "name": "llm_judge_score",
        "description": (
            "Generate steered output and have Claude Haiku judge it on three "
            "dimensions (0-3 each): identity (does model speak AS the bridge?), "
            "relevance (are GGB concepts present?), coherence (is output fluent?). "
            "Returns composite score (0-1, identity weighted 2x). "
            "Also generates unsteered baseline for comparison. "
            "Use this for nuanced quality assessment beyond keyword counting."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The user prompt to generate a response for.",
                },
                "multiplier": {
                    "type": "number",
                    "description": "Steering strength to test.",
                },
            },
            "required": ["prompt", "multiplier"],
        },
    },
    {
        "name": "run_evaluation_sweep",
        "description": (
            "Run a full multiplier sweep (0.0, 0.5, 1.0, 2.0, 3.0, 5.0) across "
            "all 5 validation prompts. For each combination, computes keyword "
            "density and LLM judge scores. Returns a dose-response table showing "
            "whether increasing multiplier increases Golden Gate identity. "
            "This is the key diagnostic for 'is steering working at all?'. "
            "Use this early to check if the vector has any effect before spending "
            "time optimizing. Warning: this makes ~30 generation + ~30 judge calls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_validation_score",
        "description": (
            "Compute the composite validation metric (0-1 scale). Uses a gated "
            "metric: if probe CV accuracy < 0.7, score is penalized. If probe "
            "passes, score = 0.2 * probe_acc + 0.8 * mean_judge_composite. "
            "Uses LLM judge for nuanced scoring (identity/relevance/coherence). "
            "This is the number you are trying to maximize. Call this after "
            "you've found your best configuration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]
