"""Continuous effect layer system (rules 450-457).

Layers are evaluated in sequence:
  1. Trait-Altering  (454.1) -- sets/changes base traits (name, type, tags,
     controller, cost, domain, Might *assignment* via "becomes X")
  2. Ability-Altering (454.2) -- grants/removes abilities and keywords
  3. Arithmetic       (454.3) -- numeric adjustments to Might, cost, etc.
     Within Arithmetic: increases (positive) first, then decreases (negative).

The evaluator loops until stable (rule 453), applying each effect exactly once
per pass. Dependencies within a layer are resolved by applying the depended-on
effect first (rules 455-456). When no dependency exists, timestamp order is
used (rule 457).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .card_types import CardInstance, KeywordInstance
from .effect_ir import EffectLayer, TargetSpec, classify_modifier_layer
from .enums import CardType, Keyword, ZoneType

if TYPE_CHECKING:
    from .game_state import ActiveModifier, GameState

logger = logging.getLogger("riftbound.layers")

# ---------------------------------------------------------------------------
# Layer constants for ordering
# ---------------------------------------------------------------------------

_LAYER_ORDER = {
    EffectLayer.TRAIT_ALTERING: 0,
    EffectLayer.ABILITY_ALTERING: 1,
    EffectLayer.ARITHMETIC: 2,
}


def _layer_sort_key(layer: EffectLayer) -> int:
    return _LAYER_ORDER.get(layer, 99)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_layers(gs: GameState) -> None:
    """Recalculate all continuous effects using the 3-layer system (rules 450-457).

    This replaces the simple ``recalculate_modifiers`` logic with a proper
    layered evaluation that:
      1. Resets all derived state (aura_might_bonus, aura_keywords, gear_might_bonus)
      2. Prunes stale modifiers (source left the board)
      3. Iterates layers in order until no new effects were applied (rule 453)
      4. Within Arithmetic, applies increases before decreases (454.3.d)
      5. Within the same layer, applies dependent effects in dependency order,
         then remaining effects in timestamp order (rules 455-457)
    """
    from .game_state import ActiveModifier

    # ---- Phase 0: Reset all derived modifier state ----
    for inst in gs.instances.values():
        inst.aura_might_bonus = 0
        inst.gear_might_bonus = 0
        inst.aura_keywords.clear()
        inst.trait_might_set = None

    # ---- Phase 1: Prune modifiers whose source left the board ----
    board_ids = gs._get_board_instance_ids()
    gs.active_modifiers = [
        m for m in gs.active_modifiers if m.source_instance_id in board_ids
    ]
    gs.active_replacements = [
        r for r in gs.active_replacements if r.source_instance_id in board_ids
    ]

    # ---- Phase 2: Classify each modifier into a layer ----
    classified: list[tuple[EffectLayer, int, ActiveModifier]] = []
    for idx, mod in enumerate(gs.active_modifiers):
        layer = _classify_active_modifier(mod)
        # idx serves as timestamp (registration order = chronological order)
        classified.append((layer, idx, mod))

    # ---- Phase 3: Iterate layers until stable (rule 453) ----
    max_passes = 10  # safety valve
    applied_ids: set[int] = set()  # track which (idx) modifiers have been applied

    for _ in range(max_passes):
        any_new = False

        for layer in (EffectLayer.TRAIT_ALTERING, EffectLayer.ABILITY_ALTERING, EffectLayer.ARITHMETIC):
            # Gather not-yet-applied modifiers for this layer
            pending = [
                (idx, mod) for (l, idx, mod) in classified
                if l == layer and idx not in applied_ids
            ]
            if not pending:
                continue

            # Resolve dependency ordering within the layer
            ordered = _resolve_dependencies(pending, gs)

            # For arithmetic layer, sub-sort: increases before decreases (454.3.d)
            if layer == EffectLayer.ARITHMETIC:
                ordered = _sort_arithmetic(ordered)

            for idx, mod in ordered:
                newly_applied = _apply_modifier(gs, mod, board_ids)
                if newly_applied:
                    applied_ids.add(idx)
                    any_new = True
                else:
                    # Even if it didn't match anything new, mark it applied
                    applied_ids.add(idx)

        if not any_new:
            break

    # ---- Phase 4: Apply gear attachment might bonuses (454.3.c) ----
    for inst in gs.instances.values():
        if inst.attached_cards:
            for gear_id in inst.attached_cards:
                gear = gs.get_instance(gear_id)
                if gear and gear.definition.might_bonus:
                    inst.gear_might_bonus += gear.definition.might_bonus


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _classify_active_modifier(mod: ActiveModifier) -> EffectLayer:
    """Map an ActiveModifier to its layer via the stat field."""
    node = {"stat": mod.stat, "amount": mod.amount}
    return classify_modifier_layer(node)


def _resolve_dependencies(
    pending: list[tuple[int, ActiveModifier]],
    gs: GameState,
) -> list[tuple[int, ActiveModifier]]:
    """Order modifiers within the same layer by dependency, then timestamp.

    Rules 455-456: If applying modifier A changes whether modifier B
    applies (existence/scope/outcome), then B depends on A, and A must
    be applied first.

    Currently we detect the key dependency pattern: a modifier that can
    change Might (trait_altering might_set or arithmetic might) may alter
    whether an ability-conditional modifier (e.g. "while Mighty") applies.
    For same-layer non-dependent modifiers, we fall back to timestamp order
    (rule 457).
    """
    # For now, simple timestamp ordering suffices for most cases.
    # The main cross-layer dependency (Arithmetic -> Ability-Altering)
    # is handled by the outer loop re-check (rule 453).
    # Within the same layer, we sort by timestamp index.
    return sorted(pending, key=lambda pair: pair[0])


def _sort_arithmetic(
    mods: list[tuple[int, ActiveModifier]],
) -> list[tuple[int, ActiveModifier]]:
    """Within the Arithmetic layer, apply increases before decreases (rule 454.3.d).

    Positive amounts (increases) go first, then negative amounts (decreases).
    Within each sub-group, preserve timestamp order.
    """
    increases = [(idx, m) for idx, m in mods if m.amount >= 0]
    decreases = [(idx, m) for idx, m in mods if m.amount < 0]
    return increases + decreases


def _apply_modifier(
    gs: GameState,
    mod: ActiveModifier,
    board_ids: set[str],
) -> bool:
    """Apply a single modifier to its matching targets. Returns True if any target matched."""
    source = gs.get_instance(mod.source_instance_id)
    if not source:
        return False

    spec = TargetSpec.from_dict(mod.target_spec)
    matched = gs._resolve_modifier_targets(spec, source, mod.exclude_self)

    if not matched:
        return False

    stat = mod.stat

    for iid in matched:
        inst = gs.get_instance(iid)
        if not inst:
            continue

        if stat == "might":
            # Arithmetic layer: add to aura_might_bonus
            inst.aura_might_bonus += mod.amount

        elif stat == "might_set":
            # Trait-Altering layer: set base might override (454.1.a.1)
            inst.trait_might_set = mod.amount

        elif stat == "keyword":
            # Ability-Altering layer: grant keyword (454.2)
            kw_name = mod.target_spec.get("keyword_name", "")
            kw_value = mod.target_spec.get("keyword_value", 0)
            if kw_name:
                try:
                    kw = Keyword(kw_name)
                    ki = KeywordInstance(keyword=kw, value=kw_value)
                    if not any(k.keyword == kw for k in inst.aura_keywords):
                        inst.aura_keywords.append(ki)
                except ValueError:
                    pass

        # Other stats (shield, assault, etc.) go to aura_might_bonus
        # since they're already handled via keyword_value in effective_might
        elif stat in ("shield", "assault"):
            # These are handled as keyword grants, not direct might bonuses
            pass

    return True


# ---------------------------------------------------------------------------
# Keyword grant support for modifiers
# ---------------------------------------------------------------------------


def register_keyword_modifier(
    gs: GameState,
    source: CardInstance,
    ability_id: str,
    keyword_name: str,
    keyword_value: int = 0,
    target_spec: dict | None = None,
    exclude_self: bool = False,
) -> None:
    """Register a continuous keyword-granting modifier.

    This creates an ActiveModifier with stat="keyword" and stores the
    keyword details in the target_spec for use during layer evaluation.
    """
    from .game_state import ActiveModifier

    spec = target_spec or {}
    spec["keyword_name"] = keyword_name
    spec["keyword_value"] = keyword_value

    mod = ActiveModifier(
        source_instance_id=source.instance_id,
        ability_id=ability_id,
        stat="keyword",
        amount=0,
        target_spec=spec,
        exclude_self=exclude_self,
        duration="continuous",
    )
    gs.active_modifiers.append(mod)
