"""Effect IR tree walker — resolves effect nodes into game state mutations.

The resolver takes an IR node (a dict with a "type" key), a source CardInstance,
a GameState, and optionally pre-resolved targets, then walks the tree calling
primitives or recursing into composition nodes.

Usage:
    from .effect_resolver import resolve_effect_ir
    logs = resolve_effect_ir(ir_node, source, gs, targets)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .effect_ir import (
    ALL_NODE_TYPES,
    ARRAY,
    COMPOSITION_TYPES,
    CONDITIONAL,
    CHOOSE_ONE,
    DELAYED_TRIGGER,
    DESCRIPTOR_TYPES,
    FOR_EACH,
    MODIFIER,
    OPTIONAL,
    PLAYER_CHOICE,
    PRIMITIVE_TYPES,
    REPLACEMENT,
    REPEAT_EFFECT,
    RULE_BREAKER_TYPES,
    SEQUENCE,
    TRIGGERED_EFFECT,
    ConditionSpec,
    TargetSpec,
)
from .effect_primitives import PRIMITIVE_DISPATCH

if TYPE_CHECKING:
    from .card_types import CardInstance
    from .game_state import GameState

logger = logging.getLogger("riftbound.resolver")

# Monotonic counter used to stamp modifiers/replacements with a relative
# timestamp so that effects in the same layer can be ordered chronologically
# (rule 457).  The counter is module-level so it survives across calls.
_timestamp_counter: int = 0


def _next_timestamp() -> int:
    """Return the next monotonically increasing timestamp (rule 457.1)."""
    global _timestamp_counter
    _timestamp_counter += 1
    return _timestamp_counter


def resolve_effect_ir(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str] | None = None,
) -> list[str]:
    """Walk an IR tree and execute it against the game state.

    Args:
        node: An effect IR node (dict with "type" key).
        source: The card/ability that produced this effect.
        gs: The current game state (mutated in place).
        targets: Pre-resolved target instance_ids chosen by the player or
                 auto-player. Consumed left-to-right by primitives that need them.

    Returns:
        List of log message strings describing what happened.
    """
    if not node or not isinstance(node, dict):
        return ["Empty effect node"]

    node_type = node.get("type")
    if not node_type:
        return ["Effect node missing type"]

    if node_type not in ALL_NODE_TYPES:
        logger.warning("Unknown effect node type: %s", node_type)
        return [f"Unknown effect: {node_type}"]

    if targets is None:
        targets = []

    if node_type in PRIMITIVE_TYPES:
        return _resolve_primitive(node, source, gs, targets)
    elif node_type in COMPOSITION_TYPES:
        return _resolve_composition(node, source, gs, targets)
    elif node_type in RULE_BREAKER_TYPES:
        return _resolve_rule_breaker(node, source, gs, targets)
    elif node_type in DESCRIPTOR_TYPES:
        # Descriptor nodes (e.g. "event") are sub-node metadata, not
        # standalone executable effects.  No-op if reached directly.
        return []
    else:
        return [f"Unhandled node type: {node_type}"]


def _resolve_primitive(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Execute a leaf (primitive) node.

    Before executing, broadcasts intent to the replacement system. If an
    active ReplacementNode intercepts this primitive, the alternative IR
    is executed instead.

    String-valued ``amount`` / ``count`` fields are resolved to ints via
    :func:`~.effect_primitives._resolve_amount` *before* the primitive
    function is called, so primitives always receive concrete numbers.
    """
    from .effect_primitives import _resolve_amount
    from .trigger_system import check_primitive_replacement

    node_type = node["type"]
    fn = PRIMITIVE_DISPATCH.get(node_type)
    if fn is None:
        return [f"No primitive handler for: {node_type}"]

    # Determine which targets to pass to the primitive.
    # If the node has a "target" spec and no targets were pre-resolved,
    # we try to auto-resolve from the game state.
    resolved = targets
    if not resolved and "target" in node:
        resolved = _auto_resolve_targets(node["target"], source, gs)

    # --- Pre-resolve dynamic string amounts --------------------------------
    # Make a shallow copy so we don't mutate the original IR definition.
    # Only the top-level keys are replaced; nested dicts (like "target")
    # remain shared references, which is safe since we only touch "amount"
    # and "count" here.
    #
    # Expressions that depend on which target is being iterated
    # (e.g. "reciprocal_might", "current_might", "target.might") are
    # left as strings here; the per-target loop inside each primitive
    # will call _resolve_amount with the correct target context.
    _PER_TARGET_EXPRESSIONS = frozenset({
        "reciprocal_might", "current_might", "doubled",
        "target.might", "target_might", "target.cost",
    })
    params = dict(node)
    for key in ("amount", "count"):
        raw = params.get(key)
        if isinstance(raw, str) and raw != "all" and raw not in _PER_TARGET_EXPRESSIONS:
            # Source-scoped expression: resolve once for the whole call
            first_target = None
            if resolved:
                first_target = gs.get_instance(resolved[0])
            params[key] = _resolve_amount(
                raw, source, first_target,
                all_targets=resolved, gs=gs,
            )

    # Broadcast intent — check if any replacement intercepts this primitive
    alt_ir = check_primitive_replacement(gs, node_type, source, resolved)
    if alt_ir:
        logs = resolve_effect_ir(alt_ir, source, gs, resolved)
        gs.last_effect_succeeded = bool(logs and not any("No valid" in l for l in logs))
        return logs

    logs = fn(source, gs, params, resolved)
    # Track success: a primitive succeeded if it produced logs without "No valid" failures
    gs.last_effect_succeeded = bool(logs and not any("No valid" in l for l in logs))
    return logs


def _resolve_composition(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Execute a branch (composition) node."""
    node_type = node["type"]

    if node_type == SEQUENCE:
        return _resolve_sequence(node, source, gs, targets)
    elif node_type == CONDITIONAL:
        return _resolve_conditional(node, source, gs, targets)
    elif node_type == FOR_EACH:
        return _resolve_for_each(node, source, gs, targets)
    elif node_type == CHOOSE_ONE:
        return _resolve_choose_one(node, source, gs, targets)
    elif node_type == ARRAY:
        return _resolve_array(node, source, gs, targets)
    elif node_type == OPTIONAL:
        return _resolve_optional(node, source, gs, targets)
    elif node_type == REPEAT_EFFECT:
        return _resolve_repeat(node, source, gs, targets)
    else:
        return [f"Unhandled composition: {node_type}"]


def _resolve_sequence(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Execute steps in order, accumulating logs."""
    logs: list[str] = []
    for step in node.get("steps", []):
        logs.extend(resolve_effect_ir(step, source, gs, targets))
    return logs


def _resolve_conditional(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Evaluate condition, execute then or else branch."""
    condition = node.get("condition", {})
    if evaluate_condition(condition, source, gs):
        then_node = node.get("then")
        if then_node:
            return resolve_effect_ir(then_node, source, gs, targets)
    else:
        else_node = node.get("else")
        if else_node:
            return resolve_effect_ir(else_node, source, gs, targets)
    return []


def _resolve_for_each(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Apply effect to each target matching the spec."""
    target_spec = dict(node.get("targets", {}))
    # for_each always iterates ALL matching targets
    target_spec["count"] = -1
    matched = _auto_resolve_targets(target_spec, source, gs)
    effect = node.get("effect")
    if not effect:
        return []

    logs: list[str] = []
    for tid in matched:
        logs.extend(resolve_effect_ir(effect, source, gs, [tid]))
    return logs


def _resolve_choose_one(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Choose one option to execute. Auto-play: pick first valid option."""
    options = node.get("options", [])
    if not options:
        return []
    # Auto-resolution: execute the first option
    # In a real game, this would require player input
    return resolve_effect_ir(options[0], source, gs, targets)


def _resolve_array(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Execute all options in an array node (multi-modal ability list).

    The "array" type holds multiple sub-effects (like optional abilities
    on a card).  Each option is resolved independently and sequentially.
    """
    logs: list[str] = []
    for option in node.get("options", []):
        logs.extend(resolve_effect_ir(option, source, gs, targets))
    return logs


def _resolve_optional(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """'You may' effect — auto-play always does it if targets are available."""
    effect = node.get("effect")
    if not effect:
        return []
    # Auto-resolution: always execute optional effects
    return resolve_effect_ir(effect, source, gs, targets)


def _resolve_repeat(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Execute the effect again (for Repeat keyword). Just runs effect once more."""
    effect = node.get("effect")
    if not effect:
        return []
    return resolve_effect_ir(effect, source, gs, targets)


# ---------------------------------------------------------------------------
# Rule-breaker node resolution (modifier, replacement, player_choice)
# ---------------------------------------------------------------------------

def _resolve_rule_breaker(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Resolve a rule-breaker (stateful/interactive) node."""
    node_type = node["type"]

    if node_type == MODIFIER:
        return _resolve_modifier(node, source, gs)
    elif node_type == REPLACEMENT:
        return _resolve_replacement(node, source, gs)
    elif node_type == PLAYER_CHOICE:
        return _resolve_player_choice(node, source, gs, targets)
    elif node_type == DELAYED_TRIGGER:
        return _resolve_delayed_trigger(node, source, gs)
    elif node_type == TRIGGERED_EFFECT:
        return _resolve_triggered_effect(node, source, gs)
    else:
        return [f"Unhandled rule-breaker: {node_type}"]


def _resolve_modifier(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
) -> list[str]:
    """Register a continuous modifier (aura) on the game state.

    The modifier is tracked on GameState.active_modifiers and recalculated
    each cleanup cycle. This never mutates base card data.

    Auto-classifies the layer (rule 454) if not explicitly set on the node.
    """
    from .effect_ir import classify_modifier_layer

    stat = node.get("stat", "might")
    amount = node.get("amount", 0)

    # Ensure the node carries a layer classification (as a string value)
    if not node.get("layer"):
        layer_enum = classify_modifier_layer(node)
        node["layer"] = layer_enum.value if hasattr(layer_enum, "value") else str(layer_enum)

    # Find the ability_id that owns this IR (use source + a hash key)
    ability_id = f"{source.instance_id}_mod_{stat}_{amount}"

    # Don't double-register
    for existing in gs.active_modifiers:
        if existing.source_instance_id == source.instance_id and existing.ability_id == ability_id:
            return [f"{source.name} modifier already active"]

    gs.register_modifier(source, ability_id, node)

    # Stamp the newly registered modifier with a monotonic timestamp so
    # the layer system can order effects chronologically (rule 457).
    new_mod = gs.active_modifiers[-1]
    new_mod.timestamp = _next_timestamp()  # type: ignore[attr-defined]

    recalculate_modifiers_layered(gs)

    exclude_tag = " (excluding self)" if node.get("exclude_self") else ""
    sign = "+" if amount >= 0 else ""
    layer_tag = f" [layer: {node['layer']}]"
    return [f"{source.name}: {sign}{amount} {stat} aura active{exclude_tag}{layer_tag}"]


def _resolve_replacement(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
) -> list[str]:
    """Register a replacement effect on the game state.

    The replacement watches for a specific event/primitive and intercepts
    it with an alternative IR when conditions are met.
    """
    watch_event = node.get("watch_event", "")
    ability_id = f"{source.instance_id}_rep_{watch_event}"

    # Don't double-register
    for existing in gs.active_replacements:
        if existing.source_instance_id == source.instance_id and existing.ability_id == ability_id:
            return [f"{source.name} replacement already active"]

    gs.register_replacement(source, ability_id, node)
    return [f"{source.name}: replacement effect active (watches {watch_event})"]


def _resolve_delayed_trigger(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
) -> list[str]:
    """Register a delayed trigger on the game state (rules 382-385).

    Delayed abilities are created by spells/abilities and fire at a future
    time or event.  They persist independently of the creating source --
    even if the source leaves play, the delayed trigger still resolves
    (rule 385).

    The delayed trigger is stored on ``gs.delayed_triggers`` (a list).
    The trigger system checks this list when events fire.
    """
    trigger_event = node.get("trigger_event", "")
    effect_ir = node.get("effect_ir", {})
    duration = node.get("duration", "turn")
    max_fires = node.get("max_fires", 1)
    condition = node.get("condition", {"cond_type": "always"})

    if not hasattr(gs, "delayed_triggers"):
        gs.delayed_triggers = []  # type: ignore[attr-defined]

    delayed = {
        "trigger_event": trigger_event,
        "effect_ir": effect_ir,
        "condition": condition,
        "duration": duration,
        "max_fires": max_fires,
        "fires_remaining": max_fires if max_fires > 0 else -1,
        "source_instance_id": source.instance_id,
        "controller_id": source.controller_id,
        "created_turn": getattr(gs, "turn_number", 0),
    }
    gs.delayed_triggers.append(delayed)  # type: ignore[attr-defined]

    return [f"{source.name}: delayed trigger registered (watches {trigger_event}, "
            f"duration={duration})"]


def _resolve_triggered_effect(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
) -> list[str]:
    """Register a triggered effect (similar to delayed_trigger but event-based).

    A triggered_effect node contains a "trigger" sub-node describing the
    event that activates it, and an "effect" sub-node that fires when the
    trigger matches. This is used for cards like "Whenever you hide a
    card, ready this unit."

    We store this as a delayed trigger with permanent duration tied to
    the source remaining on the board.
    """
    trigger = node.get("trigger", {})
    effect = node.get("effect", {})

    trigger_event = trigger.get("event_type", trigger.get("type", "unknown"))
    scope = trigger.get("scope", "any")
    duration = node.get("duration", "permanent")

    if not hasattr(gs, "delayed_triggers"):
        gs.delayed_triggers = []  # type: ignore[attr-defined]

    delayed = {
        "trigger_event": trigger_event,
        "effect_ir": effect,
        "condition": {"cond_type": "always"},
        "duration": duration,
        "max_fires": 0,  # unlimited
        "fires_remaining": -1,
        "source_instance_id": source.instance_id,
        "controller_id": source.controller_id,
        "created_turn": getattr(gs, "turn_number", 0),
        "scope": scope,
    }
    gs.delayed_triggers.append(delayed)  # type: ignore[attr-defined]

    return [f"{source.name}: triggered effect registered (watches {trigger_event}, "
            f"scope={scope})"]


def _resolve_player_choice(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Handle a player_choice node — halts execution for player input.

    In auto-play mode (no pending_choice support), picks the first option.
    When the choice state machine is active (Task 3), this pushes
    CHOICE_PENDING and halts.
    """
    # If the game state supports choice halting, use it
    if hasattr(gs, "pending_choice") and gs.pending_choice is None:
        prompt = node.get("prompt", "Make a choice")
        options = node.get("options", [])
        target_spec = node.get("target")
        min_c = node.get("min_choices", 1)
        max_c = node.get("max_choices", 1)

        # Store the full context needed to resume later
        gs.pending_choice = {
            "prompt": prompt,
            "options": [opt if isinstance(opt, dict) else opt for opt in options],
            "target": target_spec,
            "min_choices": min_c,
            "max_choices": max_c,
            "source_instance_id": source.instance_id,
            "controller_id": source.controller_id,
            "remaining_ir": node,  # the full node for resume
            "pre_targets": targets,
        }
        return [f"CHOICE_PENDING: {prompt}"]

    # Auto-play fallback: pick first option or first valid target
    options = node.get("options", [])
    if options:
        first = options[0]
        effect = first.get("effect", first)
        return resolve_effect_ir(effect, source, gs, targets)

    # Target-selection choice: auto-resolve
    target_spec = node.get("target")
    if target_spec:
        resolved = _auto_resolve_targets(target_spec, source, gs)
        max_c = node.get("max_choices", 1)
        resolved = resolved[:max_c]
        if resolved:
            return [f"Auto-selected targets: {resolved}"]

    return ["No choice options available"]


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(
    condition: dict[str, Any],
    source: CardInstance,
    gs: GameState,
) -> bool:
    """Evaluate a ConditionSpec dict against the current game state."""
    cond_type = condition.get("cond_type", "")
    params = condition.get("params", {})

    if cond_type == "legion":
        # True if controller played another Main Deck card this turn
        ps = gs.players.get(source.controller_id)
        return bool(ps and ps.played_main_deck_this_turn)

    elif cond_type == "mighty":
        # True if source unit has Might >= 5
        return source.effective_might >= 5

    elif cond_type == "xp_gte":
        threshold = params.get("threshold", 0)
        ps = gs.players.get(source.controller_id)
        xp = getattr(ps, "xp", 0) if ps else 0
        return xp >= threshold

    elif cond_type == "has_keyword":
        from .enums import Keyword
        kw_str = params.get("keyword", "")
        try:
            kw = Keyword(kw_str)
            return source.has_keyword(kw)
        except ValueError:
            return False

    elif cond_type == "unit_count_gte":
        threshold = params.get("threshold", 0)
        scope = params.get("scope", "friendly")
        controller = source.controller_id
        count = 0
        for uid_list in gs.base_units.values():
            for uid in uid_list:
                unit = gs.get_instance(uid)
                if unit:
                    if scope == "friendly" and unit.controller_id == controller:
                        count += 1
                    elif scope == "enemy" and unit.controller_id != controller:
                        count += 1
                    elif scope == "any":
                        count += 1
        for bf in gs.battlefields.values():
            for uid in bf.units:
                unit = gs.get_instance(uid)
                if unit:
                    if scope == "friendly" and unit.controller_id == controller:
                        count += 1
                    elif scope == "enemy" and unit.controller_id != controller:
                        count += 1
                    elif scope == "any":
                        count += 1
        return count >= threshold

    elif cond_type == "controls_battlefield":
        from .enums import ControlStatus
        controller = source.controller_id
        for bf in gs.battlefields.values():
            if bf.control_status == ControlStatus.CONTROLLED and bf.controller_id == controller:
                return True
        return False

    elif cond_type == "has_buff":
        return source.buff_counter

    elif cond_type == "is_attacker":
        from .enums import CombatRole
        return source.combat_role == CombatRole.ATTACKER

    elif cond_type == "is_defender":
        from .enums import CombatRole
        return source.combat_role == CombatRole.DEFENDER

    elif cond_type == "card_played_this_turn":
        ps = gs.players.get(source.controller_id)
        return bool(ps and ps.played_main_deck_this_turn)

    elif cond_type == "previous_effect_succeeded":
        return gs.last_effect_succeeded

    elif cond_type == "always":
        return True

    elif cond_type == "never":
        return False

    elif cond_type == "hand_count_lte":
        threshold = params.get("threshold", 0)
        ps = gs.players.get(source.controller_id)
        return bool(ps and len(ps.hand) <= threshold)

    elif cond_type == "hand_count_gte":
        # Rule 361.3a: conditional passive — "while you have N or more cards
        # in hand"
        threshold = params.get("threshold", 0)
        ps = gs.players.get(source.controller_id)
        return bool(ps and len(ps.hand) >= threshold)

    elif cond_type == "opponent_hand_count_gte":
        # Rule 361: "while you have 2 or more cards in your hand"
        # (from opponent's perspective, e.g., "+1[M] while opponent has 2+
        # cards in hand")
        threshold = params.get("threshold", 0)
        opp_id = gs.opponent_id(source.controller_id)
        opp = gs.players.get(opp_id)
        return bool(opp and len(opp.hand) >= threshold)

    elif cond_type == "is_alone":
        # "While I'm attacking or defending alone" — only one unit from
        # the controller at the same battlefield
        bf = gs.get_battlefield_for_unit(source.instance_id)
        if not bf:
            return False
        own_units = [
            uid for uid in bf.units
            if gs.get_instance(uid) and gs.get_instance(uid).controller_id == source.controller_id
        ]
        return len(own_units) == 1

    elif cond_type == "source_at_battlefield":
        # True if the source card is at a battlefield (not in base)
        from .enums import ZoneType
        return source.zone == ZoneType.BATTLEFIELD

    else:
        logger.warning("Unknown condition type: %s", cond_type)
        return False


# ---------------------------------------------------------------------------
# Auto-target resolution (used when no player-chosen targets available)
# ---------------------------------------------------------------------------

def _auto_resolve_targets(
    target_spec: dict[str, Any],
    source: CardInstance,
    gs: GameState,
) -> list[str]:
    """Find matching targets from the game state based on a TargetSpec dict.

    This is a simplified auto-resolver used when the effect system needs
    targets but none were pre-chosen. The full target_system.py (Phase 3)
    will replace this with comprehensive target resolution.
    """
    from .enums import CardType, CombatRole, Keyword, ZoneType

    spec = TargetSpec.from_dict(target_spec) if isinstance(target_spec, dict) else target_spec
    controller = source.controller_id
    candidates: list[str] = []

    # Collect candidates based on obj_type and zone
    if spec.obj_type in ("unit", "permanent", "card"):
        # Units in bases
        for pid, uid_list in gs.base_units.items():
            for uid in uid_list:
                candidates.append(uid)
        # Units at battlefields
        for bf in gs.battlefields.values():
            for uid in bf.units:
                candidates.append(uid)

    if spec.obj_type in ("gear", "permanent", "card"):
        for pid, gid_list in gs.base_gear.items():
            for gid in gid_list:
                candidates.append(gid)

    if spec.obj_type == "spell" and spec.zone == "chain":
        for item in gs.chain.stack:
            if item.card_instance_id:
                card = gs.get_instance(item.card_instance_id)
                if card and card.card_type == CardType.SPELL:
                    candidates.append(item.card_instance_id)

    if spec.obj_type == "rune":
        for pid, rid_list in gs.base_runes.items():
            for rid in rid_list:
                candidates.append(rid)

    # Filter by scope
    filtered: list[str] = []
    for cid in candidates:
        card = gs.get_instance(cid)
        if not card:
            continue

        if spec.scope == "self" and cid != source.instance_id:
            continue
        if spec.scope == "friendly" and card.controller_id != controller:
            continue
        if spec.scope == "enemy" and card.controller_id == controller:
            continue

        # Filter by zone
        if spec.zone == "base" and card.zone != ZoneType.BASE:
            continue
        if spec.zone == "battlefield" and card.zone != ZoneType.BATTLEFIELD:
            continue

        # Filter by location
        if spec.location == "here" and source.location_id:
            if card.location_id != source.location_id:
                continue

        # Apply additional filters
        passes_filters = True
        for f in spec.filters:
            if not _check_filter(card, f):
                passes_filters = False
                break
        if not passes_filters:
            continue

        filtered.append(cid)

    # Apply count
    count = spec.count
    if count == -1:
        return filtered  # all
    return filtered[:count]


def _check_filter(card: CardInstance, f_dict: Any) -> bool:
    """Check if a card passes a single filter."""
    from .effect_ir import FilterSpec
    from .enums import Keyword

    filt = FilterSpec.from_dict(f_dict) if isinstance(f_dict, dict) else f_dict

    if filt.field == "keyword":
        try:
            kw = Keyword(filt.value)
        except ValueError:
            return False
        has = card.has_keyword(kw)
        if filt.op == "has":
            return has
        elif filt.op == "not_has":
            return not has

    elif filt.field == "might":
        m = card.effective_might
        if filt.op == "lte":
            return m <= filt.value
        elif filt.op == "gte":
            return m >= filt.value
        elif filt.op == "eq":
            return m == filt.value

    elif filt.field == "tag":
        tags = card.definition.tags
        if filt.op == "has":
            return filt.value in tags
        elif filt.op == "not_has":
            return filt.value not in tags

    elif filt.field == "card_type":
        if filt.op == "eq":
            return card.card_type.value == filt.value

    elif filt.field == "is_exhausted":
        if filt.op == "eq":
            return card.exhausted == filt.value

    return True  # unknown filter passes by default


# ---------------------------------------------------------------------------
# Layer-aware modifier recalculation (rules 453-457)
# ---------------------------------------------------------------------------

_LAYER_ORDER = ("trait_altering", "ability_altering", "arithmetic")
_MAX_LAYER_ITERATIONS = 5  # safety cap for the re-evaluation loop


def recalculate_modifiers_layered(gs: GameState) -> None:
    """Recompute all continuous modifiers using the 3-layer system (rule 454).

    Layers are applied in order:
      1. Trait-Altering  (454.1) -- name/type/tag/controller/domain/might-set
      2. Ability-Altering (454.2) -- keywords / abilities / rules text
      3. Arithmetic       (454.3) -- might increases THEN decreases

    Within each layer, modifiers without dependency are applied in timestamp
    order (rule 457). The whole sequence is re-evaluated until no changes
    occur (rule 453.2), capped at ``_MAX_LAYER_ITERATIONS`` to prevent loops.

    Within the Arithmetic layer (rule 454.3.d):
      - Increases (positive amounts) are applied first  (454.3.d.1)
      - Decreases (negative amounts) are applied second (454.3.d.2)

    Arithmetic effects with a ``min_might`` or ``max_might`` limitation are
    *snapshotted* at application time (rule 454.3.b): the effective delta is
    clamped when first computed, and that clamped value is remembered for the
    duration of the effect.

    Keyword/ability modifiers (ability-altering layer) grant or remove
    keywords on matching targets while their source is on the board.
    """
    from .effect_ir import EffectLayer, TargetSpec, classify_modifier_layer

    # 1. Zero all derived bonuses and remove aura-granted keywords.
    #    Aura-granted keywords are tracked as a count-per-keyword via the
    #    ``_aura_kw_counts`` dict stamped on each CardInstance during the
    #    previous pass.  We remove exactly that many entries per keyword,
    #    leaving player-granted duplicates intact.
    for inst in gs.instances.values():
        inst.aura_might_bonus = 0
        inst.gear_might_bonus = 0
        prev_aura_counts: dict[str, int] = getattr(inst, "_aura_kw_counts", {})
        if prev_aura_counts:
            remaining_counts = dict(prev_aura_counts)
            new_list: list = []
            for gk in inst.granted_keywords:
                kw_val = gk if isinstance(gk, str) else getattr(gk, "keyword", gk)
                if isinstance(kw_val, str) and remaining_counts.get(kw_val, 0) > 0:
                    remaining_counts[kw_val] -= 1
                    # Skip this entry (it was aura-granted)
                else:
                    new_list.append(gk)
            inst.granted_keywords = new_list
        inst._aura_kw_counts = {}  # type: ignore[attr-defined]

    # 2. Prune stale modifiers/replacements whose source left the board
    board_ids = gs._get_board_instance_ids()
    gs.active_modifiers = [
        m for m in gs.active_modifiers if m.source_instance_id in board_ids
    ]
    gs.active_replacements = [
        r for r in gs.active_replacements if r.source_instance_id in board_ids
    ]

    # 3. Classify EVERY modifier into its layer (not just "might").
    layer_buckets: dict[str, list] = {layer: [] for layer in _LAYER_ORDER}
    for mod in gs.active_modifiers:
        layer_key = classify_modifier_layer(
            {"stat": mod.stat, "amount": mod.amount}
        )
        if isinstance(layer_key, EffectLayer):
            layer_key = layer_key.value
        if layer_key not in layer_buckets:
            layer_key = EffectLayer.ARITHMETIC.value
        layer_buckets[layer_key].append(mod)

    # 4. Sort within each layer by timestamp (rule 457)
    for layer_key in _LAYER_ORDER:
        bucket = layer_buckets[layer_key]
        bucket.sort(key=lambda m: getattr(m, "timestamp", 0))

    # 5. Iterative re-evaluation loop (rule 453.2)
    applied_ids: set[tuple[str, str]] = set()  # (source_id, ability_id)

    for _iteration in range(_MAX_LAYER_ITERATIONS):
        changed = False

        for layer_key in _LAYER_ORDER:
            mods = layer_buckets[layer_key]
            if not mods:
                continue

            if layer_key == EffectLayer.ARITHMETIC.value:
                # Rule 454.3.d: increases first, then decreases
                increases = [m for m in mods if m.amount >= 0]
                decreases = [m for m in mods if m.amount < 0]
                ordered = increases + decreases
            else:
                ordered = mods

            for mod in ordered:
                mod_key = (mod.source_instance_id, mod.ability_id)
                if mod_key in applied_ids:
                    continue

                source = gs.get_instance(mod.source_instance_id)
                if not source:
                    continue

                spec = TargetSpec.from_dict(mod.target_spec)
                matched = gs._resolve_modifier_targets(spec, source, mod.exclude_self)
                if not matched:
                    continue

                if mod.stat == "keyword":
                    # Ability-Altering layer: grant keyword to matched targets
                    _apply_keyword_modifier(gs, mod, matched)
                elif mod.stat in ("might", "energy_cost", "power_cost"):
                    # Arithmetic layer: apply might/cost adjustments
                    _apply_arithmetic_modifier(gs, mod, matched)
                # Trait-Altering modifiers (name/type/tag/controller/domain)
                # are not yet implemented beyond might_set, which goes through
                # the arithmetic path with a ``set_value`` flag.

                applied_ids.add(mod_key)
                changed = True

        if not changed:
            break

    # 6. Apply gear attachment might bonuses (always arithmetic layer)
    for inst in gs.instances.values():
        if inst.attached_cards:
            for gear_id in inst.attached_cards:
                gear = gs.get_instance(gear_id)
                if gear and gear.definition.might_bonus:
                    inst.gear_might_bonus += gear.definition.might_bonus


def _apply_arithmetic_modifier(
    gs: GameState,
    mod: Any,
    matched: list[str],
) -> None:
    """Apply a single arithmetic modifier to matched targets (rule 454.3).

    Handles snapshotting (rule 454.3.b): when the modifier carries a
    ``min_might`` field, the effective delta is clamped so the target's
    might does not fall below the minimum.
    """
    for iid in matched:
        inst = gs.get_instance(iid)
        if not inst:
            continue
        effective_delta = mod.amount

        # Snapshotting (rule 454.3.b): clamp to floor / ceiling when
        # the modifier specifies a min or max constraint.
        min_might = getattr(mod, "min_might", None)
        if min_might is not None and effective_delta < 0:
            # How far can might drop before hitting the min?
            current = inst.definition.base_might + inst.aura_might_bonus
            headroom = current - min_might
            if headroom <= 0:
                effective_delta = 0
            else:
                effective_delta = max(effective_delta, -headroom)

        inst.aura_might_bonus += effective_delta


def _apply_keyword_modifier(
    gs: GameState,
    mod: Any,
    matched: list[str],
) -> None:
    """Apply an ability-altering modifier that grants a keyword (rule 454.2).

    Uses the same string convention as ``prim_give_keyword``: appends the
    keyword's string value to ``granted_keywords``.  The keyword string is
    also counted in the ``_aura_kw_counts`` dict on the CardInstance so that
    ``recalculate_modifiers_layered`` can strip exactly those entries on
    the next pass without removing player-granted keywords.
    """
    from .enums import Keyword

    keyword_str = getattr(mod, "keyword_value", None) or str(mod.amount)
    try:
        kw = Keyword(keyword_str)
    except (ValueError, KeyError):
        return

    for iid in matched:
        inst = gs.get_instance(iid)
        if not inst:
            continue
        aura_counts: dict[str, int] = getattr(inst, "_aura_kw_counts", {})
        # Always append -- the count tracks how many to strip later
        inst.granted_keywords.append(kw.value)
        aura_counts[kw.value] = aura_counts.get(kw.value, 0) + 1
        inst._aura_kw_counts = aura_counts  # type: ignore[attr-defined]
