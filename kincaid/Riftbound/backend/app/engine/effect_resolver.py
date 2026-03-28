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
    COMPOSITION_TYPES,
    CONDITIONAL,
    CHOOSE_ONE,
    FOR_EACH,
    OPTIONAL,
    PRIMITIVE_TYPES,
    REPEAT_EFFECT,
    SEQUENCE,
    ConditionSpec,
    TargetSpec,
)
from .effect_primitives import PRIMITIVE_DISPATCH

if TYPE_CHECKING:
    from .card_types import CardInstance
    from .game_state import GameState

logger = logging.getLogger("riftbound.resolver")


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
    else:
        return [f"Unhandled node type: {node_type}"]


def _resolve_primitive(
    node: dict[str, Any],
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """Execute a leaf (primitive) node."""
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

    return fn(source, gs, node, resolved)


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

    elif cond_type == "always":
        return True

    elif cond_type == "never":
        return False

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
