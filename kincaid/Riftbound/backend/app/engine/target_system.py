"""Unified target resolution and validation system.

Replaces the ad-hoc _is_valid_target / _count_available_targets in action_validator.
Provides centralized target matching for both human-chosen and auto-resolved targets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .enums import CardType, CombatRole, ControlStatus, Keyword, ZoneType

if TYPE_CHECKING:
    from .card_types import CardInstance
    from .game_state import GameState

logger = logging.getLogger("riftbound.targets")


def resolve_targets(
    gs: GameState,
    target_spec: dict[str, Any],
    source: CardInstance | None = None,
    controller_id: str = "",
) -> list[str]:
    """Find all valid target instance_ids matching a TargetSpec dict.

    Args:
        gs: Current game state.
        target_spec: Dict with keys: obj_type, scope, zone, location,
                     count, filters, chooser.
        source: The card/ability producing this effect (for scope/location).
        controller_id: Fallback controller for scope resolution.

    Returns:
        List of matching instance_ids, limited by count.
    """
    if not target_spec:
        return []

    ctrl = controller_id or (source.controller_id if source else "")
    candidates = _gather_candidates(gs, target_spec)
    filtered = _apply_filters(gs, candidates, target_spec, source, ctrl)

    count = target_spec.get("count", 1)
    if count == -1:
        return filtered
    return filtered[:count]


def validate_target(
    gs: GameState,
    target_id: str,
    target_spec: dict[str, Any],
    source: CardInstance | None = None,
    controller_id: str = "",
) -> bool:
    """Check if a specific target_id is valid for the given spec."""
    valid = resolve_targets(gs, target_spec, source, controller_id)
    return target_id in valid


def count_available_targets(
    gs: GameState,
    target_spec: dict[str, Any],
    source: CardInstance | None = None,
    controller_id: str = "",
) -> int:
    """Count how many valid targets exist for a spec (ignoring count limit)."""
    spec_all = dict(target_spec)
    spec_all["count"] = -1
    return len(resolve_targets(gs, spec_all, source, controller_id))


# ---------------------------------------------------------------------------
# Candidate gathering — collect potential targets by obj_type and zone
# ---------------------------------------------------------------------------

def _gather_candidates(gs: GameState, spec: dict[str, Any]) -> list[str]:
    """Collect all candidate instance_ids based on obj_type and zone."""
    obj_type = spec.get("obj_type", "unit")
    zone = spec.get("zone", "any")
    candidates: list[str] = []

    if obj_type in ("unit", "permanent", "card"):
        if zone in ("base", "any"):
            for uid_list in gs.base_units.values():
                candidates.extend(uid_list)
        if zone in ("battlefield", "any"):
            for bf in gs.battlefields.values():
                candidates.extend(bf.units)
        if zone == "hand":
            for ps in gs.players.values():
                candidates.extend(ps.hand)
        if zone == "trash":
            for ps in gs.players.values():
                candidates.extend(ps.trash)

    if obj_type in ("gear", "permanent", "card"):
        if zone in ("base", "any"):
            for gid_list in gs.base_gear.values():
                candidates.extend(gid_list)

    if obj_type == "spell":
        if zone in ("chain", "any"):
            for item in gs.chain.stack:
                if item.card_instance_id:
                    card = gs.get_instance(item.card_instance_id)
                    if card and card.card_type == CardType.SPELL:
                        candidates.append(item.card_instance_id)

    if obj_type == "rune":
        if zone in ("base", "rune_board", "any"):
            for rid_list in gs.base_runes.values():
                candidates.extend(rid_list)

    return candidates


# ---------------------------------------------------------------------------
# Filtering — apply scope, location, and custom filters
# ---------------------------------------------------------------------------

def _apply_filters(
    gs: GameState,
    candidates: list[str],
    spec: dict[str, Any],
    source: CardInstance | None,
    controller_id: str,
) -> list[str]:
    """Filter candidates by scope, location, and additional filters."""
    scope = spec.get("scope", "any")
    location = spec.get("location", "any")
    filters = spec.get("filters", [])

    result: list[str] = []

    for cid in candidates:
        card = gs.get_instance(cid)
        if not card:
            continue

        # Scope filter
        if scope == "self":
            if not source or cid != source.instance_id:
                continue
        elif scope == "friendly":
            if card.controller_id != controller_id:
                continue
        elif scope == "enemy":
            if card.controller_id == controller_id:
                continue

        # Location filter
        if location == "here" and source:
            if card.location_id != source.location_id:
                continue
        elif location not in ("any", "here") and location:
            if card.location_id != location:
                continue

        # Custom filters
        if not _passes_all_filters(card, filters):
            continue

        result.append(cid)

    return result


def _passes_all_filters(card: CardInstance, filters: list[dict]) -> bool:
    """Check if a card passes all filter predicates."""
    for f in filters:
        if not _check_single_filter(card, f):
            return False
    return True


def _check_single_filter(card: CardInstance, filt: dict) -> bool:
    """Evaluate a single FilterSpec dict against a card."""
    field = filt.get("field", "")
    op = filt.get("op", "eq")
    value = filt.get("value")

    if field == "keyword":
        try:
            kw = Keyword(value)
        except ValueError:
            return False
        has = card.has_keyword(kw)
        return has if op == "has" else not has

    elif field == "might":
        m = card.effective_might
        if op == "lte":
            return m <= value
        elif op == "gte":
            return m >= value
        elif op == "eq":
            return m == value

    elif field == "tag":
        tags = card.definition.tags
        if op == "has":
            return value in tags
        elif op == "not_has":
            return value not in tags

    elif field == "card_type":
        return (card.card_type.value == value) if op == "eq" else (card.card_type.value != value)

    elif field == "domain":
        domains = [d.value for d in card.definition.domains]
        if op == "has":
            return value in domains
        elif op == "not_has":
            return value not in domains

    elif field == "is_exhausted":
        return card.exhausted == value

    elif field == "name":
        if op == "eq":
            return card.name == value
        elif op == "neq":
            return card.name != value

    elif field == "controller":
        if op == "eq":
            return card.controller_id == value
        elif op == "neq":
            return card.controller_id != value

    elif field == "combat_role":
        if op == "eq":
            return card.combat_role.value == value

    return True  # Unknown filter passes by default
