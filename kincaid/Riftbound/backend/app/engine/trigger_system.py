"""Event-driven trigger system for card abilities.

Scans board objects for abilities matching game events and pushes triggered
abilities onto the chain. Replaces the ad-hoc trigger handling scattered
across chain.py, cleanup.py, combat.py, and scoring.py.

Usage:
    from .trigger_system import fire_event, GameEvent
    fire_event(gs, GameEvent.UNIT_ENTERED, {"card_id": unit.instance_id})
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from .enums import AbilityType, CardType, ZoneType
from .game_state import ChainItem

if TYPE_CHECKING:
    from .card_types import CardInstance
    from .game_state import GameState

logger = logging.getLogger("riftbound.triggers")


class GameEvent(str, Enum):
    """Events that can trigger card abilities."""

    UNIT_ENTERED = "unit_entered"          # A unit entered the board
    SPELL_PLAYED = "spell_played"          # A spell was played
    UNIT_DIED = "unit_died"                # A unit died
    UNIT_MOVED = "unit_moved"              # A unit moved between zones
    UNIT_MOVED_TO_BF = "unit_moved_to_bf"  # A unit moved to a battlefield
    CONQUER = "conquer"                    # A battlefield was conquered
    HOLD = "hold"                          # A battlefield was held
    CONQUER_OR_HOLD = "conquer_or_hold"    # Either conquer or hold
    ATTACK_STARTED = "attack_started"      # A unit started attacking
    DEFEND_STARTED = "defend_started"      # A unit started defending
    ATTACK_OR_DEFEND = "attack_or_defend"  # Either attack or defend
    EQUIP = "equip"                        # A gear was equipped
    DAMAGE_DEALT = "damage_dealt"          # Damage was dealt
    TURN_START = "turn_start"              # Start of turn
    TURN_END = "turn_end"                  # End of turn
    BEGINNING_PHASE = "beginning_phase"    # Beginning phase
    CARD_DRAWN = "card_drawn"              # A card was drawn
    FRIENDLY_DEATH = "friendly_death"      # A friendly unit died
    ENEMY_DEATH = "enemy_death"            # An enemy unit died
    SPELL_PLAYED_HERE = "spell_played_here"  # A spell targeting this BF
    RECYCLE_RUNE = "recycle_rune"          # A rune was recycled
    COMBAT_WIN = "combat_win"              # Player won a combat
    UNIT_PLAYED = "unit_played"            # A unit was played (synonym)
    GEAR_ENTERED = "gear_entered"          # A gear entered the board


# Map from trigger_condition strings on AbilityDefinition to GameEvents
TRIGGER_TO_EVENTS: dict[str, list[GameEvent]] = {
    "on_play": [GameEvent.UNIT_ENTERED, GameEvent.UNIT_PLAYED, GameEvent.GEAR_ENTERED],
    "on_conquer": [GameEvent.CONQUER, GameEvent.CONQUER_OR_HOLD],
    "on_hold": [GameEvent.HOLD, GameEvent.CONQUER_OR_HOLD],
    "on_conquer_or_hold": [GameEvent.CONQUER, GameEvent.HOLD, GameEvent.CONQUER_OR_HOLD],
    "on_death": [GameEvent.UNIT_DIED],
    "on_attack": [GameEvent.ATTACK_STARTED, GameEvent.ATTACK_OR_DEFEND],
    "on_defend": [GameEvent.DEFEND_STARTED, GameEvent.ATTACK_OR_DEFEND],
    "on_attack_or_defend": [GameEvent.ATTACK_STARTED, GameEvent.DEFEND_STARTED, GameEvent.ATTACK_OR_DEFEND],
    "on_move": [GameEvent.UNIT_MOVED, GameEvent.UNIT_MOVED_TO_BF],
    "on_move_to_bf": [GameEvent.UNIT_MOVED_TO_BF],
    "on_friendly_death": [GameEvent.FRIENDLY_DEATH],
    "on_enemy_death": [GameEvent.ENEMY_DEATH],
    "on_spell_played": [GameEvent.SPELL_PLAYED],
    "on_any_spell_played": [GameEvent.SPELL_PLAYED],
    "on_unit_played": [GameEvent.UNIT_PLAYED],
    "on_equip": [GameEvent.EQUIP],
    "on_turn_start": [GameEvent.TURN_START],
    "on_turn_end": [GameEvent.TURN_END],
    "on_recycle_rune": [GameEvent.RECYCLE_RUNE],
    "on_combat_win": [GameEvent.COMBAT_WIN],
    "on_card_drawn": [GameEvent.CARD_DRAWN],
    "on_beginning_phase": [GameEvent.BEGINNING_PHASE],
    "on_damage_dealt": [GameEvent.DAMAGE_DEALT],
}

# Reverse: which trigger_conditions respond to each event
EVENT_TO_TRIGGERS: dict[GameEvent, set[str]] = {}
for trigger, events in TRIGGER_TO_EVENTS.items():
    for event in events:
        EVENT_TO_TRIGGERS.setdefault(event, set()).add(trigger)


def fire_event(
    gs: GameState,
    event: GameEvent,
    context: dict[str, Any] | None = None,
) -> list[str]:
    """Fire a game event, scanning all board objects for matching triggers.

    Matching triggered abilities are pushed onto the chain as Pending items.

    Ordering (rules 376.3.b / 376.3.b.1):
      1. Turn Player's triggers come first, then other players in turn order.
      2. Within each player's group, the triggers stay in the order they were
         collected (FIFO -- the order the cards were scanned).  In a real game
         the controlling player selects the order; FIFO is the auto-play
         approximation.
      3. Python's sort is *stable*, so items that compare equal keep their
         original relative order (FIFO within a player group).

    Nth-time deduplication (rule 376.1.b):
      If an ability has ``max_triggers_per_turn`` set (from "the first time"
      or "the Nth time" templating), it fires at most that many times per
      turn even if the condition is met simultaneously by multiple events.
      Tracking is per (card_instance_id, ability_id) per turn number.

    Args:
        gs: Current game state.
        event: The event that occurred.
        context: Event-specific data:
            - card_id: instance_id of the card involved
            - player_id: player who caused the event
            - battlefield_id: related battlefield
            - source_id: source of the event

    Returns:
        Log messages for any triggers that fired.
    """
    if context is None:
        context = {}

    matching_triggers = EVENT_TO_TRIGGERS.get(event, set())
    if not matching_triggers:
        return []

    logs: list[str] = []
    triggered_items: list[tuple[str, str, str, CardInstance]] = []
    # (controller_id, ability_id, card_instance_id, source_card)

    # Lazily initialise the per-turn trigger count tracker (rule 376.1.b).
    if not hasattr(gs, "_trigger_fire_counts"):
        gs._trigger_fire_counts = {}  # type: ignore[attr-defined]
    # Reset tracker when a new turn starts (keyed by turn_number)
    trigger_counts: dict[tuple[str, str], int] = gs._trigger_fire_counts  # type: ignore[attr-defined]
    # Purge stale entries from previous turns
    current_turn = getattr(gs, "turn_number", 0)
    if not hasattr(gs, "_trigger_counts_turn"):
        gs._trigger_counts_turn = current_turn  # type: ignore[attr-defined]
    if gs._trigger_counts_turn != current_turn:  # type: ignore[attr-defined]
        trigger_counts.clear()
        gs._trigger_counts_turn = current_turn  # type: ignore[attr-defined]

    # Scan all board objects
    for card in _get_board_objects(gs):
        for ability in card.definition.abilities:
            if ability.ability_type not in (AbilityType.TRIGGERED, "triggered"):
                continue
            if ability.trigger_condition not in matching_triggers:
                continue
            if not ability.effect_ir:
                continue

            # Check if this trigger applies to this specific event context
            if not _trigger_applies(card, ability, event, context, gs):
                continue

            # Rule 376.1.b — Nth-time deduplication.
            # ``max_triggers_per_turn`` is an optional int on AbilityDefinition
            # (set from card data for "the first time" / "the Nth time"
            # templating).  Default 0 means unlimited.
            max_per_turn: int = getattr(ability, "max_triggers_per_turn", 0)
            if max_per_turn > 0:
                trig_key = (card.instance_id, ability.ability_id)
                already_fired = trigger_counts.get(trig_key, 0)
                if already_fired >= max_per_turn:
                    continue
                trigger_counts[trig_key] = already_fired + 1

            triggered_items.append((
                card.controller_id,
                ability.ability_id,
                card.instance_id,
                card,
            ))

    if not triggered_items:
        return []

    # Order per rules 376.3.b / 376.3.b.1
    turn_player = gs.turn_player_id
    player_order = gs.player_order

    def sort_key(item: tuple) -> tuple:
        ctrl = item[0]
        if ctrl == turn_player:
            return (0,)
        try:
            idx = player_order.index(ctrl)
        except ValueError:
            idx = 999
        return (1, idx)

    triggered_items.sort(key=sort_key)

    # Push each triggered ability onto the chain
    for ctrl_id, ab_id, card_iid, source in triggered_items:
        chain_item = ChainItem.create(
            controller_id=ctrl_id,
            source_instance_id=card_iid,
            ability_id=ab_id,
        )
        gs.chain.push(chain_item)
        logs.append(f"{source.name} triggers: {ab_id}")
        logger.info("[TRIGGER] %s fires %s (event: %s)", source.name, ab_id, event.value)

    # Also check delayed triggers (rules 382-385).
    # Delayed triggers are source-independent: they fire even if the card
    # that created them has left the board.
    delayed_logs = _check_delayed_triggers(gs, event, context)
    logs.extend(delayed_logs)

    return logs


def _check_delayed_triggers(
    gs: GameState,
    event: GameEvent,
    context: dict[str, Any],
) -> list[str]:
    """Check and fire any delayed triggers matching the current event (rules 382-385).

    Delayed triggers persist independently of their creating source and
    fire when their event/condition is matched. After firing, they
    decrement ``fires_remaining``; once exhausted or expired, they are
    removed.

    Returns log messages for any delayed triggers that fired.
    """
    from .effect_resolver import evaluate_condition, resolve_effect_ir

    if not hasattr(gs, "delayed_triggers"):
        return []

    delayed_list: list[dict] = gs.delayed_triggers  # type: ignore[attr-defined]
    if not delayed_list:
        return []

    logs: list[str] = []
    to_remove: list[int] = []
    current_turn = getattr(gs, "turn_number", 0)

    for idx, dt in enumerate(delayed_list):
        # Check event match
        if dt["trigger_event"] != event.value:
            continue

        # Check duration expiry
        if dt["duration"] == "turn" and dt["created_turn"] != current_turn:
            to_remove.append(idx)
            continue

        # Evaluate condition
        source = gs.get_instance(dt["source_instance_id"])
        # Rule 385: delayed triggers fire even if source is gone, but we
        # still need a source for condition evaluation.  Create a minimal
        # fallback if the source has been removed from instances.
        if not source:
            # Source is gone -- condition that depends on source state
            # cannot be evaluated; fire unconditionally if condition is "always".
            cond = dt.get("condition", {"cond_type": "always"})
            if cond.get("cond_type") != "always":
                to_remove.append(idx)
                continue
        else:
            cond = dt.get("condition", {"cond_type": "always"})
            if not evaluate_condition(cond, source, gs):
                continue

        # Fire the delayed trigger's effect
        effect_ir = dt.get("effect_ir", {})
        if effect_ir:
            # Use the original source if still around, otherwise create a
            # synthetic ChainItem for execution context.
            if source:
                result_logs = resolve_effect_ir(effect_ir, source, gs)
                logs.extend(result_logs)
            else:
                logs.append(f"Delayed trigger fired (source gone): {dt['trigger_event']}")

        logger.info(
            "[DELAYED_TRIGGER] fires for event %s (controller: %s)",
            event.value, dt.get("controller_id", "?"),
        )

        # Decrement fires remaining
        remaining = dt.get("fires_remaining", -1)
        if remaining > 0:
            dt["fires_remaining"] = remaining - 1
            if dt["fires_remaining"] <= 0:
                to_remove.append(idx)
        elif remaining == 0:
            # Already exhausted
            to_remove.append(idx)

    # Remove expired/exhausted delayed triggers (iterate in reverse to keep indices stable)
    for idx in sorted(set(to_remove), reverse=True):
        if idx < len(delayed_list):
            delayed_list.pop(idx)

    return logs


def _get_board_objects(gs: GameState) -> list[CardInstance]:
    """Get all CardInstances currently on the board."""
    objects: list[CardInstance] = []

    # Units in bases
    for uid_list in gs.base_units.values():
        for uid in uid_list:
            card = gs.get_instance(uid)
            if card:
                objects.append(card)

    # Units at battlefields
    for bf in gs.battlefields.values():
        for uid in bf.units:
            card = gs.get_instance(uid)
            if card:
                objects.append(card)

    # Gear in bases
    for gid_list in gs.base_gear.values():
        for gid in gid_list:
            card = gs.get_instance(gid)
            if card:
                objects.append(card)

    # Runes on board
    for rid_list in gs.base_runes.values():
        for rid in rid_list:
            card = gs.get_instance(rid)
            if card:
                objects.append(card)

    # Legends
    for ps in gs.players.values():
        if ps.legend_zone:
            card = gs.get_instance(ps.legend_zone)
            if card:
                objects.append(card)

    # Battlefield cards themselves
    for bf in gs.battlefields.values():
        card = gs.get_instance(bf.card_instance_id)
        if card:
            objects.append(card)

    return objects


def _trigger_applies(
    card: CardInstance,
    ability: Any,
    event: GameEvent,
    context: dict[str, Any],
    gs: GameState,
) -> bool:
    """Check if a trigger on this specific card should fire for this event.

    Context-dependent rules:
    - "When you play me" only fires for the card itself entering
    - "When I conquer" only fires for the unit that conquered
    - "When I attack/defend" only fires for attacking/defending units
    - "When a friendly unit dies" fires for controller's units
    - Death triggers fire for the dying unit itself
    """
    trigger = ability.trigger_condition
    ctx_card = context.get("card_id", "")
    ctx_player = context.get("player_id", "")
    ctx_bf = context.get("battlefield_id", "")

    if trigger == "on_play":
        # "When you play me" - only fires for this specific card
        return ctx_card == card.instance_id

    if trigger == "on_death":
        # "When I die" - only fires for the dying card
        return ctx_card == card.instance_id

    if trigger in ("on_conquer", "on_hold", "on_conquer_or_hold"):
        ctx_bf = context.get("battlefield_id", "")
        # "When I conquer/hold" - card must be at the battlefield
        if ctx_card and ctx_card == card.instance_id:
            return True
        # Card's controller must be the scorer AND card must be at the battlefield
        # (units present during conquest/hold trigger, plus battlefield cards themselves)
        if ctx_player == card.controller_id and ctx_bf:
            if card.location_id == ctx_bf:
                return True
            # Battlefield card objects trigger for their own location
            if card.card_type == CardType.BATTLEFIELD and card.instance_id:
                bf_state = gs.battlefields.get(ctx_bf)
                if bf_state and bf_state.card_instance_id == card.instance_id:
                    return True
        return False

    if trigger in ("on_attack", "on_defend", "on_attack_or_defend"):
        # "When I attack/defend" - must be this specific unit
        return ctx_card == card.instance_id

    if trigger == "on_friendly_death":
        # "When a friendly unit dies" - dying unit must be friendly to card
        if ctx_card:
            dying = gs.get_instance(ctx_card)
            if dying and dying.controller_id == card.controller_id:
                # Don't trigger for yourself
                return ctx_card != card.instance_id
        return False

    if trigger == "on_enemy_death":
        # "When an enemy unit dies" - dying unit must be enemy to card
        if ctx_card:
            dying = gs.get_instance(ctx_card)
            if dying and dying.controller_id != card.controller_id:
                return True
        return False

    if trigger in ("on_spell_played", "on_any_spell_played"):
        # "When you play a spell" - controller must match
        if trigger == "on_any_spell_played":
            return True  # Any player
        return ctx_player == card.controller_id

    if trigger == "on_unit_played":
        # "When you play a unit"
        return ctx_player == card.controller_id

    if trigger == "on_move" or trigger == "on_move_to_bf":
        return ctx_card == card.instance_id

    if trigger == "on_equip":
        return ctx_card == card.instance_id

    if trigger == "on_turn_start" or trigger == "on_turn_end":
        # "At end of your turn" - card's controller must be turn player
        return card.controller_id == gs.turn_player_id

    if trigger == "on_recycle_rune":
        return ctx_player == card.controller_id

    if trigger == "on_combat_win":
        return ctx_player == card.controller_id

    if trigger == "on_card_drawn":
        # "When you draw a card" - controller must match
        return ctx_player == card.controller_id

    if trigger == "on_beginning_phase":
        # "At the beginning of your turn" - card's controller must be turn player
        return card.controller_id == gs.turn_player_id

    if trigger == "on_damage_dealt":
        # "When damage is dealt" - default: fires for controller's effects
        return ctx_player == card.controller_id

    # Default: trigger fires
    return True


def check_replacements(
    gs: GameState,
    event: GameEvent,
    context: dict[str, Any],
) -> bool:
    """Check if any replacement effect intercepts this event.

    Returns True if the event was replaced (original should not execute).
    Scans GameState.active_replacements for matching watch_events,
    evaluates conditions, and executes the alternative IR if matched.

    Per rule 368: when multiple replacements apply to the same event, the
    owner of the object being acted on chooses the order.  We approximate
    this by sorting eligible replacements so that those owned by the
    acted-on object's controller come first, then by the current turn
    player (rule 368.2 for uncontrolled battlefields).
    """
    from .effect_ir import TargetSpec
    from .effect_resolver import evaluate_condition, resolve_effect_ir

    ctx_card_id = context.get("card_id", "")
    acted_on_owner: str | None = None
    if ctx_card_id:
        acted_on = gs.get_instance(ctx_card_id)
        if acted_on:
            acted_on_owner = acted_on.controller_id

    # Collect all eligible replacements first
    eligible: list[Any] = []
    for rep in list(gs.active_replacements):
        if rep.watch_event != event.value:
            continue

        source = gs.get_instance(rep.source_instance_id)
        if not source:
            continue

        if not evaluate_condition(rep.condition, source, gs):
            continue

        if rep.target_spec and ctx_card_id:
            spec = TargetSpec.from_dict(rep.target_spec)
            target_card = gs.get_instance(ctx_card_id)
            if target_card and not _replacement_target_matches(
                spec, source, target_card
            ):
                continue

        eligible.append((rep, source))

    if not eligible:
        return False

    # Sort per rule 368: acted-on object's owner's replacements first,
    # then turn player, then others.
    def _rep_sort_key(item: tuple) -> tuple:
        _rep, _source = item
        ctrl = _source.controller_id
        if acted_on_owner and ctrl == acted_on_owner:
            return (0,)
        if ctrl == gs.turn_player_id:
            return (1,)
        try:
            idx = gs.player_order.index(ctrl)
        except ValueError:
            idx = 999
        return (2, idx)

    eligible.sort(key=_rep_sort_key)

    # Apply only the first matching replacement (rule 366)
    rep, source = eligible[0]
    if rep.alternative_ir:
        logs = resolve_effect_ir(rep.alternative_ir, source, gs)
        for msg in logs:
            gs.log.add(msg)
        logger.info(
            "[REPLACEMENT] %s replaces %s via %s",
            source.name, event.value, rep.ability_id,
        )
    return True


def check_primitive_replacement(
    gs: GameState,
    primitive_type: str,
    source: CardInstance,
    targets: list[str],
) -> dict[str, Any] | None:
    """Check if any replacement intercepts a primitive before it executes.

    Called by the effect resolver before each primitive. If a replacement
    matches, returns the alternative IR node to execute instead.
    Returns None if no replacement applies.

    Per rule 368: when multiple replacements apply, the owner of the
    acted-on object decides order.  We sort eligible replacements so
    the acted-on target's controller comes first.
    """
    from .effect_ir import TargetSpec
    from .effect_resolver import evaluate_condition

    # Determine the acted-on object's controller for ordering
    acted_on_owner: str | None = None
    if targets:
        first_target = gs.get_instance(targets[0])
        if first_target:
            acted_on_owner = first_target.controller_id

    eligible: list[tuple] = []
    for rep in list(gs.active_replacements):
        if rep.watch_event != primitive_type:
            continue

        rep_source = gs.get_instance(rep.source_instance_id)
        if not rep_source:
            continue

        if not evaluate_condition(rep.condition, rep_source, gs):
            continue

        # Check if any of the targets match the replacement's target filter
        if rep.target_spec and targets:
            spec = TargetSpec.from_dict(rep.target_spec)
            has_match = False
            for tid in targets:
                target_card = gs.get_instance(tid)
                if target_card and _replacement_target_matches(spec, rep_source, target_card):
                    has_match = True
                    break
            if not has_match:
                continue

        eligible.append((rep, rep_source))

    if not eligible:
        return None

    # Sort per rule 368
    def _rep_sort_key(item: tuple) -> tuple:
        _rep, _src = item
        ctrl = _src.controller_id
        if acted_on_owner and ctrl == acted_on_owner:
            return (0,)
        if ctrl == gs.turn_player_id:
            return (1,)
        try:
            idx = gs.player_order.index(ctrl)
        except ValueError:
            idx = 999
        return (2, idx)

    eligible.sort(key=_rep_sort_key)

    rep, rep_source = eligible[0]
    logger.info(
        "[REPLACEMENT] %s intercepts %s via %s",
        rep_source.name, primitive_type, rep.ability_id,
    )
    return rep.alternative_ir


def _replacement_target_matches(
    spec: Any,  # TargetSpec
    source: CardInstance,
    target: CardInstance,
) -> bool:
    """Check if a target card matches a replacement's target spec."""
    controller = source.controller_id
    if spec.scope == "friendly" and target.controller_id != controller:
        return False
    if spec.scope == "enemy" and target.controller_id == controller:
        return False
    if spec.scope == "self" and target.instance_id != source.instance_id:
        return False
    if spec.location == "here" and source.location_id:
        if target.location_id != source.location_id:
            return False
    return True
