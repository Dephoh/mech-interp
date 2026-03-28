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


# Map from trigger_condition strings on AbilityDefinition to GameEvents
TRIGGER_TO_EVENTS: dict[str, list[GameEvent]] = {
    "on_play": [GameEvent.UNIT_ENTERED, GameEvent.UNIT_PLAYED],
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
    Order: Turn Player's triggers first, then others in turn order.

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

    # Scan all board objects
    for card in _get_board_objects(gs):
        for ability in card.definition.abilities:
            if ability.ability_type not in (AbilityType.TRIGGERED, "triggered"):
                continue
            if ability.trigger_condition not in matching_triggers:
                continue
            if not (ability.effect_ir or ability.effect_script):
                continue

            # Check if this trigger applies to this specific event context
            if not _trigger_applies(card, ability, event, context, gs):
                continue

            triggered_items.append((
                card.controller_id,
                ability.ability_id,
                card.instance_id,
                card,
            ))

    if not triggered_items:
        return []

    # Order: Turn Player's triggers first, then by turn order
    turn_player = gs.turn_player_id
    player_order = gs.player_order

    def sort_key(item: tuple) -> tuple:
        ctrl = item[0]
        if ctrl == turn_player:
            return (0, 0)
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

    # Default: trigger fires
    return True


def check_replacements(
    gs: GameState,
    event: GameEvent,
    context: dict[str, Any],
) -> bool:
    """Check if any replacement effect intercepts this event.

    Returns True if the event was replaced (original should not execute).
    Currently a stub for Phase 4 implementation.
    """
    # Phase 4 will implement this
    return False
