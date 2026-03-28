"""Validates incoming player actions before they execute.

All validation is rules-first per rules_extracted.txt.
Key rules referenced:
  - 316:   Action Phase — only turn player plays in Neutral Open
  - 309:   Open/Closed state determines what can be played
  - 312:   Priority — who can take actions
  - 352.7: Valid targets required to put spell on chain
  - 354.1.a: Rune Add abilities (Reaction) can be used during cost payment
  - 416.2: Add abilities resolve immediately, no chain
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .enums import (
    ActionType,
    CardType,
    Keyword,
    Phase,
    TurnState,
    ZoneType,
)
from .game_state import GameState
from .keywords import can_play_in_state

logger = logging.getLogger("riftbound.validator")


@dataclass
class ValidationResult:
    ok: bool
    error: str = ""


def validate_action(
    gs: GameState,
    player_id: str,
    action_type: ActionType,
    payload: dict,
) -> ValidationResult:
    """Validate an incoming action. Returns ValidationResult."""

    if gs.game_over:
        return ValidationResult(False, "Game is over")

    validators = {
        ActionType.MULLIGAN_CHOICE: _validate_mulligan,
        ActionType.ADVANCE_PHASE: _validate_advance_phase,
        ActionType.PASS_PRIORITY: _validate_pass_priority,
        ActionType.PASS_FOCUS: _validate_pass_focus,
        ActionType.PLAY_CARD: _validate_play_card,
        ActionType.MOVE_UNIT: _validate_move_unit,
        ActionType.EXHAUST_RUNE: _validate_exhaust_rune,
        ActionType.RECYCLE_RUNE: _validate_recycle_rune,
        ActionType.ACTIVATE_ABILITY: _validate_activate_ability,
        ActionType.ASSIGN_DAMAGE: _validate_assign_damage,
        ActionType.CONCEDE: _validate_concede,
    }

    validator = validators.get(action_type)
    if not validator:
        return ValidationResult(False, f"Unknown action type: {action_type}")

    return validator(gs, player_id, payload)


def _validate_mulligan(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    if gs.phase != Phase.SETUP_MULLIGAN:
        return ValidationResult(False, "Not in mulligan phase")
    if gs.mulligan_done.get(player_id, False):
        return ValidationResult(False, "Already completed mulligan")
    return ValidationResult(True)


def _validate_advance_phase(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    if player_id != gs.turn_player_id:
        return ValidationResult(False, "Not your turn")
    if gs.phase not in (Phase.AWAKEN, Phase.ACTION):
        return ValidationResult(False, f"Cannot advance from {gs.phase.value}")
    if not gs.chain.is_empty:
        return ValidationResult(False, "Chain must be empty to advance phase")
    if gs.active_showdown:
        return ValidationResult(False, "Cannot advance during a showdown")
    if gs.active_combat:
        return ValidationResult(False, "Cannot advance during combat")
    # Check for staged combats/showdowns
    for bf in gs.battlefields.values():
        if bf.showdown_staged or bf.combat_staged:
            return ValidationResult(False, "Staged combat/showdown must resolve first")
    return ValidationResult(True)


def _validate_pass_priority(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    """Rule 334.1.c: Active Player passes priority to next Player in Turn Order."""
    if player_id != gs.active_player_id:
        return ValidationResult(False, "You don't have priority")
    if gs.chain.is_empty:
        return ValidationResult(False, "No chain to pass priority on")
    return ValidationResult(True)


def _validate_pass_focus(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    if gs.active_showdown is None:
        return ValidationResult(False, "No active showdown")
    if player_id != gs.active_showdown.focus_player_id:
        return ValidationResult(False, "You don't have focus")
    return ValidationResult(True)


def _validate_play_card(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    """
    Validates card play per rules 346-356.
    Key checks:
      - Card is in a playable zone (hand, champion zone, facedown)
      - Correct phase (Action or valid Showdown/Closed state with keywords)
      - Current turn state allows play (Open/Closed + keyword check)
      - Cost is payable
      - Target validation (rule 352.7)
    """
    instance_id = payload.get("instance_id")
    if not instance_id:
        return ValidationResult(False, "No card specified")

    card = gs.get_instance(instance_id)
    if not card:
        return ValidationResult(False, "Card not found")

    # Must be in hand or champion zone or facedown
    if card.zone not in (ZoneType.HAND, ZoneType.CHAMPION_ZONE, ZoneType.FACEDOWN_ZONE):
        return ValidationResult(False, f"Card is in {card.zone.value}, not playable")

    if card.owner_id != player_id and card.zone != ZoneType.FACEDOWN_ZONE:
        return ValidationResult(False, "Not your card")

    # --- PHASE CHECK ---
    # Rule 316: Cards can only be played during Action Phase (or combat sub-phases for Action/Reaction)
    valid_play_phases = (Phase.ACTION, Phase.SHOWDOWN, Phase.COMBAT_DAMAGE, Phase.COMBAT_RESOLUTION)
    if gs.phase not in valid_play_phases:
        return ValidationResult(False, f"Cannot play cards during {gs.phase.value} phase")

    # --- TIMING CHECK ---
    turn_state = gs.get_turn_state()
    is_showdown = turn_state in (TurnState.SHOWDOWN_OPEN, TurnState.SHOWDOWN_CLOSED)
    is_closed = turn_state in (TurnState.NEUTRAL_CLOSED, TurnState.SHOWDOWN_CLOSED)

    # Rule 316.2.b: In Neutral Open, only Turn Player can play cards
    if turn_state == TurnState.NEUTRAL_OPEN and player_id != gs.turn_player_id:
        return ValidationResult(False, "Not your turn")

    # Rule 312.1: In Closed state, must have priority to play Reactions
    if is_closed and player_id != gs.active_player_id:
        return ValidationResult(False, "You don't have priority")

    # Rule 309/310: Check if card keywords allow play in current state
    if is_closed or is_showdown:
        if not can_play_in_state(list(card.definition.keywords), is_showdown, is_closed):
            return ValidationResult(False, "Card cannot be played in current state")

    # --- COST CHECK ---
    if card.zone != ZoneType.FACEDOWN_ZONE:  # Hidden cards play for free
        ps = gs.players[player_id]
        cost_e = card.definition.cost_energy
        cost_p = card.definition.cost_power_dict()

        logger.debug(
            "[VALIDATE] %s playing %s | pool: E=%d P=%s | cost: E=%d P=%s",
            player_id[:6], card.name,
            ps.rune_pool.energy, dict(ps.rune_pool.power),
            cost_e, cost_p,
        )

        if not ps.rune_pool.can_pay(cost_e, cost_p):
            return ValidationResult(
                False,
                f"Cannot afford {card.name} (need E={cost_e} P={cost_p}, "
                f"have E={ps.rune_pool.energy} P={dict(ps.rune_pool.power)})"
            )

    # --- TARGET VALIDATION (rule 352.7) ---
    targets = payload.get("targets", [])
    if card.card_type == CardType.SPELL:
        result = _validate_spell_targets(gs, card, player_id, targets)
        if not result.ok:
            return result

    return ValidationResult(True)


def _validate_spell_targets(
    gs: GameState,
    card,
    player_id: str,
    targets: list[str],
) -> ValidationResult:
    """
    Rule 352.7: 'In order to put a spell or ability on the chain,
    valid choices must be made for all targets.'
    """
    for ability in card.definition.abilities:
        if ability.targets_required > 0:
            # Check targets were provided
            if len(targets) < ability.targets_required:
                # Check if valid targets even exist
                available = _count_available_targets(gs, ability.target_type, player_id)
                if available == 0:
                    return ValidationResult(
                        False,
                        f"No valid targets available for {card.name}"
                    )
                return ValidationResult(
                    False,
                    f"{card.name} requires {ability.targets_required} target(s), "
                    f"got {len(targets)}"
                )

            # Validate each target is legal
            for tid in targets[:ability.targets_required]:
                if not _is_valid_target(gs, tid, ability.target_type, player_id):
                    return ValidationResult(
                        False,
                        f"Invalid target for {card.name}"
                    )
    return ValidationResult(True)


def _count_available_targets(gs: GameState, target_type: str, player_id: str) -> int:
    """Count how many valid targets exist for a given target type."""
    count = 0
    if target_type in ("unit", "unit_at_battlefield"):
        for inst in gs.instances.values():
            if inst.card_type == CardType.UNIT and inst.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD):
                if target_type == "unit_at_battlefield" and inst.zone != ZoneType.BATTLEFIELD:
                    continue
                count += 1
    elif target_type == "friendly_unit":
        for inst in gs.instances.values():
            if (inst.card_type == CardType.UNIT
                    and inst.controller_id == player_id
                    and inst.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD)):
                count += 1
    elif target_type == "spell_on_chain":
        for item in gs.chain.stack:
            if item.card_instance_id:
                c = gs.get_instance(item.card_instance_id)
                if c and c.card_type == CardType.SPELL:
                    count += 1
    elif target_type in ("unit_and_friendly_unit", "friendly_unit_and_battlefield"):
        # For multi-target spells, just check at least some targets exist
        count = 1  # simplified — let individual target validation catch issues
    return count


def _is_valid_target(gs: GameState, target_id: str, target_type: str, player_id: str) -> bool:
    """Check if a specific target is valid for the given target type."""
    target = gs.get_instance(target_id)
    if not target:
        # Might be a battlefield_id
        if target_type in ("friendly_unit_and_battlefield",) and target_id in gs.battlefields:
            return True
        return False

    if target_type == "unit":
        return target.card_type == CardType.UNIT and target.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD)
    elif target_type == "unit_at_battlefield":
        return target.card_type == CardType.UNIT and target.zone == ZoneType.BATTLEFIELD
    elif target_type == "friendly_unit":
        return (target.card_type == CardType.UNIT
                and target.controller_id == player_id
                and target.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD))
    elif target_type == "spell_on_chain":
        return (target.card_type == CardType.SPELL
                and target.zone == ZoneType.CHAIN
                and target.controller_id != player_id)
    elif target_type in ("unit_and_friendly_unit", "friendly_unit_and_battlefield"):
        # Multi-type validation: just check it's a valid game object
        return target.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD)
    return True  # unknown type — allow


def _validate_move_unit(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    """Rule 143: Standard Move — only during Neutral Open Action Phase."""
    if player_id != gs.turn_player_id:
        return ValidationResult(False, "Not your turn")

    if gs.phase != Phase.ACTION:
        return ValidationResult(False, "Can only move during Action Phase")

    turn_state = gs.get_turn_state()
    if turn_state != TurnState.NEUTRAL_OPEN:
        return ValidationResult(False, "Can only move in Neutral Open state")

    instance_id = payload.get("instance_id")
    card = gs.get_instance(instance_id)
    if not card:
        return ValidationResult(False, "Unit not found")

    if card.card_type != CardType.UNIT:
        return ValidationResult(False, "Can only move units")

    if card.controller_id != player_id:
        return ValidationResult(False, "Not your unit")

    if card.exhausted:
        return ValidationResult(False, "Unit is exhausted")

    destination = payload.get("destination", {})
    dest_zone = destination.get("zone")
    dest_id = destination.get("id")

    if dest_zone == "battlefield":
        if card.zone == ZoneType.BATTLEFIELD:
            # Battlefield to battlefield requires Ganking
            if not card.has_keyword(Keyword.GANKING):
                return ValidationResult(False, "Unit needs Ganking to move between battlefields")
        elif card.zone != ZoneType.BASE:
            return ValidationResult(False, "Unit must be in base or at a battlefield to move")

        # Check destination battlefield exists
        if dest_id not in gs.battlefields:
            return ValidationResult(False, "Destination battlefield not found")

    elif dest_zone == "base":
        if card.zone != ZoneType.BATTLEFIELD:
            return ValidationResult(False, "Unit must be at a battlefield to move to base")
    else:
        return ValidationResult(False, "Invalid destination")

    return ValidationResult(True)


def _validate_exhaust_rune(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    """
    Rule 160.2.a / 416: '[E]: [Reaction] — Add [1]'
    Rune Add abilities are Reaction-timed and resolve immediately.
    Can be used in ANY state (Open, Closed, Showdown) and on any player's turn.
    """
    instance_id = payload.get("instance_id")
    rune = gs.get_instance(instance_id)
    if not rune:
        return ValidationResult(False, "Rune not found")
    if rune.card_type != CardType.RUNE:
        return ValidationResult(False, "Not a rune")
    if rune.controller_id != player_id:
        return ValidationResult(False, "Not your rune")
    if rune.zone != ZoneType.RUNE_BOARD:
        return ValidationResult(False, "Rune not on board")
    if rune.exhausted:
        return ValidationResult(False, "Rune already exhausted")
    # No phase or state restriction — Reaction timing (rule 416.3)
    return ValidationResult(True)


def _validate_recycle_rune(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    """
    Rule 160.2.b / 416: 'Recycle this: [Reaction] — Add [C]'
    Rune Recycle abilities are Reaction-timed and resolve immediately.
    Can be used in ANY state.
    """
    instance_id = payload.get("instance_id")
    rune = gs.get_instance(instance_id)
    if not rune:
        return ValidationResult(False, "Rune not found")
    if rune.card_type != CardType.RUNE:
        return ValidationResult(False, "Not a rune")
    if rune.controller_id != player_id:
        return ValidationResult(False, "Not your rune")
    if rune.zone != ZoneType.RUNE_BOARD:
        return ValidationResult(False, "Rune not on board")
    return ValidationResult(True)


def _validate_activate_ability(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    source_id = payload.get("source_id")
    ability_id = payload.get("ability_id")

    source = gs.get_instance(source_id)
    if not source:
        return ValidationResult(False, "Source not found")
    if source.controller_id != player_id:
        return ValidationResult(False, "Not your game object")

    # Find the ability
    ability = None
    for ab in source.definition.abilities:
        if ab.ability_id == ability_id:
            ability = ab
            break
    if not ability:
        return ValidationResult(False, "Ability not found")

    # Check timing
    turn_state = gs.get_turn_state()
    is_showdown = turn_state in (TurnState.SHOWDOWN_OPEN, TurnState.SHOWDOWN_CLOSED)
    is_closed = turn_state in (TurnState.NEUTRAL_CLOSED, TurnState.SHOWDOWN_CLOSED)

    if turn_state == TurnState.NEUTRAL_OPEN and player_id != gs.turn_player_id:
        if ability.timing not in ("reaction",):
            return ValidationResult(False, "Not your turn")

    # Check if ability can be activated (exhaust cost)
    if ability.cost and ability.cost.exhaust_source and source.exhausted:
        return ValidationResult(False, "Source is exhausted")

    return ValidationResult(True)


def _validate_assign_damage(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    if not gs.active_combat:
        return ValidationResult(False, "No active combat")
    if gs.active_combat.phase != "waiting_assignments":
        return ValidationResult(False, "Not in damage assignment phase")

    combat = gs.active_combat
    if player_id == combat.attacker_id and combat.attacker_assignment is not None:
        return ValidationResult(False, "Already submitted assignment")
    if player_id == combat.defender_id and combat.defender_assignment is not None:
        return ValidationResult(False, "Already submitted assignment")
    if player_id not in (combat.attacker_id, combat.defender_id):
        return ValidationResult(False, "Not a combat participant")

    return ValidationResult(True)


def _validate_concede(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    return ValidationResult(True)
