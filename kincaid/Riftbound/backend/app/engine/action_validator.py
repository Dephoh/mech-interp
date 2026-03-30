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
    ControlStatus,
    Keyword,
    Phase,
    TurnState,
    ZoneType,
)
from .game_state import GameState
from .keywords import (
    can_play_in_state,
    check_unique_violation,
    get_accelerate_cost,
    get_deflect_cost,
)
from .target_system import count_available_targets, validate_target

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
        ActionType.SUBMIT_CHOICE: _validate_submit_choice,
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

    # --- UNIQUE CHECK (rule 355.2 / Unique keyword) ---
    # A player cannot play a second copy of a Unique card they already control.
    if check_unique_violation(card, gs):
        return ValidationResult(
            False,
            f"Cannot play {card.name}: Unique card already on the board"
        )

    # --- ADDITIONAL COST FEASIBILITY (rule 353.2.a) ---
    # If the card has mandatory additional costs (e.g., "kill a friendly unit"),
    # verify the cost can actually be paid before allowing play.
    if card.definition.abilities:
        for ability in card.definition.abilities:
            if ability.cost and ability.cost.additional_costs:
                for add_cost in ability.cost.additional_costs:
                    result = _validate_additional_cost_feasible(gs, player_id, add_cost, card)
                    if not result.ok:
                        return result

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

        # If paying Accelerate, include that cost in the check (rule 731)
        if payload.get("pay_accelerate") and card.has_keyword(Keyword.ACCELERATE):
            acc_e, acc_p = get_accelerate_cost(card)
            total_e = cost_e + acc_e
            total_p = dict(cost_p)
            for dom, amt in acc_p.items():
                total_p[dom] = total_p.get(dom, 0) + amt
            if not ps.rune_pool.can_pay(total_e, total_p):
                return ValidationResult(
                    False,
                    f"Cannot afford {card.name} with Accelerate "
                    f"(need E={total_e} P={total_p}, "
                    f"have E={ps.rune_pool.energy} P={dict(ps.rune_pool.power)})"
                )
        elif not ps.rune_pool.can_pay(cost_e, cost_p):
            return ValidationResult(
                False,
                f"Cannot afford {card.name} (need E={cost_e} P={cost_p}, "
                f"have E={ps.rune_pool.energy} P={dict(ps.rune_pool.power)})"
            )

    # --- UNIT DESTINATION VALIDATION (rule 352.2) ---
    if card.card_type == CardType.UNIT:
        destination = payload.get("destination")
        if destination:
            dest_zone = destination.get("zone")
            dest_id = destination.get("id")
            if dest_zone == "battlefield":
                if dest_id not in gs.battlefields:
                    return ValidationResult(False, "Destination battlefield not found")
                bf = gs.battlefields[dest_id]
                # Rule 352.2.a: can play to base or a battlefield the controller controls
                # Ambush exception: can play to any BF where you have units
                has_ambush = card.has_keyword(Keyword.AMBUSH)
                player_controls_bf = (
                    bf.control_status == ControlStatus.CONTROLLED
                    and bf.controller_id == player_id
                )
                if has_ambush:
                    # Ambush allows playing to any BF where the player has units
                    player_has_units_here = any(
                        (inst := gs.get_instance(iid)) and
                        (inst.controller_id == player_id or inst.owner_id == player_id)
                        for iid in bf.units
                    )
                    if not player_controls_bf and not player_has_units_here:
                        return ValidationResult(
                            False,
                            "Ambush units can only be played to a battlefield where you have units"
                        )
                elif not player_controls_bf:
                    return ValidationResult(
                        False,
                        "Can only play units to your Base or a Battlefield you control"
                    )
            elif dest_zone not in ("base", None):
                return ValidationResult(False, "Invalid destination for unit")
        # No destination specified: defaults to base (rule 352.2.a)

    # --- GEAR DESTINATION VALIDATION (rule 356.2.d) ---
    # Gear always enters the player's Base — reject attempts to send it elsewhere.
    if card.card_type == CardType.GEAR:
        destination = payload.get("destination")
        if destination:
            dest_zone = destination.get("zone")
            if dest_zone and dest_zone not in ("base",):
                return ValidationResult(
                    False,
                    "Gear must enter the base (rule 356.2.d)"
                )

    # --- TARGET VALIDATION (rule 352.7 / 352.9.c) ---
    # Rule 352.9.c: Units and gear never have targets (their abilities may,
    # but those are validated when the abilities are activated, not at play time).
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
    Rule 352.8.c: A spell cannot target itself on the chain.
    Rule 353.2.a.2 / 735: Deflect imposes mandatory additional Power cost.
    """
    for ability in card.definition.abilities:
        if ability.targets_required > 0:
            # Build target spec from IR or legacy target_type
            spec = _build_target_spec(ability, player_id)

            # Check targets were provided
            if len(targets) < ability.targets_required:
                # Check if valid targets even exist
                if spec:
                    available = count_available_targets(gs, spec, controller_id=player_id)
                else:
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
                # Rule 352.8.c: A spell cannot target itself
                if tid == card.instance_id:
                    return ValidationResult(
                        False,
                        f"{card.name} cannot target itself"
                    )

                if spec:
                    if not validate_target(gs, tid, spec, controller_id=player_id):
                        return ValidationResult(False, f"Invalid target for {card.name}")
                else:
                    if not _is_valid_target(gs, tid, ability.target_type, player_id):
                        return ValidationResult(False, f"Invalid target for {card.name}")

                # Rule 353.2.a.2 / 735: Deflect — if the target has Deflect,
                # the caster must pay additional Power (mandatory additional cost).
                target_inst = gs.get_instance(tid)
                if target_inst:
                    deflect_cost = get_deflect_cost(target_inst)
                    if deflect_cost > 0:
                        extra_power = payload_deflect_power(
                            gs, player_id, card, deflect_cost
                        )
                        if not extra_power:
                            return ValidationResult(
                                False,
                                f"Cannot afford Deflect cost of {deflect_cost} "
                                f"Power to target {target_inst.name}"
                            )
    return ValidationResult(True)


def payload_deflect_power(
    gs: GameState,
    player_id: str,
    card,
    deflect_cost: int,
) -> bool:
    """Check if the player can afford the Deflect additional cost.

    Rule 735: Spells/abilities that choose a permanent with Deflect [X]
    must pay [X] additional Power of any domain. This is checked AFTER
    the base cost has been validated (so the pool might already be committed).
    We do a conservative check: total available power minus what the card's
    base power cost requires must cover the deflect cost.
    """
    ps = gs.players[player_id]
    base_power = card.definition.cost_power_dict()
    remaining_power = sum(ps.rune_pool.power.values()) - sum(base_power.values())
    return remaining_power >= deflect_cost


def _build_target_spec(ability, player_id: str) -> dict | None:
    """Build a TargetSpec dict from an ability's effect_ir or target_type string.

    Prefers the structured target in effect_ir; falls back to converting
    the legacy target_type string.
    """
    # Prefer target spec from effect_ir
    if ability.effect_ir:
        ir = ability.effect_ir
        # Walk through sequence to find first node with a target
        if ir.get("type") == "sequence":
            for step in ir.get("steps", []):
                if "target" in step:
                    return step["target"]
        elif "target" in ir:
            return ir["target"]

    # Fall back to legacy target_type string → TargetSpec dict
    target_type = ability.target_type
    if not target_type:
        return None

    _LEGACY_MAP = {
        "unit": {"obj_type": "unit", "scope": "any"},
        "enemy_unit": {"obj_type": "unit", "scope": "enemy"},
        "friendly_unit": {"obj_type": "unit", "scope": "friendly"},
        "unit_at_battlefield": {"obj_type": "unit", "scope": "any", "zone": "battlefield"},
        "spell_on_chain": {"obj_type": "spell", "zone": "chain"},
        "gear": {"obj_type": "gear", "scope": "any"},
        "friendly_gear": {"obj_type": "gear", "scope": "friendly"},
        "enemy_gear": {"obj_type": "gear", "scope": "enemy"},
    }
    spec = _LEGACY_MAP.get(target_type)
    if spec:
        return dict(spec)  # copy

    # Unknown target type — allow validation to pass
    return None


def _count_available_targets(gs: GameState, target_type: str, player_id: str) -> int:
    """Count how many valid targets exist for a given target type.

    Delegates to target_system.count_available_targets when possible.
    """
    # Build a spec from the target_type string
    _LEGACY_MAP = {
        "unit": {"obj_type": "unit", "scope": "any"},
        "enemy_unit": {"obj_type": "unit", "scope": "enemy"},
        "friendly_unit": {"obj_type": "unit", "scope": "friendly"},
        "unit_at_battlefield": {"obj_type": "unit", "scope": "any", "zone": "battlefield"},
        "spell_on_chain": {"obj_type": "spell", "zone": "chain"},
        "gear": {"obj_type": "gear", "scope": "any"},
        "friendly_gear": {"obj_type": "gear", "scope": "friendly"},
    }
    spec = _LEGACY_MAP.get(target_type)
    if spec:
        return count_available_targets(gs, spec, controller_id=player_id)

    # Multi-target or unknown — optimistic fallback
    return 1


def _is_valid_target(gs: GameState, target_id: str, target_type: str, player_id: str) -> bool:
    """Check if a specific target is valid for the given target type.

    Delegates to target_system.validate_target when possible.
    """
    _LEGACY_MAP = {
        "unit": {"obj_type": "unit", "scope": "any"},
        "enemy_unit": {"obj_type": "unit", "scope": "enemy"},
        "friendly_unit": {"obj_type": "unit", "scope": "friendly"},
        "unit_at_battlefield": {"obj_type": "unit", "scope": "any", "zone": "battlefield"},
        "spell_on_chain": {"obj_type": "spell", "zone": "chain"},
        "gear": {"obj_type": "gear", "scope": "any"},
        "friendly_gear": {"obj_type": "gear", "scope": "friendly"},
    }
    spec = _LEGACY_MAP.get(target_type)
    if spec:
        return validate_target(gs, target_id, spec, controller_id=player_id)

    # Battlefield IDs for multi-target abilities
    if target_id in gs.battlefields:
        return True

    # Unknown type — allow
    return True


def _validate_additional_cost_feasible(
    gs: GameState,
    player_id: str,
    cost_key: str,
    card,
) -> ValidationResult:
    """Rule 353.2.a: Validate that a mandatory additional cost can be paid.

    Additional costs are encoded as short keys in CostDefinition.additional_costs,
    e.g. "kill_friendly_unit", "discard_card". This checks the game state has
    at least one valid object to pay each cost type.
    """
    if cost_key == "kill_friendly_unit":
        # Must control at least one unit on the board
        friendly_units = 0
        for uid_list in gs.base_units.values():
            for uid in uid_list:
                inst = gs.get_instance(uid)
                if inst and inst.controller_id == player_id:
                    friendly_units += 1
        for bf in gs.battlefields.values():
            for uid in bf.units:
                inst = gs.get_instance(uid)
                if inst and inst.controller_id == player_id:
                    friendly_units += 1
        if friendly_units == 0:
            return ValidationResult(
                False,
                f"Cannot play {card.name}: no friendly unit to pay additional cost"
            )
    elif cost_key == "discard_card":
        ps = gs.players[player_id]
        # Must have at least one card in hand (besides the card being played)
        other_cards = [c for c in ps.hand if c != card.instance_id]
        if len(other_cards) == 0:
            return ValidationResult(
                False,
                f"Cannot play {card.name}: no card in hand to discard for additional cost"
            )
    # Unknown additional cost keys are allowed through (future-proof)
    return ValidationResult(True)


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

    # Check energy/power cost (rule 369)
    if ability.cost and (ability.cost.energy > 0 or ability.cost.power):
        ps = gs.players[player_id]
        cost_power = ability.cost.power_dict()
        if not ps.rune_pool.can_pay(ability.cost.energy, cost_power):
            return ValidationResult(False, "Cannot afford ability cost")

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


def _validate_submit_choice(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    if gs.pending_choice is None:
        return ValidationResult(False, "No pending choice")
    if gs.pending_choice.get("controller_id") != player_id:
        return ValidationResult(False, "Not your choice to make")
    return ValidationResult(True)


def _validate_concede(gs: GameState, player_id: str, payload: dict) -> ValidationResult:
    return ValidationResult(True)
