"""Executes validated player actions, mutating GameState."""

from __future__ import annotations

from .chain import (
    pass_priority,
    push_ability_to_chain,
    push_card_to_chain,
    resolve_top_item,
)
from .cleanup import run_cleanup
from .combat import pass_focus, submit_damage_assignment
from .enums import (
    ActionType,
    CardType,
    ControlStatus,
    Keyword,
    ZoneType,
)
from .game_state import GameState, _draw_cards
from .keywords import apply_accelerate, get_accelerate_cost
from .state_machine import advance_phase, start_game
from .trigger_system import GameEvent, fire_event


def execute_action(
    gs: GameState,
    player_id: str,
    action_type: ActionType,
    payload: dict,
) -> list[str]:
    """
    Execute a validated action. Returns log messages.
    Caller must validate before calling this.
    """
    executors = {
        ActionType.MULLIGAN_CHOICE: _exec_mulligan,
        ActionType.ADVANCE_PHASE: _exec_advance_phase,
        ActionType.PASS_PRIORITY: _exec_pass_priority,
        ActionType.PASS_FOCUS: _exec_pass_focus,
        ActionType.PLAY_CARD: _exec_play_card,
        ActionType.MOVE_UNIT: _exec_move_unit,
        ActionType.EXHAUST_RUNE: _exec_exhaust_rune,
        ActionType.RECYCLE_RUNE: _exec_recycle_rune,
        ActionType.ACTIVATE_ABILITY: _exec_activate_ability,
        ActionType.ASSIGN_DAMAGE: _exec_assign_damage,
        ActionType.CONCEDE: _exec_concede,
    }

    executor = executors.get(action_type)
    if not executor:
        return [f"No executor for action type: {action_type}"]

    logs = executor(gs, player_id, payload)

    # Run cleanup after every state mutation (except mulligan and concede which handle their own)
    if action_type not in (ActionType.MULLIGAN_CHOICE, ActionType.CONCEDE):
        cleanup_logs = run_cleanup(gs)
        logs.extend(cleanup_logs)

    return logs


def _exec_mulligan(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Process mulligan choice: keep specified cards, redraw the rest."""
    ps = gs.players[player_id]
    keep_indices = payload.get("keep_indices", [])
    logs: list[str] = []

    current_hand = list(ps.hand)
    keep_ids = set()
    for idx in keep_indices:
        if 0 <= idx < len(current_hand):
            keep_ids.add(current_hand[idx])

    # Return non-kept cards to deck
    returned = []
    for iid in current_hand:
        if iid not in keep_ids:
            inst = gs.instances[iid]
            inst.zone = ZoneType.MAIN_DECK
            inst.location_id = None
            ps.hand.remove(iid)
            ps.main_deck.append(iid)
            returned.append(iid)

    # Shuffle deck
    import random
    random.shuffle(ps.main_deck)

    # Draw replacements
    num_draw = len(returned)
    drawn = _draw_cards(gs, player_id, num_draw)
    logs.append(f"{ps.display_name} mulligans {len(returned)} cards, draws {len(drawn)}")

    # Mark mulligan complete
    gs.mulligan_done[player_id] = True

    # Check if all players have completed mulligan
    if all(gs.mulligan_done.values()):
        logs.append("All players have completed mulligan")
        logs.extend(start_game(gs))

    return logs


def _exec_advance_phase(gs: GameState, player_id: str, payload: dict) -> list[str]:
    return advance_phase(gs)


def _exec_pass_priority(gs: GameState, player_id: str, payload: dict) -> list[str]:
    logs: list[str] = []
    should_resolve = pass_priority(gs, player_id)

    if should_resolve:
        # Both players passed — resolve top chain item
        resolve_logs = resolve_top_item(gs)
        logs.extend(resolve_logs)
    else:
        logs.append(f"Priority passes to opponent")

    return logs


def _exec_pass_focus(gs: GameState, player_id: str, payload: dict) -> list[str]:
    return pass_focus(gs, player_id)


def _exec_play_card(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Play a card from hand/champion zone onto the chain."""
    instance_id = payload["instance_id"]
    targets = payload.get("targets", [])
    pay_accelerate = payload.get("pay_accelerate", False)

    card = gs.instances[instance_id]
    ps = gs.players[player_id]
    logs: list[str] = []

    # Pay costs (unless facedown)
    if card.zone != ZoneType.FACEDOWN_ZONE:
        cost_e = card.definition.cost_energy
        cost_p = card.definition.cost_power_dict()
        ps.rune_pool.spend(cost_e, cost_p)

        # Pay Accelerate cost if requested
        if pay_accelerate and card.has_keyword(Keyword.ACCELERATE):
            acc_e, acc_p = get_accelerate_cost(card)
            if ps.rune_pool.can_pay(acc_e, acc_p):
                ps.rune_pool.spend(acc_e, acc_p)
                card.accelerated = True
                logs.append(f"Accelerate cost paid for {card.name}")
            else:
                logs.append(f"Cannot afford Accelerate for {card.name}")

    # Remove from current zone
    if card.zone == ZoneType.HAND:
        ps.hand.remove(instance_id)
    elif card.zone == ZoneType.CHAMPION_ZONE:
        ps.champion_zone = None
    elif card.zone == ZoneType.FACEDOWN_ZONE:
        # Remove from battlefield facedown slot
        for bf in gs.battlefields.values():
            if bf.facedown_card == instance_id:
                bf.facedown_card = None
                break
        card.facedown = False

    # Track main deck cards played this turn
    if card.zone in (ZoneType.HAND,):
        ps.played_main_deck_this_turn += 1

    # Spells go on the chain; permanents also go through the chain
    push_card_to_chain(gs, card, player_id, targets)
    logs.append(f"{ps.display_name} plays {card.name}")

    return logs


def _exec_move_unit(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Move a unit between base and battlefields."""
    instance_id = payload["instance_id"]
    destination = payload["destination"]
    dest_zone = destination["zone"]
    dest_id = destination.get("id")

    unit = gs.instances[instance_id]
    logs: list[str] = []

    # Remove from current location
    if unit.zone == ZoneType.BASE:
        if instance_id in (gs.base_units.get(player_id) or []):
            gs.base_units[player_id].remove(instance_id)
    elif unit.zone == ZoneType.BATTLEFIELD:
        for bf in gs.battlefields.values():
            if instance_id in bf.units:
                bf.units.remove(instance_id)
                break

    # Place at destination
    if dest_zone == "battlefield":
        bf = gs.battlefields[dest_id]
        bf.units.append(instance_id)
        unit.zone = ZoneType.BATTLEFIELD
        unit.location_id = dest_id
        unit.exhausted = True

        # Apply Contested status if battlefield is uncontrolled or controlled by opponent
        if bf.control_status == ControlStatus.UNCONTROLLED:
            bf.contested_by = player_id
            logs.append(f"{unit.name} contests the battlefield")
        elif bf.control_status == ControlStatus.CONTROLLED and bf.controller_id != player_id:
            bf.contested_by = player_id
            logs.append(f"{unit.name} contests {bf.controller_id}'s battlefield")

        logs.append(f"{unit.name} moves to battlefield")
        # Fire move-to-battlefield trigger (covers on_move + on_move_to_bf)
        evt_logs = fire_event(gs, GameEvent.UNIT_MOVED_TO_BF, {
            "card_id": instance_id,
            "player_id": player_id,
            "battlefield_id": dest_id,
        })
        logs.extend(evt_logs)

    elif dest_zone == "base":
        gs.base_units.setdefault(player_id, []).append(instance_id)
        unit.zone = ZoneType.BASE
        unit.location_id = player_id
        unit.exhausted = True
        logs.append(f"{unit.name} retreats to base")
        # Fire move trigger (on_move only)
        evt_logs = fire_event(gs, GameEvent.UNIT_MOVED, {
            "card_id": instance_id,
            "player_id": player_id,
        })
        logs.extend(evt_logs)

    return logs


def _exec_exhaust_rune(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Exhaust a rune to gain Energy."""
    instance_id = payload["instance_id"]
    rune = gs.instances[instance_id]
    logs: list[str] = []

    # Find the exhaust ability and push to chain (or resolve immediately for basic runes)
    for ability in rune.definition.abilities:
        if ability.effect_script == "rune_add_energy":
            from .effects import resolve_effect
            result = resolve_effect("rune_add_energy", rune, gs, [])
            logs.extend(result)
            break
    else:
        rune.exhausted = True
        gs.players[player_id].rune_pool.add_energy(1)
        logs.append(f"{rune.name} exhausted: +1 Energy")

    return logs


def _exec_recycle_rune(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Recycle a rune to gain Power of its domain."""
    instance_id = payload["instance_id"]
    rune = gs.instances[instance_id]
    logs: list[str] = []

    for ability in rune.definition.abilities:
        if ability.effect_script == "rune_recycle_power":
            from .effects import resolve_effect
            result = resolve_effect("rune_recycle_power", rune, gs, [])
            logs.extend(result)
            break
    else:
        # Fallback: basic recycle
        from .effects import _recycle_rune
        if rune.definition.domains:
            domain = rune.definition.domains[0]
            gs.players[player_id].rune_pool.add_power(domain, 1)
            _recycle_rune(rune, gs)
            logs.append(f"{rune.name} recycled: +1 {domain.value} Power")

    # Fire recycle rune trigger ("When you recycle a rune, ...")
    evt_logs = fire_event(gs, GameEvent.RECYCLE_RUNE, {"player_id": player_id})
    logs.extend(evt_logs)

    return logs


def _exec_activate_ability(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Activate an ability on a game object, pushing it to the chain."""
    source_id = payload["source_id"]
    ability_id = payload["ability_id"]
    targets = payload.get("targets", [])

    source = gs.instances[source_id]

    # Find the ability
    ability = None
    for ab in source.definition.abilities:
        if ab.ability_id == ability_id:
            ability = ab
            break

    # Pay exhaust cost if needed
    if ability.cost and ability.cost.exhaust_source:
        source.exhausted = True

    # Push ability to chain
    push_ability_to_chain(gs, source, ability_id, player_id, targets)
    return [f"{source.name} ability activated"]


def _exec_assign_damage(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Submit damage assignment during combat."""
    assignments = payload.get("assignments", {})
    return submit_damage_assignment(gs, player_id, assignments)


def _exec_concede(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Player concedes the game."""
    winner_id = gs.opponent_id(player_id)
    gs.game_over = True
    gs.winner_id = winner_id
    gs.phase = gs.phase  # keep current phase in log
    loser_name = gs.players[player_id].display_name
    winner_name = gs.players[winner_id].display_name
    gs.log.add(f"{loser_name} concedes. {winner_name} wins!")
    return [f"{loser_name} concedes. {winner_name} wins!"]
