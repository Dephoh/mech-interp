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
        ActionType.SUBMIT_CHOICE: _exec_submit_choice,
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
    """Process mulligan choice: keep specified cards, redraw the rest.
    Rule 117.1: A player may choose up to two cards to set aside.
    Rule 117.3: Set-aside cards are recycled (placed at bottom of deck).
    """
    ps = gs.players[player_id]
    keep_indices = payload.get("keep_indices", [])
    logs: list[str] = []

    current_hand = list(ps.hand)
    keep_ids = set()
    for idx in keep_indices:
        if 0 <= idx < len(current_hand):
            keep_ids.add(current_hand[idx])

    # Rule 117.1: max 2 cards can be set aside
    returned = []
    for iid in current_hand:
        if iid not in keep_ids and len(returned) < 2:
            returned.append(iid)

    # Remove returned cards from hand
    for iid in returned:
        inst = gs.instances[iid]
        inst.zone = ZoneType.MAIN_DECK
        inst.location_id = None
        ps.hand.remove(iid)

    # Draw replacements first (before recycling set-aside cards)
    num_draw = len(returned)
    drawn = _draw_cards(gs, player_id, num_draw)

    # Rule 117.3: Recycle set-aside cards (place at bottom of deck)
    for iid in returned:
        ps.main_deck.append(iid)

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
        # Rule 353.5: Energy and Power costs can't be reduced below 0.
        cost_e = max(0, card.definition.cost_energy)
        cost_p = {dom: max(0, amt) for dom, amt in card.definition.cost_power_dict().items()}
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
    destination = payload.get("destination")
    push_card_to_chain(gs, card, player_id, targets, destination=destination)
    logs.append(f"{ps.display_name} plays {card.name}")

    # Rule 355: Post-play legality check.
    # Verify the board state is legal after the card enters. If a unit was
    # played to a battlefield, confirm the controller still controls that
    # battlefield (e.g. they didn't kill their only unit there as an
    # additional cost). This is a safety net; the validator should catch
    # most cases, but costs can change state.
    if card.card_type == CardType.UNIT and destination:
        dest_zone = destination.get("zone")
        dest_id = destination.get("id")
        if dest_zone == "battlefield" and dest_id in gs.battlefields:
            bf = gs.battlefields[dest_id]
            # Check the controller still has units there (including the
            # newly placed unit itself).
            controller_units = [
                uid for uid in bf.units
                if gs.get_instance(uid) and gs.get_instance(uid).controller_id == player_id
            ]
            if not controller_units:
                logs.append(
                    f"WARNING: {card.name} placed at battlefield "
                    f"{dest_id} but controller has no units there — "
                    f"legality check flagged (rule 355)"
                )

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
            bf.control_status = ControlStatus.CONTESTED
            bf.contested_by = player_id
            logs.append(f"{unit.name} contests the battlefield")
        elif bf.control_status == ControlStatus.CONTROLLED and bf.controller_id != player_id:
            bf.control_status = ControlStatus.CONTESTED
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
    """Exhaust a rune to gain Energy.

    Rule 160.2.a / 416: '[E]: [Reaction] — Add [1]'.
    Resolves via effect_ir when the ability carries one, otherwise inlines
    the basic exhaust-and-add-energy logic.
    """
    from .effect_resolver import resolve_effect_ir

    instance_id = payload["instance_id"]
    rune = gs.instances[instance_id]
    logs: list[str] = []

    # Prefer the effect_ir path on the exhaust ability
    for ability in rune.definition.abilities:
        if ability.ability_id.endswith("_exhaust") and ability.effect_ir:
            rune.exhausted = True
            result = resolve_effect_ir(ability.effect_ir, rune, gs, [])
            logs.extend(result)
            break
    else:
        # Fallback: inline basic exhaust + add energy
        rune.exhausted = True
        gs.players[player_id].rune_pool.add_energy(1)
        logs.append(f"{rune.name} exhausted: +1 Energy")

    return logs


def _exec_recycle_rune(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Recycle a rune to gain Power of its domain.

    Rule 160.2.b / 416: 'Recycle this: [Reaction] — Add [C]'.
    Resolves via effect_ir when the ability carries one, otherwise inlines
    the basic recycle-and-add-power logic.
    """
    from .effect_resolver import resolve_effect_ir

    instance_id = payload["instance_id"]
    rune = gs.instances[instance_id]
    logs: list[str] = []

    for ability in rune.definition.abilities:
        if ability.ability_id.endswith("_recycle") and ability.effect_ir:
            result = resolve_effect_ir(ability.effect_ir, rune, gs, [])
            logs.extend(result)
            _recycle_rune_to_deck(rune, gs)
            break
    else:
        # Fallback: inline basic recycle
        if rune.definition.domains:
            domain = rune.definition.domains[0]
            gs.players[player_id].rune_pool.add_power(domain, 1)
            _recycle_rune_to_deck(rune, gs)
            logs.append(f"{rune.name} recycled: +1 {domain.value} Power")

    # Fire recycle rune trigger ("When you recycle a rune, ...")
    evt_logs = fire_event(gs, GameEvent.RECYCLE_RUNE, {"player_id": player_id})
    logs.extend(evt_logs)

    return logs


def _recycle_rune_to_deck(rune: CardInstance, gs: GameState) -> None:
    """Move a rune from its current zone to the bottom of its owner's rune deck."""
    ps = gs.players[rune.owner_id]
    # Remove from rune board
    if rune.instance_id in (gs.base_runes.get(rune.controller_id) or []):
        gs.base_runes[rune.controller_id].remove(rune.instance_id)
    # Put on bottom of rune deck
    rune.zone = ZoneType.RUNE_DECK
    rune.location_id = None
    rune.exhausted = False
    ps.rune_deck.append(rune.instance_id)


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

    # Pay costs (rule 369)
    if ability.cost:
        if ability.cost.exhaust_source:
            source.exhausted = True
        if ability.cost.energy > 0 or ability.cost.power:
            ps = gs.players[player_id]
            ps.rune_pool.spend(ability.cost.energy, ability.cost.power_dict())

    # Push ability to chain
    push_ability_to_chain(gs, source, ability_id, player_id, targets)
    return [f"{source.name} ability activated"]


def _exec_assign_damage(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Submit damage assignment during combat."""
    assignments = payload.get("assignments", {})
    return submit_damage_assignment(gs, player_id, assignments)


def _exec_concede(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Player concedes the game.

    In 2v2 the entire team loses (rule 466.6.a.1).
    In FFA with >2 remaining, the player is removed but play continues
    (rule 651.2) — for now we end the game if only one player/team remains.
    """
    from .enums import GameMode

    loser_name = gs.players[player_id].display_name
    logs: list[str] = []

    remaining = [pid for pid in gs.player_order if pid != player_id]

    # In 2v2, teammate also loses
    if gs.game_mode == GameMode.TWO_VS_TWO:
        teammate = gs.teammate_id(player_id)
        if teammate:
            remaining = [pid for pid in remaining if pid != teammate]

    if len(remaining) == 1:
        winner_id = remaining[0]
        gs.game_over = True
        gs.winner_id = winner_id
        winner_name = gs.players[winner_id].display_name
        gs.log.add(f"{loser_name} concedes. {winner_name} wins!")
        logs.append(f"{loser_name} concedes. {winner_name} wins!")
    elif gs.teams:
        # 2v2: exactly one team left — pick first remaining as representative
        winner_id = remaining[0]
        gs.game_over = True
        gs.winner_id = winner_id
        winner_name = gs.players[winner_id].display_name
        gs.log.add(f"{loser_name}'s team concedes. {winner_name}'s team wins!")
        logs.append(f"{loser_name}'s team concedes. {winner_name}'s team wins!")
    else:
        # FFA with >2 remaining — remove player, game continues
        gs.player_order.remove(player_id)
        gs.log.add(f"{loser_name} concedes and is removed from the game.")
        logs.append(f"{loser_name} concedes and is removed from the game.")
        # If it was their turn, advance to next player
        if gs.turn_player_id == player_id:
            gs.turn_player_id = gs.player_order[0]
        if gs.active_player_id == player_id:
            gs.active_player_id = gs.turn_player_id

    return logs


def _exec_submit_choice(gs: GameState, player_id: str, payload: dict) -> list[str]:
    """Resume execution after a player submits their choice.

    Pops the pending_choice from GameState and resumes the IR walker
    with the player's selection applied.
    """
    from .effect_resolver import resolve_effect_ir

    pending = gs.pending_choice
    if pending is None:
        return ["No pending choice to submit"]

    if pending["controller_id"] != player_id:
        return ["Not your choice to make"]

    logs: list[str] = []

    chosen_option_index = payload.get("chosen_option_index")
    chosen_option_indices = payload.get("chosen_option_indices")
    chosen_target_ids = payload.get("chosen_target_ids", [])

    # Clear the pending choice before resolving (so nested choices can work)
    gs.pending_choice = None

    source_id = pending["source_instance_id"]
    source = gs.get_instance(source_id)
    if not source:
        return ["Choice source no longer exists"]

    pre_targets = pending.get("pre_targets", [])

    # Modal choice: player picked a single option index
    if chosen_option_index is not None:
        options = pending.get("options", [])
        if 0 <= chosen_option_index < len(options):
            chosen = options[chosen_option_index]
            effect = chosen.get("effect", chosen)
            result = resolve_effect_ir(effect, source, gs, pre_targets)
            logs.extend(result)
            logs.insert(0, f"Choice submitted: option {chosen_option_index}")
        else:
            logs.append(f"Invalid option index: {chosen_option_index}")

    # Multi-select modal choice: player picked multiple option indices
    elif chosen_option_indices is not None:
        options = pending.get("options", [])
        min_c = pending.get("min_choices", 1)
        max_c = pending.get("max_choices", len(options))
        if not (min_c <= len(chosen_option_indices) <= max_c):
            logs.append(
                f"Invalid selection count: got {len(chosen_option_indices)}, "
                f"expected {min_c}-{max_c}"
            )
        else:
            for idx in chosen_option_indices:
                if 0 <= idx < len(options):
                    chosen = options[idx]
                    effect = chosen.get("effect", chosen)
                    result = resolve_effect_ir(effect, source, gs, pre_targets)
                    logs.extend(result)
                else:
                    logs.append(f"Invalid option index: {idx}")
            logs.insert(0, f"Choice submitted: options {chosen_option_indices}")

    # Target-selection choice: player picked specific targets
    elif chosen_target_ids:
        # Validate target count
        min_c = pending.get("min_choices", 1)
        max_c = pending.get("max_choices", 1)
        if not (min_c <= len(chosen_target_ids) <= max_c):
            logs.append(
                f"Invalid target count: got {len(chosen_target_ids)}, "
                f"expected {min_c}-{max_c}"
            )
        else:
            logs.append(f"Targets selected: {chosen_target_ids}")
            # The remaining IR after the choice point may use these targets.
            # Callers higher up the chain should provide a continuation IR.
            # For now, targets are stored and the chain resolver will pick them up.

    else:
        logs.append("No choice provided")

    return logs
