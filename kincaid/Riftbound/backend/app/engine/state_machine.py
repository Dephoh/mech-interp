"""Turn phase state machine: drives the progression of phases."""

from __future__ import annotations

from .cleanup import kill_temporary_units, run_cleanup
from .enums import Phase, ZoneType
from .game_state import GameState, _draw_cards
from .scoring import perform_burn_out, score_hold


def advance_phase(gs: GameState) -> list[str]:
    """
    Advance to the next phase of the turn. Returns log messages.
    Only valid when chain is empty and no staged combats/showdowns.
    """
    logs: list[str] = []

    if gs.phase == Phase.SETUP_MULLIGAN:
        # Should not be called here — mulligan is handled separately
        return ["Mulligan not yet complete"]

    elif gs.phase == Phase.AWAKEN:
        logs.extend(_execute_awaken(gs))
        gs.phase = Phase.BEGINNING
        logs.extend(_execute_beginning(gs))
        # Auto-advance through Channel and Draw (non-interactive)
        gs.phase = Phase.CHANNEL
        logs.extend(_execute_channel(gs))
        gs.phase = Phase.DRAW
        logs.extend(_execute_draw(gs))
        gs.phase = Phase.ACTION
        gs.active_player_id = gs.turn_player_id
        logs.append(f"Action Phase begins for {gs.players[gs.turn_player_id].display_name}")

    elif gs.phase == Phase.ACTION:
        # Player is ending their turn
        gs.phase = Phase.END_OF_TURN
        logs.extend(_execute_end_of_turn(gs))
        # Move to next player's turn
        logs.extend(_start_next_turn(gs))

    else:
        logs.append(f"Cannot advance from phase {gs.phase.value}")

    return logs


def start_game(gs: GameState) -> list[str]:
    """Called after all mulligans are done. Starts the first turn."""
    gs.phase = Phase.AWAKEN
    gs.turn_number = 1
    gs.turn_player_id = gs.player_order[0]
    gs.active_player_id = gs.player_order[0]

    logs = [f"Turn 1 begins for {gs.players[gs.turn_player_id].display_name}"]
    # Execute the full start-of-turn sequence
    logs.extend(advance_phase(gs))
    return logs


def _execute_awaken(gs: GameState) -> list[str]:
    """Awaken Phase: ready all game objects controlled by turn player."""
    pid = gs.turn_player_id
    ps = gs.players[pid]
    logs: list[str] = []

    # Reset per-turn tracking
    ps.reset_turn_tracking()

    # Ready all units in base
    for uid in gs.base_units.get(pid, []):
        unit = gs.instances.get(uid)
        if unit:
            unit.exhausted = False

    # Ready all gear in base
    for gid in gs.base_gear.get(pid, []):
        gear = gs.instances.get(gid)
        if gear:
            gear.exhausted = False

    # Ready all runes on board
    for rid in gs.base_runes.get(pid, []):
        rune = gs.instances.get(rid)
        if rune:
            rune.exhausted = False

    # Ready units at battlefields
    for bf in gs.battlefields.values():
        for uid in bf.units:
            unit = gs.instances.get(uid)
            if unit and unit.controller_id == pid:
                unit.exhausted = False

    # Clear entered_this_turn for all units (needed for Temporary check in Beginning Phase)
    for inst in gs.instances.values():
        if inst.card_type.value == "unit" and inst.controller_id == pid:
            inst.entered_this_turn = False

    # Reset battlefield scored tracking for this turn
    for bf in gs.battlefields.values():
        if bf.scored_this_turn_by == pid:
            bf.scored_this_turn_by = None

    logs.append("Awaken: all game objects readied")
    return logs


def _execute_beginning(gs: GameState) -> list[str]:
    """Beginning Phase: kill Temporary units, then score Hold, then triggers."""
    pid = gs.turn_player_id
    logs: list[str] = []

    # Kill Temporary units before scoring
    temp_logs = kill_temporary_units(gs)
    logs.extend(temp_logs)

    # Run cleanup after killing temporaries
    cleanup_logs = run_cleanup(gs)
    logs.extend(cleanup_logs)

    if gs.game_over:
        return logs

    # Scoring Step: Hold
    hold_logs = score_hold(gs, pid)
    logs.extend(hold_logs)

    # Check win after scoring
    cleanup_logs = run_cleanup(gs)
    logs.extend(cleanup_logs)

    return logs


def _execute_channel(gs: GameState) -> list[str]:
    """Channel Phase: channel 2 runes (3 if player goes second on first turn)."""
    pid = gs.turn_player_id
    ps = gs.players[pid]
    logs: list[str] = []

    num_runes = 2
    if ps.is_first_turn and ps.goes_second:
        num_runes = 3

    for _ in range(num_runes):
        if ps.rune_deck:
            rune_id = ps.rune_deck.pop(0)
            rune = gs.instances[rune_id]
            rune.zone = ZoneType.RUNE_BOARD
            rune.location_id = pid
            rune.exhausted = False
            gs.base_runes.setdefault(pid, []).append(rune_id)
            logs.append(f"Channeled {rune.name}")
        else:
            logs.append("No runes left in rune deck")
            break

    return logs


def _execute_draw(gs: GameState) -> list[str]:
    """Draw Phase: draw 1 card, then empty ALL players' rune pools (rule 315.4.d)."""
    pid = gs.turn_player_id
    ps = gs.players[pid]
    logs: list[str] = []

    # First turn exception: player going first may still draw (per rules)
    # The rules say player going first draws normally; second player gets extra rune instead

    if not ps.main_deck:
        burn_logs = perform_burn_out(gs, pid)
        logs.extend(burn_logs)
        # Check win after burn out
        cleanup_logs = run_cleanup(gs)
        logs.extend(cleanup_logs)
        if gs.game_over:
            return logs

    drawn = _draw_cards(gs, pid, 1)
    if drawn:
        card = gs.instances.get(drawn[0])
        name = card.name if card else "a card"
        logs.append(f"Draw: {name}")
    else:
        logs.append("Cannot draw (deck empty after burn out)")

    # Rule 315.4.d: "As the Draw Phase ends, each player's Rune Pool empties."
    for p in gs.players.values():
        p.rune_pool.empty()
    logs.append("Rune pool emptied")

    return logs


def _execute_end_of_turn(gs: GameState) -> list[str]:
    """End of Turn Phase: heal all units, expire 'this turn' effects, empty rune pool."""
    pid = gs.turn_player_id
    ps = gs.players[pid]
    logs: list[str] = []

    # Heal all units on the board
    for inst in gs.instances.values():
        if inst.card_type.value == "unit" and inst.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD):
            inst.heal()

    # Expire "this turn" effects on all game objects
    for inst in gs.instances.values():
        if inst.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD, ZoneType.RUNE_BOARD):
            inst.clear_turn_state()

    # Empty all players' rune pools
    for p in gs.players.values():
        p.rune_pool.empty()

    # Mark first turn as done
    if ps.is_first_turn:
        ps.is_first_turn = False

    logs.append("End of turn: all units healed, effects expired, rune pools emptied")
    return logs


def _start_next_turn(gs: GameState) -> list[str]:
    """Advance to the next player's turn."""
    current_idx = gs.player_order.index(gs.turn_player_id)
    next_idx = (current_idx + 1) % len(gs.player_order)
    gs.turn_player_id = gs.player_order[next_idx]
    gs.active_player_id = gs.turn_player_id
    gs.turn_number += 1
    gs.phase = Phase.AWAKEN

    name = gs.players[gs.turn_player_id].display_name
    logs = [f"Turn {gs.turn_number} begins for {name}"]

    # Auto-execute Awaken through to Action
    logs.extend(advance_phase(gs))
    return logs
