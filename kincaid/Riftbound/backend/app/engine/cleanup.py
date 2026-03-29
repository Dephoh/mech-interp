"""The 9-step cleanup loop that runs after every state change."""

from __future__ import annotations

from .combat import open_showdown, start_combat
from .enums import CardType, CombatRole, ControlStatus, Phase, ZoneType
from .game_state import GameState
from .keywords import should_die_temporary
from .scoring import check_win
from .trigger_system import GameEvent, fire_event


def run_cleanup(gs: GameState) -> list[str]:
    """
    Run the full cleanup loop until stable. Returns all log messages generated.
    """
    all_logs: list[str] = []
    max_iterations = 100  # safety valve

    for _ in range(max_iterations):
        changed = False
        logs: list[str] = []

        # Step 1: Check win condition
        winner = check_win(gs)
        if winner:
            gs.game_over = True
            gs.winner_id = winner
            gs.phase = Phase.GAME_OVER
            logs.append(f"{gs.players[winner].display_name} wins!")
            all_logs.extend(logs)
            return all_logs

        # Step 2: Kill units with damage >= might (Deathknell first)
        killed = _kill_dead_units(gs, logs)
        if killed:
            changed = True

        # Step 3: Assign attacker/defender designations if combat in progress
        if gs.active_combat:
            _assign_combat_roles(gs)

        # Step 4: Battlefields with no units and no Contested → Uncontrolled
        if _reset_empty_battlefields(gs, logs):
            changed = True

        # Step 5: Recall gear from battlefields
        if _recall_gear(gs, logs):
            changed = True

        # Step 6: Remove facedown cards at battlefields not controlled by their controller
        _cleanup_facedown(gs, logs)

        # Step 7: Stage Showdowns at Contested uncontrolled battlefields
        if _stage_showdowns(gs, logs):
            changed = True

        # Step 8: Stage Combats at battlefields with units from 2 players
        if _stage_combats(gs, logs):
            changed = True

        # Step 9: Open next staged Showdown or Combat (only in Neutral Open)
        if (
            gs.active_showdown is None
            and gs.active_combat is None
            and gs.chain.is_empty
            and gs.phase == Phase.ACTION
        ):
            if _open_next_staged(gs, logs):
                changed = True

        all_logs.extend(logs)
        if not changed:
            break

    return all_logs


def _assign_combat_roles(gs: GameState) -> None:
    """Ensure attacker/defender roles are assigned to units at the combat battlefield."""
    combat = gs.active_combat
    if not combat:
        return
    bf = gs.battlefields.get(combat.battlefield_id)
    if not bf:
        return
    for uid in bf.units:
        unit = gs.instances.get(uid)
        if not unit:
            continue
        if unit.controller_id == combat.attacker_id:
            unit.combat_role = CombatRole.ATTACKER
        elif unit.controller_id == combat.defender_id:
            unit.combat_role = CombatRole.DEFENDER


def _kill_dead_units(gs: GameState, logs: list[str]) -> bool:
    """Kill all units with damage >= effective_might. Returns True if any died."""
    killed_any = False
    units_to_kill: list[str] = []

    # Find all units that should die
    for iid, inst in gs.instances.items():
        if (
            inst.card_type == CardType.UNIT
            and inst.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD)
            and inst.damage > 0
            and inst.damage >= inst.effective_might
            and inst.effective_might > 0
        ):
            units_to_kill.append(iid)

    # Phase 1: Fire death events for all dying units while still on board.
    # This lets trigger_system find on_death (self), on_friendly_death (allies),
    # and on_enemy_death (opponents) triggers and push them onto the chain.
    for iid in units_to_kill:
        inst = gs.instances[iid]
        killed_any = True
        ctx = {"card_id": iid, "player_id": inst.controller_id}
        # UNIT_DIED → on_death (self deathknell)
        evt_logs = fire_event(gs, GameEvent.UNIT_DIED, ctx)
        logs.extend(evt_logs)
        # FRIENDLY_DEATH → on_friendly_death (allies observe)
        evt_logs = fire_event(gs, GameEvent.FRIENDLY_DEATH, ctx)
        logs.extend(evt_logs)
        # ENEMY_DEATH → on_enemy_death (opponents observe)
        evt_logs = fire_event(gs, GameEvent.ENEMY_DEATH, ctx)
        logs.extend(evt_logs)

    # Phase 2: Remove dead units from board and move to trash.
    for iid in units_to_kill:
        inst = gs.instances[iid]
        _remove_from_board(gs, inst)

        inst.zone = ZoneType.TRASH
        inst.location_id = None
        inst.damage = 0
        inst.exhausted = False
        inst.buff_counter = False
        inst.stunned = False
        inst.combat_role = CombatRole.NONE
        inst.granted_keywords.clear()
        inst.might_modifiers.clear()
        gs.players[inst.owner_id].trash.append(iid)
        logs.append(f"{inst.name} is killed")

    return killed_any


def _remove_from_board(gs: GameState, inst) -> None:
    """Remove a card instance from its current board location."""
    iid = inst.instance_id
    pid = inst.controller_id

    if inst.zone == ZoneType.BASE:
        if iid in (gs.base_units.get(pid) or []):
            gs.base_units[pid].remove(iid)
        if iid in (gs.base_gear.get(pid) or []):
            gs.base_gear[pid].remove(iid)
        if iid in (gs.base_runes.get(pid) or []):
            gs.base_runes[pid].remove(iid)
    elif inst.zone == ZoneType.BATTLEFIELD:
        for bf in gs.battlefields.values():
            if iid in bf.units:
                bf.units.remove(iid)
                break


def _reset_empty_battlefields(gs: GameState, logs: list[str]) -> bool:
    """Battlefields with no units and no Contested status → Uncontrolled."""
    changed = False
    for bf in gs.battlefields.values():
        if (
            not bf.units
            and bf.control_status == ControlStatus.CONTROLLED
            and bf.contested_by is None
        ):
            bf.control_status = ControlStatus.UNCONTROLLED
            bf.controller_id = None
            logs.append(f"Battlefield becomes uncontrolled (no units)")
            changed = True
    return changed


def _recall_gear(gs: GameState, logs: list[str]) -> bool:
    """Recall any gear that ended up at a battlefield."""
    # In our model gear is always in base, so this is mainly a safety check
    return False


def _cleanup_facedown(gs: GameState, logs: list[str]) -> None:
    """Remove facedown cards at battlefields not controlled by the card's controller."""
    for bf in gs.battlefields.values():
        if bf.facedown_card:
            card = gs.instances.get(bf.facedown_card)
            if card and bf.controller_id != card.controller_id:
                # Remove facedown card → trash
                card.zone = ZoneType.TRASH
                card.facedown = False
                card.hidden_at_battlefield = None
                card.hidden_ready = False
                gs.players[card.owner_id].trash.append(card.instance_id)
                bf.facedown_card = None
                logs.append(f"Hidden card removed from battlefield (lost control)")


def _stage_showdowns(gs: GameState, logs: list[str]) -> bool:
    """Stage showdowns at newly-contested uncontrolled battlefields."""
    changed = False
    for bf_id, bf in gs.battlefields.items():
        if (
            bf.contested_by is not None
            and bf.control_status == ControlStatus.UNCONTROLLED
            and not bf.showdown_staged
            and not bf.combat_staged
        ):
            # Check: only the contesting player's units here (no combat needed)
            units_by_player = bf.units_by_player(gs.instances)
            if len(units_by_player) == 1:
                bf.showdown_staged = True
                changed = True
                logs.append(f"Showdown staged at {bf_id}")
    return changed


def _stage_combats(gs: GameState, logs: list[str]) -> bool:
    """Stage combats at battlefields with units from 2 opposing players."""
    changed = False
    for bf_id, bf in gs.battlefields.items():
        if bf.combat_staged:
            continue
        units_by_player = bf.units_by_player(gs.instances)
        if len(units_by_player) >= 2 and bf.contested_by is not None:
            bf.combat_staged = True
            bf.showdown_staged = False  # combat supersedes showdown
            changed = True
            logs.append(f"Combat staged at {bf_id}")
    return changed


def _open_next_staged(gs: GameState, logs: list[str]) -> bool:
    """Open the next staged showdown or combat."""
    # Prefer showdowns first (simpler resolution)
    for bf_id, bf in gs.battlefields.items():
        if bf.showdown_staged and not bf.combat_staged:
            bf.showdown_staged = False
            open_logs = open_showdown(gs, bf_id, bf.contested_by or gs.turn_player_id)
            logs.extend(open_logs)
            return True

    for bf_id, bf in gs.battlefields.items():
        if bf.combat_staged:
            bf.combat_staged = False
            combat_logs = start_combat(gs, bf_id)
            logs.extend(combat_logs)
            return True

    return False


def kill_temporary_units(gs: GameState) -> list[str]:
    """Kill units with the Temporary keyword at start of Beginning Phase."""
    logs: list[str] = []
    units_to_kill: list[str] = []

    for iid, inst in gs.instances.items():
        if (
            inst.card_type == CardType.UNIT
            and inst.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD)
            and inst.controller_id == gs.turn_player_id
            and should_die_temporary(inst)
        ):
            units_to_kill.append(iid)

    for iid in units_to_kill:
        inst = gs.instances[iid]
        _remove_from_board(gs, inst)
        inst.zone = ZoneType.TRASH
        inst.location_id = None
        inst.damage = 0
        inst.exhausted = False
        inst.buff_counter = False
        inst.combat_role = CombatRole.NONE
        inst.granted_keywords.clear()
        inst.might_modifiers.clear()
        gs.players[inst.owner_id].trash.append(iid)
        logs.append(f"{inst.name} is killed (Temporary)")

    return logs
