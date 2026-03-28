"""Combat and Showdown resolution."""

from __future__ import annotations

from .enums import CardType, CombatRole, ControlStatus, ZoneType
from .game_state import (
    BattlefieldState,
    CombatState,
    GameState,
    ShowdownState,
)
from .keywords import check_lethal_before_next, check_tank_ordering
from .scoring import score_conquer
from .trigger_system import GameEvent, fire_event


def open_showdown(
    gs: GameState,
    battlefield_id: str,
    initiator_id: str,
    is_combat: bool = False,
) -> list[str]:
    """Open a showdown at a battlefield. Returns log messages."""
    gs.active_showdown = ShowdownState(
        battlefield_id=battlefield_id,
        initiator_id=initiator_id,
        focus_player_id=initiator_id,
        is_combat_showdown=is_combat,
    )
    gs.active_player_id = initiator_id
    gs.log.add(f"Showdown opened at {battlefield_id}")
    return [f"Showdown begins at {battlefield_id}"]


def pass_focus(gs: GameState, player_id: str) -> list[str]:
    """
    Player passes focus in a showdown. Returns log messages.
    If both players pass, the showdown closes.
    """
    sd = gs.active_showdown
    if not sd:
        return ["No active showdown"]

    sd.passed_players.add(player_id)
    logs: list[str] = []

    if len(sd.passed_players) >= len(gs.player_order):
        # Both passed — showdown ends
        logs.extend(close_showdown(gs))
    else:
        # Pass focus to opponent
        next_player = gs.opponent_id(player_id)
        sd.focus_player_id = next_player
        gs.active_player_id = next_player
        logs.append(f"Focus passes to {next_player}")

    return logs


def close_showdown(gs: GameState) -> list[str]:
    """Close the active showdown and resolve its outcome."""
    sd = gs.active_showdown
    if not sd:
        return []

    bf_id = sd.battlefield_id
    bf = gs.battlefields[bf_id]
    logs: list[str] = []

    if sd.is_combat_showdown:
        # Part of combat — proceed to combat damage step
        gs.active_showdown = None
        logs.append("Showdown closes — proceeding to combat damage")
        logs.extend(execute_combat_damage(gs))
    else:
        # Standalone showdown (uncontrolled battlefield)
        gs.active_showdown = None
        units_by_player = bf.units_by_player(gs.instances)

        if len(units_by_player) == 1:
            # One player's units remain — they conquer
            winner_id = list(units_by_player.keys())[0]
            bf.control_status = ControlStatus.CONTROLLED
            bf.controller_id = winner_id
            bf.contested_by = None
            logs.append(f"{winner_id} establishes control of {bf_id}")
            logs.extend(score_conquer(gs, winner_id, bf_id))
        elif len(units_by_player) >= 2:
            # Two players — this becomes a staged combat
            players = list(units_by_player.keys())
            bf.combat_staged = True
            logs.append(f"Two players at {bf_id} — combat staged")
        else:
            # No units — battlefield stays uncontrolled
            bf.control_status = ControlStatus.UNCONTROLLED
            bf.controller_id = None
            bf.contested_by = None
            logs.append(f"{bf_id} becomes uncontrolled")

        # Return to action phase
        gs.active_player_id = gs.turn_player_id

    return logs


def start_combat(gs: GameState, battlefield_id: str) -> list[str]:
    """Initialize combat at a battlefield between two players."""
    bf = gs.battlefields[battlefield_id]
    units_by_player = bf.units_by_player(gs.instances)
    players = list(units_by_player.keys())

    if len(players) < 2:
        return [f"Combat requires 2 players at {battlefield_id}"]

    # Determine attacker/defender
    attacker_id = bf.contested_by or players[0]
    defender_id = [p for p in players if p != attacker_id][0]

    gs.active_combat = CombatState(
        battlefield_id=battlefield_id,
        attacker_id=attacker_id,
        defender_id=defender_id,
        phase="showdown",
    )

    # Assign combat roles to units
    for uid in bf.units:
        unit = gs.instances.get(uid)
        if unit:
            if unit.controller_id == attacker_id:
                unit.combat_role = CombatRole.ATTACKER
            elif unit.controller_id == defender_id:
                unit.combat_role = CombatRole.DEFENDER

    bf.combat_staged = False
    logs = [f"Combat begins at {battlefield_id}: {attacker_id} attacks, {defender_id} defends"]

    # Fire attack/defend triggers for each unit (rule: triggers when designation gained)
    for uid in bf.units:
        unit = gs.instances.get(uid)
        if not unit:
            continue
        if unit.combat_role == CombatRole.ATTACKER:
            evt_logs = fire_event(gs, GameEvent.ATTACK_STARTED, {
                "card_id": uid, "player_id": attacker_id,
                "battlefield_id": battlefield_id,
            })
            logs.extend(evt_logs)
        elif unit.combat_role == CombatRole.DEFENDER:
            evt_logs = fire_event(gs, GameEvent.DEFEND_STARTED, {
                "card_id": uid, "player_id": defender_id,
                "battlefield_id": battlefield_id,
            })
            logs.extend(evt_logs)

    # Open the showdown sub-phase
    logs.extend(open_showdown(gs, battlefield_id, attacker_id, is_combat=True))
    return logs


def execute_combat_damage(gs: GameState) -> list[str]:
    """
    Combat damage step. If units remain on both sides, request damage assignments.
    Otherwise skip to resolution.
    """
    combat = gs.active_combat
    if not combat:
        return ["No active combat"]

    bf = gs.battlefields[combat.battlefield_id]
    attackers = [
        gs.instances[uid] for uid in bf.units
        if gs.instances[uid].controller_id == combat.attacker_id
    ]
    defenders = [
        gs.instances[uid] for uid in bf.units
        if gs.instances[uid].controller_id == combat.defender_id
    ]

    if not attackers or not defenders:
        # One side was eliminated during showdown — skip to resolution
        return resolve_combat(gs)

    # Need damage assignments from both players
    combat.phase = "waiting_assignments"

    # Auto-assign if there's only one target on either side
    attacker_total = sum(u.combat_might for u in attackers)
    defender_total = sum(u.combat_might for u in defenders)

    if len(defenders) == 1:
        combat.attacker_assignment = {defenders[0].instance_id: attacker_total}
    if len(attackers) == 1:
        combat.defender_assignment = {attackers[0].instance_id: defender_total}

    if combat.attacker_assignment and combat.defender_assignment:
        # Both auto-assigned — apply immediately
        return apply_combat_damage(gs)

    logs = [f"Combat damage step: awaiting damage assignments"]
    # Set active player to the one who needs to assign
    if not combat.attacker_assignment:
        gs.active_player_id = combat.attacker_id
    elif not combat.defender_assignment:
        gs.active_player_id = combat.defender_id
    return logs


def submit_damage_assignment(
    gs: GameState,
    player_id: str,
    assignment: dict[str, int],
) -> list[str]:
    """Player submits their damage assignment. Returns logs."""
    combat = gs.active_combat
    if not combat:
        return ["No active combat"]

    bf = gs.battlefields[combat.battlefield_id]
    logs: list[str] = []

    # Validate the assignment
    if player_id == combat.attacker_id:
        attackers = [
            gs.instances[uid] for uid in bf.units
            if gs.instances[uid].controller_id == combat.attacker_id
        ]
        defenders = [
            gs.instances[uid] for uid in bf.units
            if gs.instances[uid].controller_id == combat.defender_id
        ]
        total_might = sum(u.combat_might for u in attackers)
        validation = validate_assignment(total_might, assignment, defenders)
        if validation:
            return [f"Invalid damage assignment: {validation}"]
        combat.attacker_assignment = assignment
        logs.append(f"Attacker assigns damage")

    elif player_id == combat.defender_id:
        defenders = [
            gs.instances[uid] for uid in bf.units
            if gs.instances[uid].controller_id == combat.defender_id
        ]
        attackers = [
            gs.instances[uid] for uid in bf.units
            if gs.instances[uid].controller_id == combat.attacker_id
        ]
        total_might = sum(u.combat_might for u in defenders)
        validation = validate_assignment(total_might, assignment, attackers)
        if validation:
            return [f"Invalid damage assignment: {validation}"]
        combat.defender_assignment = assignment
        logs.append(f"Defender assigns damage")

    # If both assignments are in, apply damage
    if combat.attacker_assignment is not None and combat.defender_assignment is not None:
        logs.extend(apply_combat_damage(gs))

    return logs


def validate_assignment(
    total_damage: int,
    assignment: dict[str, int],
    targets: list,
) -> str | None:
    """Validate a damage assignment. Returns error string or None."""
    assigned_total = sum(assignment.values())
    if assigned_total != total_damage:
        return f"Must assign exactly {total_damage} damage (assigned {assigned_total})"

    for uid, dmg in assignment.items():
        if dmg < 0:
            return "Cannot assign negative damage"

    # Check Tank ordering
    if not check_tank_ordering(targets, assignment):
        return "Must assign lethal to Tank units before non-Tank units"

    # Check lethal-before-next rule
    if not check_lethal_before_next(targets, assignment):
        return "Must assign lethal to each unit before moving to the next"

    return None


def apply_combat_damage(gs: GameState) -> list[str]:
    """Apply damage from both sides simultaneously, then resolve combat."""
    combat = gs.active_combat
    if not combat:
        return []

    logs: list[str] = []

    # Apply attacker's damage to defenders
    if combat.attacker_assignment:
        for uid, dmg in combat.attacker_assignment.items():
            unit = gs.instances.get(uid)
            if unit and dmg > 0:
                unit.damage += dmg
                logs.append(f"{unit.name} takes {dmg} combat damage")

    # Apply defender's damage to attackers
    if combat.defender_assignment:
        for uid, dmg in combat.defender_assignment.items():
            unit = gs.instances.get(uid)
            if unit and dmg > 0:
                unit.damage += dmg
                logs.append(f"{unit.name} takes {dmg} combat damage")

    combat.phase = "resolution"
    logs.extend(resolve_combat(gs))
    return logs


def resolve_combat(gs: GameState) -> list[str]:
    """
    Combat resolution step:
    1. Heal all units (combat cleanup)
    2. Recall attackers if defenders remain
    3. Remove designations
    4. Determine control
    """
    combat = gs.active_combat
    if not combat:
        return []

    bf = gs.battlefields[combat.battlefield_id]
    logs: list[str] = []

    # Note: units killed by damage will be handled by cleanup
    # Here we do the combat-specific resolution after cleanup runs

    # Check who has units remaining (after damage is applied but before cleanup kills)
    attacker_units = [
        gs.instances[uid] for uid in list(bf.units)
        if uid in gs.instances and gs.instances[uid].controller_id == combat.attacker_id
    ]
    defender_units = [
        gs.instances[uid] for uid in list(bf.units)
        if uid in gs.instances and gs.instances[uid].controller_id == combat.defender_id
    ]

    # Heal all units at this battlefield
    for uid in list(bf.units):
        unit = gs.instances.get(uid)
        if unit:
            unit.heal()

    # Check which units survived (after healing, before checking kills — kills handled by cleanup)
    surviving_attackers = [u for u in attacker_units if u.is_alive]
    surviving_defenders = [u for u in defender_units if u.is_alive]

    # Recall attackers if defenders survive
    if surviving_defenders:
        for unit in surviving_attackers:
            _recall_unit(gs, unit)
            logs.append(f"{unit.name} recalled to base")

    # Remove combat designations
    for uid in list(bf.units):
        unit = gs.instances.get(uid)
        if unit:
            unit.combat_role = CombatRole.NONE

    # Determine control
    remaining_players = set()
    for uid in bf.units:
        unit = gs.instances.get(uid)
        if unit and unit.is_alive:
            remaining_players.add(unit.controller_id)

    if len(remaining_players) == 1:
        winner_id = remaining_players.pop()
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = winner_id
        bf.contested_by = None
        logs.append(f"{winner_id} takes control of the battlefield")
        logs.extend(score_conquer(gs, winner_id, combat.battlefield_id))
        # Fire combat win trigger
        evt_logs = fire_event(gs, GameEvent.COMBAT_WIN, {
            "player_id": winner_id,
            "battlefield_id": combat.battlefield_id,
        })
        logs.extend(evt_logs)
    elif len(remaining_players) == 0:
        bf.control_status = ControlStatus.UNCONTROLLED
        bf.controller_id = None
        bf.contested_by = None
        logs.append("Battlefield becomes uncontrolled")
    # If both sides remain (shouldn't happen after recall), leave contested

    # Clear combat state
    gs.active_combat = None
    gs.active_player_id = gs.turn_player_id
    logs.append("Combat resolved")
    return logs


def _recall_unit(gs: GameState, unit) -> None:
    """Recall a unit to its controller's base."""
    # Remove from battlefield
    for bf in gs.battlefields.values():
        if unit.instance_id in bf.units:
            bf.units.remove(unit.instance_id)
            break

    # Place in base
    unit.zone = ZoneType.BASE
    unit.location_id = unit.controller_id
    unit.exhausted = True  # recalled units stay exhausted
    gs.base_units.setdefault(unit.controller_id, []).append(unit.instance_id)
