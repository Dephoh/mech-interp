"""Combat and Showdown resolution."""

from __future__ import annotations

import logging

from .enums import CardType, CombatRole, ControlStatus, Keyword, ZoneType
from .game_state import (
    BattlefieldState,
    CombatState,
    GameState,
    ShowdownState,
)
from .keywords import (
    check_backline_protection,
    check_lethal_before_next,
    check_tank_ordering,
    get_valid_damage_targets,
    reveal_hidden_card,
)
from .scoring import score_conquer
from .trigger_system import GameEvent, fire_event

logger = logging.getLogger("riftbound.combat")


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
    If all players have passed once in sequence (rule 344.3.a),
    the showdown closes.
    """
    sd = gs.active_showdown
    if not sd:
        return ["No active showdown"]

    sd.passed_players.add(player_id)
    logs: list[str] = []

    if len(sd.passed_players) >= len(gs.player_order):
        # All passed in sequence — showdown ends (rule 344.3.a / 345)
        logs.extend(close_showdown(gs))
    else:
        # Pass focus to next player in turn order (rule 344.4)
        next_p = gs.next_player(player_id)
        # Skip players who already passed
        while next_p in sd.passed_players and next_p != player_id:
            next_p = gs.next_player(next_p)
        sd.focus_player_id = next_p
        gs.active_player_id = next_p
        logs.append(f"Focus passes to {next_p}")

    return logs


def showdown_action_taken(gs: GameState) -> None:
    """
    Called when a player plays a spell or activates an ability during a
    showdown (rule 344.1/344.2).  Resets the sequential-pass tracker so
    that all players must pass again in sequence before the showdown
    closes (rule 344.3.a).
    """
    sd = gs.active_showdown
    if sd:
        sd.passed_players.clear()


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
    logs: list[str] = []

    # ------------------------------------------------------------------
    # Step 0: Reveal hidden/facedown cards at this battlefield (rule 737)
    # Hidden cards are revealed BEFORE damage assignment so they
    # participate in combat.
    # ------------------------------------------------------------------
    logs.extend(_reveal_hidden_at_battlefield(gs, bf))

    units_by_player = bf.units_by_player(gs.instances)
    players = list(units_by_player.keys())

    if len(players) < 2:
        return logs + [f"Combat requires 2 players at {battlefield_id}"]

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
    # Rule 756-757: Backline units cannot be declared as attackers.
    for uid in bf.units:
        unit = gs.instances.get(uid)
        if unit:
            if unit.controller_id == attacker_id:
                if unit.has_keyword(Keyword.BACKLINE):
                    logger.info(
                        "[COMBAT] %s has Backline — cannot attack", unit.name,
                    )
                else:
                    unit.combat_role = CombatRole.ATTACKER
            elif unit.controller_id == defender_id:
                unit.combat_role = CombatRole.DEFENDER

    bf.combat_staged = False
    logs.append(f"Combat begins at {battlefield_id}: {attacker_id} attacks, {defender_id} defends")

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
    """Validate a damage assignment against all combat rules. Returns error string or None."""
    assigned_total = sum(assignment.values())
    if assigned_total != total_damage:
        return f"Must assign exactly {total_damage} damage (assigned {assigned_total})"

    for uid, dmg in assignment.items():
        if dmg < 0:
            return "Cannot assign negative damage"

    # --- Backline: cannot assign damage to Backline units while
    # non-Backline units remain without lethal damage (rule 443.1.d.5) ---
    valid_targets = get_valid_damage_targets(targets)
    for uid, dmg in assignment.items():
        if dmg > 0:
            target = next((u for u in targets if u.instance_id == uid), None)
            if target and target not in valid_targets:
                # Check if all valid (non-Backline) targets already have lethal
                all_valid_lethal = all(
                    assignment.get(vt.instance_id, 0) >= vt.effective_might - vt.damage
                    for vt in valid_targets
                )
                if not all_valid_lethal:
                    return "Cannot assign damage to Backline unit while other units remain"

    # --- Tank ordering (rule 443.1.d.5 / 443.1.d.6) ---
    if not check_tank_ordering(targets, assignment):
        return "Must assign lethal to Tank units before non-Tank units"

    # --- Lethal-before-next (rule 443.1.d.3) ---
    if not check_lethal_before_next(targets, assignment):
        return "Must assign lethal to each unit before moving to the next"

    # --- No-overkill (rule 443.1.d.4): a unit cannot receive more damage
    # than the minimum lethal unless no further valid targets remain ---
    units_with_damage = [
        u for u in targets if assignment.get(u.instance_id, 0) > 0
    ]
    units_without_damage = [
        u for u in targets if assignment.get(u.instance_id, 0) == 0
    ]
    for unit in units_with_damage:
        lethal = max(0, unit.effective_might - unit.damage)
        assigned = assignment.get(unit.instance_id, 0)
        if lethal > 0 and assigned > lethal:
            # Overkill — only allowed if no other assignable targets remain
            # "no further units remain" means every other unit already has
            # lethal assigned OR there are no other targets at all.
            others_needing = [
                u for u in targets
                if u.instance_id != unit.instance_id
                and assignment.get(u.instance_id, 0) < max(0, u.effective_might - u.damage)
                and max(0, u.effective_might - u.damage) > 0
            ]
            if others_needing:
                return (
                    f"Cannot assign more than lethal ({lethal}) to "
                    f"{unit.name} while other units can receive damage"
                )

    return None


def apply_combat_damage(gs: GameState) -> list[str]:
    """
    Apply damage from both sides simultaneously (rule 443.1.d.1.a),
    then fire DAMAGE_DEALT triggers, then resolve combat.
    """
    combat = gs.active_combat
    if not combat:
        return []

    logs: list[str] = []

    # Collect all (unit, damage, source_player) pairs first so the
    # actual application is simultaneous (rule 443.1.d.1.a).
    damage_pairs: list[tuple[str, int, str]] = []  # (target_uid, dmg, dealer_pid)

    if combat.attacker_assignment:
        for uid, dmg in combat.attacker_assignment.items():
            if dmg > 0:
                damage_pairs.append((uid, dmg, combat.attacker_id))

    if combat.defender_assignment:
        for uid, dmg in combat.defender_assignment.items():
            if dmg > 0:
                damage_pairs.append((uid, dmg, combat.defender_id))

    # Apply all damage simultaneously
    for uid, dmg, _ in damage_pairs:
        unit = gs.instances.get(uid)
        if unit:
            unit.damage += dmg
            logs.append(f"{unit.name} takes {dmg} combat damage")

    # Fire DAMAGE_DEALT triggers after all damage is applied (rule 443.1.e)
    for uid, dmg, dealer_pid in damage_pairs:
        unit = gs.instances.get(uid)
        if unit:
            evt_logs = fire_event(gs, GameEvent.DAMAGE_DEALT, {
                "card_id": uid,
                "player_id": dealer_pid,
                "damage": dmg,
                "source": "combat",
                "battlefield_id": combat.battlefield_id,
            })
            logs.extend(evt_logs)

    combat.phase = "resolution"
    logs.extend(resolve_combat(gs))
    return logs


def resolve_combat(gs: GameState) -> list[str]:
    """
    Combat resolution step:
    1. Kill units with lethal damage (BEFORE healing)
    2. Heal surviving units
    3. Recall surviving losers
    4. Remove combat designations
    5. Determine control
    """
    combat = gs.active_combat
    if not combat:
        return []

    bf = gs.battlefields[combat.battlefield_id]
    logs: list[str] = []

    # 1. Kill units with lethal damage BEFORE healing
    dead_units: list = []
    for uid in list(bf.units):
        unit = gs.instances.get(uid)
        if unit and unit.effective_might > 0 and unit.damage >= unit.effective_might:
            dead_units.append(unit)

    for unit in dead_units:
        # Fire death events while unit is still on the board
        ctx = {"card_id": unit.instance_id, "player_id": unit.controller_id}
        logs.extend(fire_event(gs, GameEvent.UNIT_DIED, ctx))
        logs.extend(fire_event(gs, GameEvent.FRIENDLY_DEATH, ctx))
        logs.extend(fire_event(gs, GameEvent.ENEMY_DEATH, ctx))
        # Detach gear — returns to owner's base
        for gear_id in list(unit.attached_cards):
            gear = gs.instances.get(gear_id)
            if gear:
                gear.attached_to = None
                gear.zone = ZoneType.BASE
                gear.location_id = gear.owner_id
                gs.base_gear.setdefault(gear.owner_id, []).append(gear_id)
                logs.append(f"{gear.name} detached (unit died in combat)")
        unit.attached_cards.clear()
        # Remove from battlefield, move to trash
        if unit.instance_id in bf.units:
            bf.units.remove(unit.instance_id)
        gs.unregister_card(unit.instance_id)
        unit.zone = ZoneType.TRASH
        unit.location_id = None
        unit.damage = 0
        unit.exhausted = False
        unit.buff_counter = False
        unit.stunned = False
        unit.combat_role = CombatRole.NONE
        unit.granted_keywords.clear()
        unit.might_modifiers.clear()
        gs.players[unit.owner_id].trash.append(unit.instance_id)
        logs.append(f"{unit.name} destroyed in combat")

    # 2. Heal surviving units at the battlefield
    for uid in list(bf.units):
        unit = gs.instances.get(uid)
        if unit:
            unit.heal()

    # 3. Determine surviving sides and recall losers
    surviving_attackers = [
        gs.instances[uid] for uid in bf.units
        if uid in gs.instances and gs.instances[uid].controller_id == combat.attacker_id
    ]
    surviving_defenders = [
        gs.instances[uid] for uid in bf.units
        if uid in gs.instances and gs.instances[uid].controller_id == combat.defender_id
    ]

    # Track whether we already fired COMBAT_WIN for a player to avoid duplicates
    combat_win_fired_for: str | None = None

    if surviving_defenders and surviving_attackers:
        # Both sides survive — attackers recall (rule 444.1.a.2)
        for unit in surviving_attackers:
            _recall_unit(gs, unit)
            # Rule 444.1.a.3: remove Attacker designation from recalled units
            unit.combat_role = CombatRole.NONE
            logs.append(f"{unit.name} recalled to base")
        # Defenders successfully repelled the attack — fire COMBAT_WIN
        combat_win_fired_for = combat.defender_id
        evt_logs = fire_event(gs, GameEvent.COMBAT_WIN, {
            "player_id": combat.defender_id,
            "battlefield_id": combat.battlefield_id,
        })
        logs.extend(evt_logs)
    elif surviving_defenders:
        pass  # attackers all dead — defenders hold the field (control handled below)
    elif surviving_attackers:
        pass  # defenders all dead — attackers take the field (control handled below)

    # 4. Remove combat designations from any remaining units (rule 444.1.a.3)
    for uid in list(bf.units):
        unit = gs.instances.get(uid)
        if unit:
            unit.combat_role = CombatRole.NONE

    # 5. Determine control (rule 444.2)
    remaining_players = set()
    for uid in bf.units:
        unit = gs.instances.get(uid)
        if unit:
            remaining_players.add(unit.controller_id)

    if len(remaining_players) == 1:
        winner_id = remaining_players.pop()
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = winner_id
        bf.contested_by = None
        logs.append(f"{winner_id} takes control of the battlefield")
        # Rule 444.2.b: Conquer if not already scored this turn
        logs.extend(score_conquer(gs, winner_id, combat.battlefield_id))
        # Fire COMBAT_WIN only if we haven't already fired it for this player
        if combat_win_fired_for != winner_id:
            evt_logs = fire_event(gs, GameEvent.COMBAT_WIN, {
                "player_id": winner_id,
                "battlefield_id": combat.battlefield_id,
            })
            logs.extend(evt_logs)
    elif len(remaining_players) == 0:
        # Rule 444.2.d: no units remain — battlefield becomes uncontrolled
        bf.control_status = ControlStatus.UNCONTROLLED
        bf.controller_id = None
        bf.contested_by = None
        logs.append("Battlefield becomes uncontrolled")

    # 6. Clean up any lingering facedown state at the combat battlefield
    if bf.facedown_card:
        fd_card = gs.instances.get(bf.facedown_card)
        if fd_card:
            fd_card.facedown = False
            fd_card.hidden_at_battlefield = None
            fd_card.hidden_ready = False
        bf.facedown_card = None

    # 7. Clear combat state
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
    # Rule 436.1: Recall preserves exhausted status, damage, buffs — don't change state
    gs.base_units.setdefault(unit.controller_id, []).append(unit.instance_id)


def _reveal_hidden_at_battlefield(
    gs: GameState,
    bf: BattlefieldState,
) -> list[str]:
    """
    Reveal all hidden/facedown cards at a battlefield when combat begins.

    Per rule 737: Hidden cards are revealed when combat starts at their
    battlefield. This must happen BEFORE damage assignment so that
    revealed units participate in combat.

    Handles two cases:
    1. Units in bf.units that have hidden_at_battlefield set
    2. The bf.facedown_card slot (a card hidden via the Hide action)
    """
    logs: list[str] = []

    # Case 1: Reveal any units already in the units list that are hidden
    for uid in list(bf.units):
        unit = gs.instances.get(uid)
        if unit and unit.hidden_at_battlefield:
            reveal_logs = reveal_hidden_card(unit, gs)
            logs.extend(reveal_logs)
            logger.info(
                "[COMBAT] Revealed hidden unit %s at %s",
                unit.name, bf.battlefield_id,
            )

    # Case 2: Reveal the battlefield's facedown card slot
    if bf.facedown_card:
        facedown = gs.instances.get(bf.facedown_card)
        if facedown:
            reveal_logs = reveal_hidden_card(facedown, gs)
            logs.extend(reveal_logs)
            logger.info(
                "[COMBAT] Revealed facedown card %s at %s",
                facedown.name, bf.battlefield_id,
            )
            # If it is a unit, add it to the battlefield's units list
            # so it participates in combat
            if (
                facedown.card_type == CardType.UNIT
                and facedown.instance_id not in bf.units
            ):
                bf.units.append(facedown.instance_id)
                facedown.zone = ZoneType.BATTLEFIELD
                facedown.location_id = bf.battlefield_id
                logs.append(
                    f"{facedown.name} enters the battlefield from facedown"
                )

    return logs
