"""Atomic effect operations called by the effect resolver.

Each primitive is a function:
    (source: CardInstance, gs: GameState, params: dict, resolved_targets: list[str]) -> list[str]

Params come from the IR node. resolved_targets are instance_ids already chosen by
the player or auto-resolved by the target system.
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .card_types import CardInstance
    from .game_state import GameState

logger = logging.getLogger("riftbound.effects")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_target(gs: GameState, targets: list[str], index: int = 0) -> CardInstance | None:
    """Safely get a target CardInstance from the resolved targets list."""
    if index >= len(targets):
        return None
    return gs.get_instance(targets[index])


def _resolve_amount(
    raw: int | str | float,
    source: CardInstance,
    target: CardInstance | None = None,
    *,
    all_targets: list[str] | None = None,
    gs: GameState | None = None,
) -> int:
    """Resolve a dynamic amount reference to a concrete int.

    Supports the following expression strings (case-sensitive):

    Source references:
        "source.might", "source_might"  -- source.effective_might
        "source.cost"                   -- source.definition.cost_energy
        "self.might"                    -- alias for source.effective_might

    Target references (uses the per-iteration ``target`` when available,
    otherwise the first entry in ``all_targets``):
        "target.might", "target_might"  -- target.effective_might
        "target.cost"                   -- target.definition.cost_energy

    Computed values:
        "current_might" -- the current target's effective_might (used by
            effects like "double this unit's might" where the target is
            the unit receiving the buff)
        "doubled"       -- 2x the target's pre-existing damage (placeholder;
            real doubling is handled by the replacement system, but this
            provides a numeric fallback)
        "reciprocal_might" -- the OTHER target's might; only meaningful
            when exactly 2 targets are provided (e.g. Clash of Giants).
            Falls back to target.effective_might when there is no pair.

    Anything else (including ``int`` / ``float``) is returned as-is (cast
    to ``int``).  Unknown strings log a warning and default to 0.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)

    if not isinstance(raw, str):
        return 0

    # --- Source references -------------------------------------------------
    if raw in ("source.might", "source_might", "self.might"):
        return source.effective_might

    if raw == "source.cost":
        return getattr(source.definition, "cost_energy", 0)

    # --- Target references -------------------------------------------------
    # Lazily resolve the target from all_targets if not provided directly
    _target = target
    if _target is None and all_targets and gs is not None:
        _target = gs.get_instance(all_targets[0]) if all_targets else None

    if raw in ("target.might", "target_might"):
        if _target is not None:
            return _target.effective_might
        return 0

    if raw == "target.cost":
        if _target is not None:
            return getattr(_target.definition, "cost_energy", 0)
        return 0

    # --- Computed values ---------------------------------------------------
    if raw == "current_might":
        # "Double my might" pattern: the target (or source if self-targeted)
        # is the unit whose might we read.
        ref = _target if _target is not None else source
        return ref.effective_might

    if raw == "doubled":
        # Numeric fallback for "double the damage" — real doubling should
        # be implemented via replacement effects, but when this expression
        # reaches a deal_damage primitive directly, interpret as 2x the
        # target's current damage mark (useful for damage-doubling traps).
        if _target is not None:
            return _target.damage * 2
        return 0

    if raw == "reciprocal_might":
        # Clash of Giants pattern: two targets, each takes the other's might.
        # When iterating per-target inside a primitive, _target is the one
        # RECEIVING damage.  We need the OTHER target's might.
        if all_targets and gs is not None and len(all_targets) >= 2:
            # Find the "other" target
            other_id = None
            if _target is not None:
                for tid in all_targets:
                    if tid != _target.instance_id:
                        other_id = tid
                        break
            if other_id:
                other = gs.get_instance(other_id)
                if other is not None:
                    return other.effective_might
        # Fallback: if we can't find a pair, use the target's own might
        if _target is not None:
            return _target.effective_might
        return 0

    # --- Unknown -----------------------------------------------------------
    logger.warning("Unrecognised dynamic amount '%s', defaulting to 0", raw)
    return 0


def _get_player_id(gs: GameState, source: CardInstance, player: str) -> str:
    """Resolve 'controller', 'opponent', 'owner' to a player_id."""
    if player == "controller":
        return source.controller_id
    elif player == "opponent":
        return gs.opponent_id(source.controller_id)
    elif player == "owner":
        return source.owner_id
    return player  # treat as literal player_id


def _remove_from_current_zone(gs: GameState, card: CardInstance) -> None:
    """Remove a card instance from whatever zone/list it currently occupies."""
    from .enums import ZoneType

    pid = card.controller_id
    iid = card.instance_id

    if card.zone == ZoneType.BASE:
        for zone_dict in (gs.base_units, gs.base_gear, gs.base_runes):
            lst = zone_dict.get(pid)
            if lst and iid in lst:
                lst.remove(iid)
                return
    elif card.zone == ZoneType.BATTLEFIELD:
        for bf in gs.battlefields.values():
            if iid in bf.units:
                bf.units.remove(iid)
                return
            if bf.facedown_card == iid:
                bf.facedown_card = None
                return
    elif card.zone == ZoneType.HAND:
        ps = gs.players.get(card.owner_id)
        if ps and iid in ps.hand:
            ps.hand.remove(iid)
            return
    elif card.zone == ZoneType.CHAIN:
        # Chain items handled separately
        return
    elif card.zone == ZoneType.TRASH:
        ps = gs.players.get(card.owner_id)
        if ps and iid in ps.trash:
            ps.trash.remove(iid)
            return
    elif card.zone == ZoneType.BANISHMENT:
        ps = gs.players.get(card.owner_id)
        if ps and iid in ps.banishment:
            ps.banishment.remove(iid)
            return


# ---------------------------------------------------------------------------
# Primitive implementations
# ---------------------------------------------------------------------------

def prim_deal_damage(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Deal N damage to target(s).

    Rule 404.1.b: Mark the specified amount of damage on the unit.
    Rule 142.2.a: If a unit has nonzero damage >= its Might, it is Killed.
    Lethal-damage kills are Passive Kills (rule 415.1.a.2) and are handled
    by the cleanup loop; we just mark damage here.
    """
    raw_amount = params.get("amount", 1)
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            amount = _resolve_amount(raw_amount, source, target, all_targets=targets, gs=gs)
            target.damage += amount
            logs.append(f"{source.name} deals {amount} damage to {target.name}")
            # Rule 142.2.a: flag lethal damage for log clarity
            if target.damage > 0 and target.damage >= target.effective_might > 0:
                logs.append(f"{target.name} has lethal damage ({target.damage}/{target.effective_might})")
    return logs or ["No valid targets for damage"]


def prim_draw_cards(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Draw N cards for a player.

    Rule 400.4: If a player attempts to draw more cards than available:
      400.4.a: Draw as many as possible.
      400.4.b: Perform a Burn Out (rule 418).
      400.4.c: Draw the remaining cards needed.
    Rule 418.3: If deck is STILL empty after burn-out (trash was empty too),
      another burn out occurs. This repeats until draws complete or an
      opponent wins.
    """
    from .game_state import _draw_cards
    from .scoring import check_win, perform_burn_out

    count = params.get("count", 1)
    player = _get_player_id(gs, source, params.get("player", "controller"))
    logs = []
    ps = gs.players[player]
    remaining = count

    while remaining > 0:
        if gs.game_over:
            break

        available = len(ps.main_deck)
        if available >= remaining:
            # Enough cards -- draw normally
            drawn = _draw_cards(gs, player, remaining)
            for did in drawn:
                card = gs.get_instance(did)
                name = card.name if card else "a card"
                logs.append(f"{gs.players[player].display_name} draws {name}")
            remaining = 0
        else:
            # Rule 400.4.a: draw as many as possible
            if available > 0:
                drawn = _draw_cards(gs, player, available)
                for did in drawn:
                    card = gs.get_instance(did)
                    name = card.name if card else "a card"
                    logs.append(f"{gs.players[player].display_name} draws {name}")
                remaining -= available

            # Rule 400.4.b: perform burn out
            bo_logs = perform_burn_out(gs, player)
            logs.extend(bo_logs)

            # Check if burn out caused a win
            winner = check_win(gs)
            if winner:
                gs.game_over = True
                gs.winner_id = winner
                logs.append(f"Game over: {gs.players[winner].display_name} wins")
                break

            # Rule 418.3: if deck is still empty, loop will burn out again
            if not ps.main_deck:
                # Deck still empty after burn-out (trash was also empty).
                # The loop will iterate again, causing another burn out.
                continue

            # Rule 400.4.c: draw remaining (handled by next loop iteration)

    return logs or [f"{gs.players[player].display_name} has no cards to draw"]


def prim_give_might(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Give +N might to target (this turn or permanent)."""
    raw_amount = params.get("amount", 1)
    duration = params.get("duration", "turn")
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            amount = _resolve_amount(raw_amount, source, target, all_targets=targets, gs=gs)
            if duration == "turn":
                target.might_modifiers.append(amount)
                logs.append(f"{target.name} gains +{amount} Might this turn")
            else:
                # Permanent might change not natively supported on CardInstance,
                # use a persistent modifier approach
                target.might_modifiers.append(amount)
                logs.append(f"{target.name} gains +{amount} Might")
    return logs or ["No valid targets for might boost"]


def prim_buff(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Place a buff counter on target (max 1 per unit, +1 Might).

    Rule 413.1.b: Place a Buff Counter on a unit if it does not have one.
    Rule 413.1.b.1: If it already has one, it does NOT get another.
    Rule 413.1.c: Units with a buff counter can still be chosen as targets
      for buff effects, but will NOT be buffed. This distinction matters
      for "if it was buffed this way" conditionals.

    Sets gs.last_effect_succeeded so conditional follow-ups (e.g., "Buff a
    unit. Then, if it was buffed this way, draw a card.") work correctly.
    """
    logs = []
    any_buffed = False
    for tid in targets:
        target = gs.get_instance(tid)
        if not target:
            continue
        if target.buff_counter:
            # Rule 413.1.c: valid target, but no new buff
            logs.append(f"{target.name} already has a buff (no effect)")
        else:
            # Rule 413.1.b: place the buff counter
            target.buff_counter = True
            any_buffed = True
            logs.append(f"{target.name} gains a buff (+1 Might)")
    gs.last_effect_succeeded = any_buffed
    return logs or ["No valid targets for buff"]


def prim_stun(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Stun target unit(s).

    Rule 410.1.a: Stunned is a binary state.
    Rule 410.1.a.1: A Stunned Unit cannot be Stunned again. It CAN still be
      chosen as a target for stun effects, but the stun does not happen.
      (Important for "when you stun" triggers -- they should NOT fire.)
    Rule 410.1.b: Stunned unit does not contribute might to combat (handled
      by CardInstance.combat_might).
    Rule 410.1.c: Stunned unit still requires full might of damage to be
      killed (handled by CardInstance.effective_might).
    """
    logs = []
    any_stunned = False
    for tid in targets:
        target = gs.get_instance(tid)
        if not target:
            continue
        if target.stunned:
            # Rule 410.1.a.1: already stunned -- valid target but no effect
            logs.append(f"{target.name} is already stunned (no effect)")
        else:
            target.stunned = True
            any_stunned = True
            logs.append(f"{target.name} is stunned")
    gs.last_effect_succeeded = any_stunned
    return logs or ["No valid targets for stun"]


def prim_heal(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Heal N damage from target (or 'all' to fully heal).

    Rule 405.1: Damage being cleared from Units is Healing.
    Rule 405.1.a: If Damage is cleared for any reason it is considered Healing.
    Rule 405.2: More than one Unit can be Healed at the same time.
    Rule 405.3: Healing is a Limited Action.

    Healing cannot reduce damage below 0 (implicit -- damage is a non-negative
    marker). Healing does not increase Might; it only clears damage marks.
    """
    raw_amount = params.get("amount", "all")
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if not target:
            continue
        if target.damage == 0:
            logs.append(f"{target.name} has no damage to heal")
            continue
        if raw_amount == "all":
            healed = target.damage
            target.damage = 0
        else:
            amount = _resolve_amount(raw_amount, source, target, all_targets=targets, gs=gs)
            healed = min(target.damage, amount)
            target.damage -= healed
        logs.append(f"{target.name} healed {healed} damage")
    return logs or ["No valid targets for heal"]


def prim_kill(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Kill target permanent(s) -- move directly from board to trash.

    Rule 415.1: Killing is the action of a Permanent going to the trash
      from the board.
    Rule 415.1.a.1: Active Kill -- instructed by a game effect (this case).
    Rule 415.2: When a permanent is killed it is placed directly in the
      trash from its place of origin.
    Rule 415.2.a: Only considered Killed if origin was on the board.
    Rule 415.2.b: Kill is NOT a subset of Move.
    Rule 415.1.a.1.b: Deathknell triggers fire on Active Kill.
    """
    from .enums import CardType, ZoneType
    from .trigger_system import GameEvent, fire_event

    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if not target:
            continue
        # Rule 415.2.a: only kill if on the board
        if target.zone not in (ZoneType.BASE, ZoneType.BATTLEFIELD, ZoneType.RUNE_BOARD):
            logs.append(f"{target.name} is not on the board (cannot be killed)")
            continue
        if target.card_type == CardType.UNIT:
            # Fire death events BEFORE removing from board (unit must still
            # be on the board for trigger_system to find it and its abilities).
            ctx = {"card_id": tid, "player_id": target.controller_id}
            evt_logs = fire_event(gs, GameEvent.UNIT_DIED, ctx)
            logs.extend(evt_logs)
            evt_logs = fire_event(gs, GameEvent.FRIENDLY_DEATH, ctx)
            logs.extend(evt_logs)
            evt_logs = fire_event(gs, GameEvent.ENEMY_DEATH, ctx)
            logs.extend(evt_logs)

            _kill_unit(gs, target)
            logs.append(f"{target.name} is killed by {source.name}")
        else:
            _kill_non_unit(gs, target)
            logs.append(f"{target.name} is destroyed by {source.name}")
    return logs or ["No valid targets for kill"]


def _kill_unit(gs: GameState, target: CardInstance) -> None:
    """Remove a unit from the board and move to trash (Active Kill).

    Per rule 415.2, kill moves directly to trash from its place of origin.
    We mark lethal damage so that is_alive returns False consistently with
    how the engine checks death state elsewhere.
    """
    from .enums import CombatRole, ZoneType
    tid = target.instance_id

    # Snapshot the might before clearing modifiers, so we can set
    # lethal damage after clearing transient state.
    base_might = target.definition.base_might

    # Detach any gear attached to this unit
    for gear_id in list(target.attached_cards):
        gear = gs.get_instance(gear_id)
        if gear:
            gear.attached_to = None
    target.attached_cards.clear()

    # Remove from board location
    _remove_from_current_zone(gs, target)

    # Unregister modifiers/replacements owned by this card
    gs.unregister_card(tid)

    # Move to trash, clearing transient board state
    target.zone = ZoneType.TRASH
    target.location_id = None
    target.exhausted = False
    target.stunned = False
    target.buff_counter = False
    target.combat_role = CombatRole.NONE
    target.granted_keywords.clear()
    target.might_modifiers.clear()
    target.aura_might_bonus = 0
    target.gear_might_bonus = 0

    # Mark lethal damage (after clearing modifiers, effective_might == base_might)
    target.damage = max(base_might, 1)

    gs.players[target.owner_id].trash.append(tid)


def _kill_non_unit(gs: GameState, target: CardInstance) -> None:
    """Remove a non-unit card (gear, rune) from the board and move to trash."""
    from .enums import CardType, ZoneType
    tid = target.instance_id
    pid = target.controller_id

    # If gear is attached, detach first
    if target.attached_to:
        host = gs.get_instance(target.attached_to)
        if host and tid in host.attached_cards:
            host.attached_cards.remove(tid)
        target.attached_to = None

    # Remove from board location
    if target.zone == ZoneType.BASE:
        if tid in (gs.base_gear.get(pid) or []):
            gs.base_gear[pid].remove(tid)
        if tid in (gs.base_runes.get(pid) or []):
            gs.base_runes[pid].remove(tid)

    gs.unregister_card(tid)
    target.zone = ZoneType.TRASH
    target.location_id = None
    target.exhausted = False
    target.granted_keywords.clear()
    target.might_modifiers.clear()
    gs.players[target.owner_id].trash.append(tid)


def prim_move(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Move target to a destination zone/location.

    Rule 424.1: A Permanent changing position on the Board is a Move.
    Rule 425.3: Only Units can Move.
    Rule 427.2: Units cannot Move to a Battlefield that already has units
      from 2 other players present (by any means).
    Rule 425.2.b: If a Move would be invalid, the unit Recalls to base
      instead (rule 432).
    """
    from .enums import ZoneType

    dest = params.get("destination", {})
    dest_zone = dest.get("zone", "base")
    dest_loc = dest.get("location", "owner")
    is_standard_move = params.get("standard_move", False)
    logs = []

    for tid in targets:
        target = gs.get_instance(tid)
        if not target:
            continue

        _remove_from_current_zone(gs, target)

        if dest_zone == "base":
            owner = target.owner_id if dest_loc == "owner" else target.controller_id
            target.zone = ZoneType.BASE
            target.location_id = owner
            gs.base_units.setdefault(owner, []).append(target.instance_id)
            logs.append(f"{target.name} moved to base")
        elif dest_zone == "battlefield":
            bf_id = dest_loc if dest_loc not in ("here", "owner") else source.location_id
            if bf_id and bf_id in gs.battlefields:
                bf = gs.battlefields[bf_id]
                # Rule 427.2: check for 2-other-player restriction
                players_at_bf = set()
                for uid in bf.units:
                    u = gs.get_instance(uid)
                    if u:
                        players_at_bf.add(u.controller_id)
                other_players = players_at_bf - {target.controller_id}
                if len(other_players) >= 2:
                    # Rule 425.2.b: invalid destination -- recall to base
                    owner = target.owner_id
                    target.zone = ZoneType.BASE
                    target.location_id = owner
                    gs.base_units.setdefault(owner, []).append(target.instance_id)
                    logs.append(
                        f"{target.name} cannot move to battlefield (2 other players "
                        f"present) -- recalled to base"
                    )
                else:
                    target.zone = ZoneType.BATTLEFIELD
                    target.location_id = bf_id
                    bf.units.append(target.instance_id)
                    logs.append(f"{target.name} moved to battlefield {bf_id}")
            else:
                # Invalid battlefield -- recall to base
                owner = target.owner_id
                target.zone = ZoneType.BASE
                target.location_id = owner
                gs.base_units.setdefault(owner, []).append(target.instance_id)
                logs.append(f"{target.name} moved to base (invalid battlefield)")

    return logs or ["No valid targets for move"]


def prim_ready(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Ready (un-exhaust) target(s)."""
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target and target.exhausted:
            target.exhausted = False
            logs.append(f"{target.name} is readied")
    return logs or ["No valid targets for ready"]


def prim_exhaust(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Exhaust target(s)."""
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target and not target.exhausted:
            target.exhausted = True
            logs.append(f"{target.name} is exhausted")
    return logs or ["No valid targets for exhaust"]


def prim_discard(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Discard N cards from a player's hand. Uses resolved targets if provided."""
    from .enums import ZoneType

    count = params.get("count", 1)
    player = _get_player_id(gs, source, params.get("player", "controller"))
    ps = gs.players[player]
    logs = []

    if targets:
        # Specific cards chosen to discard
        for tid in targets[:count]:
            card = gs.get_instance(tid)
            if card and card.instance_id in ps.hand:
                ps.hand.remove(card.instance_id)
                card.zone = ZoneType.TRASH
                card.location_id = None
                ps.trash.append(card.instance_id)
                logs.append(f"{card.name} discarded")
    else:
        # Random discard (auto-play fallback)
        to_discard = min(count, len(ps.hand))
        for _ in range(to_discard):
            if ps.hand:
                cid = random.choice(ps.hand)
                card = gs.get_instance(cid)
                if card:
                    ps.hand.remove(cid)
                    card.zone = ZoneType.TRASH
                    card.location_id = None
                    ps.trash.append(cid)
                    logs.append(f"{card.name} discarded")

    return logs or ["Nothing to discard"]


def prim_banish(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Banish target(s) -- move to banishment zone.

    Rule 414.1: Banishing places a card from any zone into Banishment.
    Rule 414.2: Placed directly into Banishment from its origin.
    Rule 414.2.a: Banish is NOT a subset of Kill.
    Rule 414.2.b: Banish is NOT a subset of Discard.
    """
    from .enums import ZoneType

    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if not target:
            continue
        was_on_board = target.zone in (
            ZoneType.BASE, ZoneType.BATTLEFIELD, ZoneType.RUNE_BOARD,
        )
        _remove_from_current_zone(gs, target)

        # If leaving the board, clear board-specific state and
        # unregister modifiers/replacements
        if was_on_board:
            gs.unregister_card(target.instance_id)
            target.damage = 0
            target.exhausted = False
            target.stunned = False
            target.buff_counter = False
            target.granted_keywords.clear()
            target.might_modifiers.clear()

        target.zone = ZoneType.BANISHMENT
        target.location_id = None
        ps = gs.players[target.owner_id]
        ps.banishment.append(target.instance_id)
        logs.append(f"{target.name} is banished")
    return logs or ["No valid targets for banish"]


def prim_counter(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Counter a spell on the chain — remove and send to trash."""
    from .enums import ZoneType

    logs = []
    for tid in targets:
        # Find matching chain item
        for i, item in enumerate(gs.chain.stack):
            if item.item_id == tid or item.card_instance_id == tid:
                countered = gs.chain.stack.pop(i)
                if countered.card_instance_id:
                    card = gs.get_instance(countered.card_instance_id)
                    if card:
                        card.zone = ZoneType.TRASH
                        ps = gs.players[card.owner_id]
                        ps.trash.append(card.instance_id)
                        logs.append(f"{card.name} was countered")
                else:
                    logs.append("Ability countered")
                break
    return logs or ["No valid targets for counter"]


def prim_return_to_hand(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Return target(s) to owner's hand."""
    from .enums import ZoneType

    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            _remove_from_current_zone(gs, target)
            target.zone = ZoneType.HAND
            target.location_id = None
            target.damage = 0
            target.exhausted = False
            target.stunned = False
            target.buff_counter = False
            target.combat_role = target.combat_role.__class__("none")
            ps = gs.players[target.owner_id]
            ps.hand.append(target.instance_id)
            logs.append(f"{target.name} returned to hand")
    return logs or ["No valid targets for return to hand"]


def prim_return_to_deck(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Return target(s) to owner's main deck (top or bottom)."""
    from .enums import ZoneType

    position = params.get("position", "bottom")
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            _remove_from_current_zone(gs, target)
            target.zone = ZoneType.MAIN_DECK
            target.location_id = None
            target.damage = 0
            target.exhausted = False
            ps = gs.players[target.owner_id]
            if position == "top":
                ps.main_deck.insert(0, target.instance_id)
            else:
                ps.main_deck.append(target.instance_id)
            logs.append(f"{target.name} returned to {position} of deck")
    return logs or ["No valid targets for return to deck"]


def prim_recycle(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Recycle target(s) to bottom of their corresponding deck.

    Rule 403.1.a: Main Deck cards are Recycled to the Main Deck.
    Rule 403.1.b: Runes are Recycled to the Rune Deck.
    Rule 403.1.c: Each player recycles to their OWN decks.
    Rule 403.5: If 2+ cards are Recycled to the Main Deck simultaneously,
      they are placed on the bottom in a random order.
    Rule 403.5.a: If 2+ cards are Recycled to the Rune Deck simultaneously,
      they are placed on the bottom in the owner's chosen order (here we
      preserve the target order as a proxy for player choice).
    """
    from .enums import CardType, ZoneType

    logs = []
    # Group targets by destination deck for simultaneous-recycle randomization
    main_deck_targets: list[CardInstance] = []
    rune_deck_targets: list[CardInstance] = []

    for tid in targets:
        target = gs.get_instance(tid)
        if not target:
            continue
        _remove_from_current_zone(gs, target)
        target.location_id = None
        if target.card_type == CardType.RUNE:
            target.zone = ZoneType.RUNE_DECK
            target.exhausted = False
            rune_deck_targets.append(target)
            logs.append(f"{target.name} recycled to rune deck")
        else:
            target.zone = ZoneType.MAIN_DECK
            target.damage = 0
            target.exhausted = False
            target.stunned = False
            target.buff_counter = False
            target.granted_keywords.clear()
            target.might_modifiers.clear()
            main_deck_targets.append(target)
            logs.append(f"{target.name} recycled to main deck")

    # Rule 403.5: randomize order of 2+ simultaneous main deck recycles
    if len(main_deck_targets) >= 2:
        random.shuffle(main_deck_targets)
    for target in main_deck_targets:
        ps = gs.players[target.owner_id]
        ps.main_deck.append(target.instance_id)

    # Rule 403.5.a: rune deck order is owner's choice (preserve target order)
    for target in rune_deck_targets:
        ps = gs.players[target.owner_id]
        ps.rune_deck.append(target.instance_id)

    return logs or ["No valid targets for recycle"]


def prim_play_token(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Create and place token(s) on the board.

    Rule 180: Tokens are created on the board and cannot exist elsewhere.
    Rule 180.1: If a token is put into any Non-Board Zone, it ceases to
      exist immediately after moving to its new zone.
    Rule 142.4: Units enter the Board exhausted (unless Accelerate).
    Rule 181.1: A 1[M] Recruit token is domainless with 1 Might and Recruit tag.
    Rule 181.2: A 3[M] Sprite token with Temporary has Fae tag and Temporary kw.
    """
    from .card_types import CardDefinition, CardInstance as CI, KeywordInstance
    from .enums import CardType, Keyword, SuperType, ZoneType

    name = params.get("name", "Recruit")
    might = params.get("might", 1)
    temporary = params.get("temporary", False)
    ready_on_enter = params.get("ready_on_enter", False)
    count = params.get("count", 1)
    token_type_str = params.get("token_type", "unit")
    enter_exhausted = params.get("exhausted", False)
    tags = tuple(params.get("tags", []))

    token_card_type = CardType.GEAR if token_type_str == "gear" else CardType.UNIT

    kw_list: list[KeywordInstance] = []
    if temporary:
        kw_list.append(KeywordInstance(keyword=Keyword.TEMPORARY, value=0))
    for kw_data in params.get("keywords", []):
        kw_list.append(KeywordInstance(
            keyword=Keyword(kw_data["keyword"]),
            value=kw_data.get("value", 0),
        ))

    controller = source.controller_id
    logs = []

    for _ in range(count):
        token_def = CardDefinition(
            card_id=f"token_{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}",
            name=name,
            card_type=token_card_type,
            supertypes=(SuperType.TOKEN,),
            base_might=might,
            keywords=tuple(kw_list),
            tags=tags,
        )

        token = CI.create(
            definition=token_def,
            owner_id=controller,
            zone=ZoneType.BASE,
            location_id=controller,
        )
        token.entered_this_turn = True

        # Rule 142.4: Units enter the Board exhausted
        if enter_exhausted:
            token.exhausted = True
        elif ready_on_enter:
            token.exhausted = False
        else:
            token.exhausted = True

        gs.instances[token.instance_id] = token
        if token_card_type == CardType.GEAR:
            gs.base_gear.setdefault(controller, []).append(token.instance_id)
        else:
            gs.base_units.setdefault(controller, []).append(token.instance_id)

        logs.append(f"{name} token created")

    player_name = gs.players[controller].display_name
    if count > 1:
        return [f"{count} {name} tokens created for {player_name}"]
    return [f"{name} token ({might} Might) created for {player_name}"]


def destroy_token_if_off_board(gs: GameState, token: CardInstance) -> bool:
    """Rule 180.1: If a token is in a non-board zone, it ceases to exist.

    Call this after any zone change that might put a token off-board.
    Returns True if the token was destroyed.
    """
    from .enums import SuperType, ZoneType

    if SuperType.TOKEN not in token.definition.supertypes:
        return False

    board_zones = {ZoneType.BASE, ZoneType.BATTLEFIELD, ZoneType.RUNE_BOARD}
    if token.zone in board_zones:
        return False

    # Token is off-board -- remove it from the game entirely
    tid = token.instance_id
    pid = token.owner_id
    ps = gs.players.get(pid)

    # Remove from whatever non-board zone it landed in
    if ps:
        for zone_list in (ps.hand, ps.trash, ps.banishment, ps.main_deck, ps.rune_deck):
            if tid in zone_list:
                zone_list.remove(tid)

    # Remove from game instances
    gs.instances.pop(tid, None)
    gs.unregister_card(tid)
    return True


def prim_add_energy(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Add energy to a player's rune pool."""
    amount = params.get("amount", 1)
    player = _get_player_id(gs, source, params.get("player", "controller"))
    gs.players[player].rune_pool.add_energy(amount)
    return [f"+{amount} Energy for {gs.players[player].display_name}"]


def prim_add_power(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Add domain power to a player's rune pool."""
    from .enums import Domain

    domain_str = params.get("domain", "any")
    amount = params.get("amount", 1)
    player = _get_player_id(gs, source, params.get("player", "controller"))

    if domain_str == "any":
        # Universal power — for now add to first domain of source card
        if source.definition.domains:
            domain = source.definition.domains[0]
        else:
            return ["No domain for power generation"]
    else:
        domain = Domain(domain_str)

    gs.players[player].rune_pool.add_power(domain, amount)
    return [f"+{amount} {domain.value} Power for {gs.players[player].display_name}"]


def prim_channel_rune(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Channel N runes from rune deck to board.

    Rule 417.1: Channeling takes one or more Runes from the top of a
      player's Rune Deck and puts them on the board.
    Rule 417.2: The effect may specify conditions (e.g., "channel 1 rune
      exhausted").
    Rule 417.4: Formatted as "Channel X rune(s)," optionally with
      conditions like "exhausted."

    If the rune deck has fewer runes than requested, channel as many
    as possible and report the shortfall.
    """
    from .enums import ZoneType

    count = params.get("count", 1)
    controller = source.controller_id
    ps = gs.players[controller]
    logs = []
    channeled = 0

    enter_exhausted = params.get("exhausted", False)

    for _ in range(count):
        if not ps.rune_deck:
            break
        rune_id = ps.rune_deck.pop(0)
        rune = gs.get_instance(rune_id)
        if rune:
            rune.zone = ZoneType.RUNE_BOARD
            rune.location_id = controller
            rune.exhausted = enter_exhausted
            gs.base_runes.setdefault(controller, []).append(rune_id)
            suffix = " (exhausted)" if enter_exhausted else ""
            logs.append(f"{rune.name} channeled{suffix}")
            channeled += 1

    if channeled < count:
        shortfall = count - channeled
        logs.append(f"Could not channel {shortfall} rune(s) (rune deck empty)")

    # Track whether we channeled the full amount for conditional effects
    gs.last_effect_succeeded = (channeled == count)

    return logs or ["No runes to channel"]


def prim_attach(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Attach a gear to a unit.

    Two modes:
      - Equip mode (1 target): source IS the gear, target is the unit.
      - General mode (2 targets): targets[0]=gear, targets[1]=unit.
    """
    from .trigger_system import GameEvent, fire_event

    if len(targets) >= 2:
        gear = gs.get_instance(targets[0])
        unit = gs.get_instance(targets[1])
    elif len(targets) == 1:
        # Equip mode: source is the gear being equipped
        gear = source
        unit = gs.get_instance(targets[0])
    else:
        return ["Need at least one target for attach"]

    if not gear or not unit:
        return ["Invalid targets for attach"]

    # Detach from previous unit if already attached
    if gear.attached_to:
        old_unit = gs.get_instance(gear.attached_to)
        if old_unit and gear.instance_id in old_unit.attached_cards:
            old_unit.attached_cards.remove(gear.instance_id)

    gear.attached_to = unit.instance_id
    if gear.instance_id not in unit.attached_cards:
        unit.attached_cards.append(gear.instance_id)

    # Fire EQUIP event for triggers
    fire_event(gs, GameEvent.EQUIP, {
        "card_id": gear.instance_id,
        "unit_id": unit.instance_id,
        "player_id": gear.controller_id,
    })

    return [f"{gear.name} attached to {unit.name}"]


def prim_detach(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Detach a gear from its unit."""
    logs = []
    for tid in targets:
        gear = gs.get_instance(tid)
        if gear and gear.attached_to:
            unit = gs.get_instance(gear.attached_to)
            if unit and gear.instance_id in unit.attached_cards:
                unit.attached_cards.remove(gear.instance_id)
            gear.attached_to = None
            logs.append(f"{gear.name} detached")
    return logs or ["Nothing to detach"]


def prim_score_points(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Score victory points for a player."""
    amount = params.get("amount", 1)
    player = _get_player_id(gs, source, params.get("player", "controller"))
    gs.players[player].score += amount
    return [f"{gs.players[player].display_name} scores {amount} point(s)"]


def prim_gain_xp(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Gain XP for the controller."""
    amount = params.get("amount", 1)
    ps = gs.players[source.controller_id]
    xp = getattr(ps, "xp", 0)
    ps.xp = xp + amount  # type: ignore[attr-defined]
    return [f"{ps.display_name} gains {amount} XP"]


def prim_spend_xp(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Spend XP from the controller."""
    amount = params.get("amount", 1)
    ps = gs.players[source.controller_id]
    xp = getattr(ps, "xp", 0)
    if xp >= amount:
        ps.xp = xp - amount  # type: ignore[attr-defined]
        return [f"{ps.display_name} spends {amount} XP"]
    return [f"Not enough XP (have {xp}, need {amount})"]


def prim_look_at_top(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Look at top N cards of main deck. May recycle (put on bottom)."""
    count = params.get("count", 1)
    may_recycle = params.get("may_recycle", True)
    ps = gs.players[source.controller_id]

    if not ps.main_deck:
        return ["Deck is empty"]

    looked = []
    for i in range(min(count, len(ps.main_deck))):
        card = gs.get_instance(ps.main_deck[i])
        if card:
            looked.append(card.name)

    logs = [f"Vision: top card(s): {', '.join(looked)}"]

    # Auto-play: recycle if card is high cost (heuristic) or don't recycle
    # In real play this would be a player choice; for auto-play we skip recycling
    if may_recycle:
        logs.append("(May recycle top card)")

    return logs


def prim_give_keyword(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Grant a keyword to target(s) for a duration."""
    from .enums import Keyword

    keyword_str = params.get("keyword", "")
    duration = params.get("duration", "turn")

    resolved = targets
    if not resolved and "target" in params:
        target_spec = params["target"]
        if target_spec.get("scope") == "self":
            resolved = [source.instance_id]

    logs = []
    for tid in resolved:
        target = gs.get_instance(tid)
        if not target:
            continue
        try:
            kw = Keyword(keyword_str)
        except ValueError:
            logs.append(f"Unknown keyword: {keyword_str}")
            continue
        if kw.value not in [gk for gk in target.granted_keywords]:
            target.granted_keywords.append(kw.value)
        logs.append(f"{target.name} gains {kw.value}" + (" this turn" if duration == "turn" else ""))
    return logs or ["No targets for keyword grant"]


def prim_restrict(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Apply a restriction to opponents (e.g., can't play cards this turn).

    Currently logged but not enforced in the validator (requires restriction
    tracking on GameState or PlayerState).
    """
    restriction = params.get("restriction", "unknown")
    scope = params.get("scope", "opponents")
    duration = params.get("duration", "turn")
    return [f"Restriction applied: {scope} {restriction} ({duration})"]


def prim_spend_energy(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Spend energy from a player's rune pool.

    Returns failure log if insufficient energy (does not spend partial).
    """
    amount = params.get("amount", 1)
    player = _get_player_id(gs, source, params.get("player", "controller"))
    ps = gs.players[player]
    if ps.rune_pool.energy >= amount:
        ps.rune_pool.energy -= amount
        gs.last_effect_succeeded = True
        return [f"{ps.display_name} spends {amount} Energy"]
    gs.last_effect_succeeded = False
    return [f"Not enough Energy (have {ps.rune_pool.energy}, need {amount})"]


def prim_spend_power(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Spend domain power from a player's rune pool.

    Returns failure log if insufficient power for the given domain.
    """
    from .enums import Domain

    amount = params.get("amount", 1)
    domain_str = params.get("domain", "any")
    player = _get_player_id(gs, source, params.get("player", "controller"))
    ps = gs.players[player]

    if domain_str == "any":
        # Spend from first available domain with enough power
        for dom, val in ps.rune_pool.power.items():
            if val >= amount:
                ps.rune_pool.power[dom] -= amount
                gs.last_effect_succeeded = True
                return [f"{ps.display_name} spends {amount} {dom.value} Power"]
        gs.last_effect_succeeded = False
        return [f"Not enough Power of any domain (need {amount})"]

    domain = Domain(domain_str)
    available = ps.rune_pool.power.get(domain, 0)
    if available >= amount:
        ps.rune_pool.power[domain] -= amount
        gs.last_effect_succeeded = True
        return [f"{ps.display_name} spends {amount} {domain.value} Power"]
    gs.last_effect_succeeded = False
    return [f"Not enough {domain.value} Power (have {available}, need {amount})"]


def prim_fight(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Direct fight between two units — each deals damage equal to its Might
    to the other simultaneously.

    Used by cards like "Marching Orders": choose a friendly unit and an
    enemy unit; they fight.
    """
    if len(targets) < 2:
        return ["fight requires 2 targets (unit and opponent)"]

    unit_a = gs.get_instance(targets[0])
    unit_b = gs.get_instance(targets[1])
    if not unit_a or not unit_b:
        return ["No valid targets for fight"]

    might_a = unit_a.effective_might
    might_b = unit_b.effective_might

    # Simultaneous damage
    unit_a.damage += might_b
    unit_b.damage += might_a

    logs = [
        f"{unit_a.name} ({might_a} Might) fights {unit_b.name} ({might_b} Might)",
        f"{unit_b.name} takes {might_a} damage from {unit_a.name}",
        f"{unit_a.name} takes {might_b} damage from {unit_b.name}",
    ]
    # Flag lethal damage for log clarity
    for u in (unit_a, unit_b):
        if u.damage > 0 and u.damage >= u.effective_might > 0:
            logs.append(f"{u.name} has lethal damage ({u.damage}/{u.effective_might})")
    return logs


def prim_challenge(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Challenge — force a friendly unit to fight an enemy unit.

    Mechanically identical to fight but semantically a forced engagement.
    The first target is the friendly unit issuing the challenge; the second
    is the enemy unit being challenged.
    """
    return prim_fight(source, gs, params, targets)


def prim_gain_control(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Gain control of a target permanent.

    Changes the controller_id of the target to the source's controller.
    The target stays in its current zone/location.
    """
    logs = []
    new_controller = source.controller_id
    for tid in targets:
        target = gs.get_instance(tid)
        if not target:
            continue
        old_controller = target.controller_id
        if old_controller == new_controller:
            logs.append(f"{target.name} already controlled by {gs.players[new_controller].display_name}")
            continue
        target.controller_id = new_controller
        logs.append(
            f"{gs.players[new_controller].display_name} gains control of {target.name} "
            f"(from {gs.players[old_controller].display_name})"
        )
    return logs or ["No valid targets for gain control"]


def prim_set_stat(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Set a stat on the source or target to a specific value.

    Supports stat: "cost", "might". Value can be a literal int or a
    dynamic expression string (currently only literal ints are resolved;
    complex expressions are logged as stubs).
    """
    stat = params.get("stat", "might")
    raw_value = params.get("value", 0)
    logs = []

    # Resolve value — support literal ints only for now
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, dict):
        # Dynamic expression — stub for now
        return [f"[STUB] set_stat {stat} with dynamic expression {raw_value}"]
    else:
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return [f"[STUB] set_stat {stat} to '{raw_value}' (unresolvable expression)"]

    effective_targets = targets if targets else [source.instance_id]
    for tid in effective_targets:
        target = gs.get_instance(tid)
        if not target:
            continue
        if stat == "might":
            # Set effective might via modifier: delta = desired - base
            base = target.definition.base_might
            delta = value - base
            target.might_modifiers.clear()
            target.might_modifiers.append(delta)
            logs.append(f"{target.name} Might set to {value}")
        elif stat == "cost":
            # Cost modification stored as attribute on the instance
            target.cost_override = value  # type: ignore[attr-defined]
            logs.append(f"{target.name} cost set to {value}")
        else:
            logs.append(f"[STUB] set_stat '{stat}' to {value} on {target.name}")
    return logs or ["No valid targets for set_stat"]


def prim_play(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Play a card from a non-hand zone (e.g., from banishment).

    This is a complex operation that involves cost payment, zone transfer,
    and entering the board. For now we implement the zone transfer and
    log the cost bypass.
    """
    from .enums import CardType, ZoneType

    ignore_cost = params.get("ignore_cost", False)
    logs = []

    for tid in targets:
        target = gs.get_instance(tid)
        if not target:
            continue

        origin_zone = target.zone
        _remove_from_current_zone(gs, target)

        # Place on board based on card type
        controller = source.controller_id
        target.controller_id = controller
        if target.card_type == CardType.UNIT:
            target.zone = ZoneType.BASE
            target.location_id = controller
            target.exhausted = True
            target.entered_this_turn = True  # type: ignore[attr-defined]
            gs.base_units.setdefault(controller, []).append(target.instance_id)
            cost_note = " (ignoring cost)" if ignore_cost else ""
            logs.append(f"{target.name} played from {origin_zone.value}{cost_note}")
        elif target.card_type == CardType.GEAR:
            target.zone = ZoneType.BASE
            target.location_id = controller
            gs.base_gear.setdefault(controller, []).append(target.instance_id)
            cost_note = " (ignoring cost)" if ignore_cost else ""
            logs.append(f"{target.name} played from {origin_zone.value}{cost_note}")
        else:
            # Spells and other types — put in hand as fallback
            target.zone = ZoneType.HAND
            target.location_id = None
            gs.players[controller].hand.append(target.instance_id)
            logs.append(f"{target.name} moved to hand from {origin_zone.value}")

    return logs or ["No valid targets for play"]


def prim_play_card(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Play a card from hand, optionally with cost reduction.

    Similar to 'play' but specifically from hand with cost modifications.
    """
    cost_reduction = params.get("cost_reduction", 0)

    # Delegate to prim_play with adjusted params
    play_params = dict(params)
    if cost_reduction > 0:
        play_params["ignore_cost"] = True  # simplified; full implementation would reduce cost
    return prim_play(source, gs, play_params, targets)


def prim_reveal_until(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Reveal cards from top of deck until a condition is met.

    Reveals cards one at a time until the filter matches, then the found
    card can be used by a sub-effect. Remaining revealed cards go to the
    bottom of the deck.
    """
    deck_spec = params.get("source", {})
    filter_spec = params.get("filter", {})

    player = _get_player_id(gs, source, deck_spec.get("scope", "controller"))
    ps = gs.players[player]
    logs = []
    revealed: list[str] = []
    found_card = None

    field = filter_spec.get("field", "")
    op = filter_spec.get("op", "eq")
    value = filter_spec.get("value", "")

    while ps.main_deck:
        cid = ps.main_deck.pop(0)
        card = gs.get_instance(cid)
        if not card:
            continue
        revealed.append(cid)
        logs.append(f"Revealed: {card.name}")

        # Check filter
        match = False
        if field == "card_type" and op == "eq":
            match = card.card_type.value == value
        elif field == "keyword" and op == "has":
            from .enums import Keyword
            try:
                kw = Keyword(value)
                match = card.has_keyword(kw)
            except ValueError:
                match = False
        elif field == "tag" and op == "has":
            match = value in card.definition.tags

        if match:
            found_card = cid
            break

    if found_card:
        found_inst = gs.get_instance(found_card)
        found_name = found_inst.name if found_inst else "unknown"
        logs.append(f"Found: {found_name}")
        # Remaining revealed (not found) go to bottom of deck
        for rid in revealed:
            if rid != found_card:
                ps.main_deck.append(rid)
    else:
        # No match found — put all back on bottom
        for rid in revealed:
            ps.main_deck.append(rid)
        logs.append("No matching card found in deck")

    return logs


def prim_revert_battlefield(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Revert a battlefield to its uncontrolled state.

    Clears the controller and sets control status back to uncontrolled.
    Used by effects like Brush that reset battlefield ownership.
    """
    from .enums import ControlStatus

    # Determine which battlefield to revert
    bf_id = params.get("battlefield_id", None)
    if not bf_id and source.location_id:
        bf_id = source.location_id

    if bf_id and bf_id in gs.battlefields:
        bf = gs.battlefields[bf_id]
        old_controller = bf.controller_id
        bf.control_status = ControlStatus.UNCONTROLLED
        bf.controller_id = None
        bf.contested_by = None
        name = old_controller or "no one"
        if old_controller and old_controller in gs.players:
            name = gs.players[old_controller].display_name
        return [f"Battlefield {bf_id} reverted to uncontrolled (was {name})"]

    return ["No valid battlefield to revert"]


def prim_trigger_keyword(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Trigger a keyword's effect (e.g., Conquer).

    This fires the keyword as if it had just been activated. The actual
    keyword resolution depends on the keyword system; for now we log the
    trigger and attempt to invoke any matching keyword handler.
    """
    keyword_str = params.get("keyword", "")
    if not keyword_str:
        return ["No keyword specified for trigger_keyword"]

    logs = [f"Keyword triggered: {keyword_str} (source: {source.name})"]

    # Attempt to resolve known keyword effects
    if keyword_str.lower() == "conquer":
        # Conquer: score 1 point for the controller
        ps = gs.players.get(source.controller_id)
        if ps:
            ps.score += 1
            logs.append(f"{ps.display_name} scores 1 point (Conquer)")
    else:
        logs.append(f"[STUB] trigger_keyword '{keyword_str}' handler not fully implemented")

    return logs


def prim_grant_activated_ability(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Grant an activated ability to target unit(s).

    The granted ability is stored on the CardInstance and becomes available
    for the controller to activate. Complex ability definitions are stored
    as-is and interpreted by the action system.
    """
    ability = params.get("ability", {})
    if not ability:
        return ["No ability specified for grant_activated_ability"]

    effective_targets = targets if targets else [source.instance_id]
    logs = []
    for tid in effective_targets:
        target = gs.get_instance(tid)
        if not target:
            continue
        # Store the granted ability on the instance for later use
        if not hasattr(target, "granted_abilities"):
            target.granted_abilities = []  # type: ignore[attr-defined]
        target.granted_abilities.append(ability)  # type: ignore[attr-defined]
        cost_desc = ability.get("cost", [])
        effect_desc = ability.get("effect", [])
        logs.append(f"{target.name} gains activated ability: cost={cost_desc}, effect={effect_desc}")
    return logs or ["No valid targets for grant_activated_ability"]


def prim_share_abilities(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Share abilities between units.

    Copies keywords and/or abilities from source units to the target.
    This is a complex effect used by cards like Heimerdinger. Currently
    implemented as a stub that logs intent.
    """
    target_spec = params.get("target", {})
    options = params.get("options", [])

    scope = target_spec.get("scope", "self") if isinstance(target_spec, dict) else "self"
    return [f"[STUB] share_abilities for {source.name} (scope={scope}, "
            f"{len(options)} option group(s)) — requires full keyword copy implementation"]


# ---------------------------------------------------------------------------
# Registry mapping node types to primitive functions
# ---------------------------------------------------------------------------

PRIMITIVE_DISPATCH: dict[str, Any] = {
    "deal_damage": prim_deal_damage,
    "draw_cards": prim_draw_cards,
    "give_might": prim_give_might,
    "buff": prim_buff,
    "stun": prim_stun,
    "heal": prim_heal,
    "kill": prim_kill,
    "move": prim_move,
    "ready": prim_ready,
    "exhaust": prim_exhaust,
    "discard": prim_discard,
    "banish": prim_banish,
    "counter": prim_counter,
    "return_to_hand": prim_return_to_hand,
    "return_to_deck": prim_return_to_deck,
    "recycle": prim_recycle,
    "play_token": prim_play_token,
    "add_energy": prim_add_energy,
    "add_power": prim_add_power,
    "channel_rune": prim_channel_rune,
    "attach": prim_attach,
    "detach": prim_detach,
    "score_points": prim_score_points,
    "gain_xp": prim_gain_xp,
    "spend_xp": prim_spend_xp,
    "look_at_top": prim_look_at_top,
    "give_keyword": prim_give_keyword,
    "restrict": prim_restrict,
    "spend_energy": prim_spend_energy,
    "spend_power": prim_spend_power,
    "fight": prim_fight,
    "challenge": prim_challenge,
    "gain_control": prim_gain_control,
    "set_stat": prim_set_stat,
    "play": prim_play,
    "play_card": prim_play_card,
    "reveal_until": prim_reveal_until,
    "revert_battlefield": prim_revert_battlefield,
    "trigger_keyword": prim_trigger_keyword,
    "grant_activated_ability": prim_grant_activated_ability,
    "share_abilities": prim_share_abilities,
}
