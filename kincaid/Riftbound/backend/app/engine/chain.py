"""Chain management: LIFO stack for spells and abilities.

Rules references:
  - 326-336: Chain mechanics, resolution steps
  - 332:     Chain creation — creator becomes first Active Player
  - 333:     Finalize step — pending items finalized in append order (333.1.b)
  - 333.1.c: Add-resource/unit/gear abilities resolve immediately on finalize
  - 334:     Execute step — Active Player can play Reaction, pass priority
  - 335:     Pass step — all players passed → resolve
  - 336:     Resolve step — newest item resolves (LIFO)
  - 336.3:   After resolve, pending items → return to finalize
  - 154.3:   While resolving, no other items can be finalized or resolved;
             triggered abilities are deferred until resolution completes
  - 356.2:  Permanents finalize immediately (leave chain, enter board)
  - 356.3:  Spells linger on chain as Finalized Chain Items
  - 376.4.a.2: Play effects go on chain as Pending Items after permanent enters
"""

from __future__ import annotations

import logging

from .card_types import CardInstance
from .effects import resolve_effect
from .effect_resolver import resolve_effect_ir
from .enums import CardType, ZoneType
from .game_state import ChainItem, GameState
from .keywords import apply_accelerate
from .trigger_system import GameEvent, fire_event

logger = logging.getLogger("riftbound.chain")


# ---------------------------------------------------------------------------
# ChainState helpers — extends ChainState with resolution-lock bookkeeping
# without modifying game_state.py.  The 'resolving' flag and
# 'deferred_triggers' queue are set as dynamic attributes on the
# ChainState instance the first time they are needed.
# ---------------------------------------------------------------------------


def _is_resolving(chain_state: object) -> bool:
    """Check if the chain is currently in the middle of resolving an item."""
    return getattr(chain_state, "resolving", False)


def _set_resolving(chain_state: object, value: bool) -> None:
    """Set the resolution lock flag on the chain state."""
    chain_state.resolving = value  # type: ignore[attr-defined]


def _get_deferred(chain_state: object) -> list:
    """Get the deferred trigger queue, creating it if needed."""
    if not hasattr(chain_state, "deferred_triggers"):
        chain_state.deferred_triggers = []  # type: ignore[attr-defined]
    return chain_state.deferred_triggers  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Finalize step (Rule 333)
# ---------------------------------------------------------------------------

def finalize_pending_items(gs: GameState) -> list[str]:
    """Rule 333: Finalize all pending items on the chain in append order.

    Rule 333.1.b: Each item is finalized in the order it was appended.
    Marks pending items as finalized (pending=False).

    Returns log messages produced during finalization.
    """
    logs: list[str] = []

    # Walk the stack bottom-to-top (append order) to finalize pending items
    for item in gs.chain.stack:
        if not item.pending:
            continue

        item.pending = False
        logger.info("[CHAIN] Finalized item %s", item.item_id[:8])

        # Rule 333.1.c: Abilities that add resources, units, and gear resolve
        # immediately when finalized and do not progress to Step 2.
        # (Card-based items like spells linger; this applies to triggered
        #  abilities whose effects are purely additive.)
        # We don't auto-resolve here because distinguishing "add resource"
        # abilities from other triggered abilities requires effect-type
        # introspection that belongs in the card pipeline, not the chain.

    return logs


def push_card_to_chain(
    gs: GameState,
    card: CardInstance,
    controller_id: str,
    targets: list[str] | None = None,
    destination: dict | None = None,
) -> ChainItem:
    """
    Place a card onto the chain.

    Rule 332:   Creating the chain — if empty, creator is first Active Player.
    Rule 351:   Card goes on chain -> Closes State.
    Rule 356.2: Permanents finalize immediately (enter board).
    Rule 356.3: Spells linger as Finalized Chain Items.
    """
    # Rule 332.1: Track whether this push creates a new chain.
    chain_was_empty = gs.chain.is_empty

    item = ChainItem.create(
        controller_id=controller_id,
        card_instance_id=card.instance_id,
        targets=targets or [],
    )
    card.zone = ZoneType.CHAIN

    if card.card_type in (CardType.UNIT, CardType.GEAR):
        # Rule 356.2: Permanents finalize immediately — don't linger on chain.
        # They enter the board, then play-effect triggers go on chain (376.4.a.2).
        item.pending = False  # Mark finalized immediately
        logs = _finalize_permanent(gs, card, controller_id, destination=destination)
        for msg in logs:
            gs.log.add(msg)

        # Play-effect triggers will have been pushed to chain by _finalize_permanent.
        # If chain has items now, give priority to controller (they added triggers).
        if not gs.chain.is_empty:
            gs.active_player_id = controller_id

    else:
        # Rule 356.3: Spells linger on chain. Opponents can play Reactions.
        gs.chain.push(item)
        gs.log.add(f"{card.name} placed on the chain by {controller_id}")

        # Rule 333.1.b: Finalize the item (mark pending=False).
        item.pending = False

        # Fire SPELL_PLAYED event so board objects with "When you play a spell"
        # triggers can react. Their triggers go on chain on top of the spell.
        evt_logs = fire_event(gs, GameEvent.SPELL_PLAYED, {
            "card_id": card.instance_id,
            "player_id": controller_id,
        })
        for msg in evt_logs:
            gs.log.add(msg)

        # Rule 332.1 / 334: After finalization, controller gets priority.
        # If this created the chain, they are the first Active Player (332.1).
        # If adding to existing chain, they still get priority (334).
        gs.active_player_id = controller_id
        logger.info(
            "[CHAIN] Spell %s on chain%s. Priority -> %s (controller)",
            card.name,
            " (new chain)" if chain_was_empty else "",
            controller_id[:6],
        )

    return item


def push_ability_to_chain(
    gs: GameState,
    source: CardInstance,
    ability_id: str,
    controller_id: str,
    targets: list[str] | None = None,
) -> ChainItem:
    """Place an activated/triggered ability onto the chain.

    Rule 332: If the chain was empty, this creates a new chain and
    the controller becomes the first Active Player.
    Rule 334: After finalization, controller gets priority.
    """
    chain_was_empty = gs.chain.is_empty

    item = ChainItem.create(
        controller_id=controller_id,
        source_instance_id=source.instance_id,
        ability_id=ability_id,
        targets=targets or [],
    )
    gs.chain.push(item)
    # Mark as finalized immediately (abilities don't have a multi-step play process)
    item.pending = False
    gs.log.add(f"Ability of {source.name} placed on the chain")

    # Rule 332.1 / 334: Controller gets priority after adding to chain
    gs.active_player_id = controller_id
    logger.info(
        "[CHAIN] Ability of %s on chain%s. Priority -> %s",
        source.name,
        " (new chain)" if chain_was_empty else "",
        controller_id[:6],
    )
    return item


def pass_priority(gs: GameState, player_id: str) -> bool:
    """
    Player passes priority.
    
    Rule 335: If ALL players passed consecutively → proceed to Resolve (step 4).
    Rule 334.1.c.1: Priority passes to next Player in Turn Order.
    
    Returns True if the top item should now resolve.
    """
    gs.chain.passed_players.add(player_id)
    logger.info(
        "[CHAIN] %s passes priority. Passed: %s/%s",
        player_id[:6], len(gs.chain.passed_players), len(gs.player_order),
    )

    # In a 2-player game, if both have passed, resolve top
    if len(gs.chain.passed_players) >= len(gs.player_order):
        return True

    # Switch active player to next in turn order
    gs.active_player_id = gs.opponent_id(player_id)
    logger.info("[CHAIN] Priority → %s", gs.active_player_id[:6])
    return False


def resolve_top_item(gs: GameState) -> list[str]:
    """
    Rule 336: Resolve the newest item on the chain (LIFO).
    Execute its effects in their entirety.

    Rule 154.3: While resolving, no other items can be finalized or resolved.
    Triggered abilities that fire during resolution are deferred and pushed
    onto the chain only after resolution completes.

    After resolve:
    - Rule 336.2: If chain empty -> Open State, turn player gets priority
    - Rule 336.3: If chain not empty and pending items -> finalize them
    - Rule 336.4: If chain not empty, no pending -> controller of newest gets priority
    """
    if gs.chain.is_empty:
        return []

    item = gs.chain.pop()
    logs: list[str] = []

    # Rule 154.3: Set resolution lock -- prevents nested resolution and
    # ensures triggered abilities are deferred until resolution completes.
    _set_resolving(gs.chain, True)

    if item.card_instance_id:
        # Resolving a spell card
        card = gs.get_instance(item.card_instance_id)
        if card and card.card_type == CardType.SPELL:
            logs.extend(_resolve_spell(gs, card, item.targets))
        elif card and card.card_type == CardType.UNIT:
            # Shouldn't normally reach here -- permanents finalize in push_card_to_chain
            logs.extend(_finalize_permanent(gs, card, item.controller_id))
        elif card and card.card_type == CardType.GEAR:
            logs.extend(_finalize_permanent(gs, card, item.controller_id))
    elif item.ability_id and item.source_instance_id:
        # Resolving an ability
        source = gs.get_instance(item.source_instance_id)
        if source:
            logs.extend(_resolve_ability(gs, source, item.ability_id, item.targets))

    # Rule 154.3.a: Resolution complete -- release lock and flush deferred triggers.
    _set_resolving(gs.chain, False)
    flush_logs = _flush_deferred_triggers(gs)
    logs.extend(flush_logs)

    # Reset passes after resolution
    gs.chain.reset_passes()

    # Rule 336.3: If pending items exist after resolve, finalize them first.
    has_pending = any(i.pending for i in gs.chain.stack)
    if has_pending:
        logs.extend(finalize_pending_items(gs))

    if gs.chain.is_empty:
        # Rule 336.2: Chain empty -> Open State
        # Priority returns based on context:
        if gs.active_showdown:
            # Rule 343: Focus passes to next player after chain resolves during showdown
            gs.active_player_id = gs.active_showdown.focus_player_id
        else:
            # Neutral Open -- turn player gets priority
            gs.active_player_id = gs.turn_player_id
        logger.info("[CHAIN] Chain empty. Priority -> %s", gs.active_player_id[:6])
    else:
        # Rule 336.4: Controller of newest item gets priority
        newest = gs.chain.peek()
        if newest:
            gs.active_player_id = newest.controller_id
            logger.info(
                "[CHAIN] Chain not empty. Priority -> %s (newest item controller)",
                gs.active_player_id[:6],
            )

    return logs


def defer_or_push_trigger(gs: GameState, item: ChainItem) -> str:
    """Push a triggered ability to the chain, or defer it if currently resolving.

    Rule 154.3: While a spell or ability on the chain is resolving, no other
    spells or abilities can be finalized on the chain. Triggered abilities
    caused by the resolving item are deferred and flushed after resolution.

    Returns a log message describing what happened.
    """
    if _is_resolving(gs.chain):
        _get_deferred(gs.chain).append(item)
        logger.info(
            "[CHAIN] Trigger deferred (resolving): %s",
            item.ability_id or item.item_id[:8],
        )
        return f"Trigger deferred until resolution completes: {item.ability_id or item.item_id[:8]}"
    else:
        gs.chain.push(item)
        logger.info(
            "[CHAIN] Trigger pushed to chain: %s",
            item.ability_id or item.item_id[:8],
        )
        return f"Trigger pushed to chain: {item.ability_id or item.item_id[:8]}"


def _flush_deferred_triggers(gs: GameState) -> list[str]:
    """Push all deferred triggers onto the chain after resolution completes.

    Rule 154.3.a: Finish resolving all effects of a spell before addressing
    anything the spell may have triggered or caused through execution.
    """
    deferred = _get_deferred(gs.chain)
    if not deferred:
        return []

    logs: list[str] = []
    # Push deferred triggers in the order they were queued
    for item in deferred:
        gs.chain.push(item)
        logs.append(f"Deferred trigger pushed to chain: {item.ability_id or item.item_id[:8]}")
        logger.info(
            "[CHAIN] Deferred trigger flushed: %s",
            item.ability_id or item.item_id[:8],
        )
    deferred.clear()
    return logs


def _resolve_spell(gs: GameState, card: CardInstance, targets: list[str]) -> list[str]:
    """Execute a spell's effects, then send it to trash."""
    logs: list[str] = []

    for ability in card.definition.abilities:
        if ability.effect_ir:
            # New composable effect system
            result = resolve_effect_ir(ability.effect_ir, card, gs, targets)
            logs.extend(result)
        elif ability.effect_script:
            # Legacy named-function fallback
            result = resolve_effect(ability.effect_script, card, gs, targets)
            logs.extend(result)

    # Spell goes to owner's trash
    ps = gs.players[card.owner_id]
    card.zone = ZoneType.TRASH
    card.location_id = None
    ps.trash.append(card.instance_id)
    logs.append(f"{card.name} resolves and goes to trash")
    return logs


def _finalize_permanent(
    gs: GameState,
    card: CardInstance,
    controller_id: str,
    destination: dict | None = None,
) -> list[str]:
    """
    Rule 356.2: Place a unit or gear onto the board from the chain.
    Rule 352.2: Units enter at the chosen location (base or controlled battlefield).
    Rule 376.4.a.2: Play effects (triggered abilities) go on chain after entry.
    """
    logs: list[str] = []

    if card.card_type == CardType.UNIT:
        card.exhausted = True
        card.entered_this_turn = True
        # Accelerate: if paid, unit enters ready (rule 731)
        apply_accelerate(card, card.accelerated)
        card.accelerated = False
        ready_str = "ready" if not card.exhausted else "exhausted"

        # Rule 352.2/356.2.c: Place at chosen destination (base or controlled battlefield)
        dest_zone = (destination or {}).get("zone", "base")
        dest_id = (destination or {}).get("id")

        if dest_zone == "battlefield" and dest_id and dest_id in gs.battlefields:
            bf = gs.battlefields[dest_id]
            bf.units.append(card.instance_id)
            card.zone = ZoneType.BATTLEFIELD
            card.location_id = dest_id
            logs.append(f"{card.name} enters the battlefield ({ready_str})")
        else:
            # Default: enter base
            card.zone = ZoneType.BASE
            card.location_id = controller_id
            gs.base_units.setdefault(controller_id, []).append(card.instance_id)
            logs.append(f"{card.name} enters the base ({ready_str})")

    elif card.card_type == CardType.GEAR:
        # Rule 356.2.d: Gear enters base ready
        card.zone = ZoneType.BASE
        card.location_id = controller_id
        card.exhausted = False
        gs.base_gear.setdefault(controller_id, []).append(card.instance_id)
        logs.append(f"{card.name} enters the base (ready)")

    # Rule 376.4.a.2: Play effects and triggered abilities go on chain.
    # fire_event scans ALL board objects — handles both the card's own on_play
    # trigger and observer cards' "When you play a unit/gear" triggers.
    if card.card_type == CardType.UNIT:
        # UNIT_PLAYED covers on_play (self) + on_unit_played (observers)
        evt_logs = fire_event(gs, GameEvent.UNIT_PLAYED, {
            "card_id": card.instance_id,
            "player_id": controller_id,
        })
    else:
        # GEAR_ENTERED covers on_play (self) for gear permanents
        evt_logs = fire_event(gs, GameEvent.GEAR_ENTERED, {
            "card_id": card.instance_id,
            "player_id": controller_id,
        })
    logs.extend(evt_logs)

    return logs


def _resolve_ability(
    gs: GameState,
    source: CardInstance,
    ability_id: str,
    targets: list[str],
) -> list[str]:
    """Execute a triggered or activated ability."""
    for ability in source.definition.abilities:
        if ability.ability_id == ability_id:
            if ability.effect_ir:
                return resolve_effect_ir(ability.effect_ir, source, gs, targets)
            elif ability.effect_script:
                return resolve_effect(ability.effect_script, source, gs, targets)
    return [f"Ability {ability_id} not found on {source.name}"]
