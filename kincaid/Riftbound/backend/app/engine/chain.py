"""Chain management: LIFO stack for spells and abilities.

Rules references:
  - 326-336: Chain mechanics, resolution steps
  - 333:     Finalize step — Add abilities resolve immediately (333.1.c)
  - 334:     Execute step — Active Player can play Reaction, pass priority
  - 335:     Pass step — all players passed → resolve
  - 336:     Resolve step — newest item resolves
  - 356.2:  Permanents finalize immediately (leave chain, enter board)
  - 356.3:  Spells linger on chain as Finalized Chain Items
  - 376.4.a.2: Play effects go on chain as Pending Items after permanent enters
"""

from __future__ import annotations

import logging

from .card_types import CardInstance
from .effects import resolve_effect
from .enums import CardType, ZoneType
from .game_state import ChainItem, GameState
from .keywords import apply_accelerate

logger = logging.getLogger("riftbound.chain")


def push_card_to_chain(
    gs: GameState,
    card: CardInstance,
    controller_id: str,
    targets: list[str] | None = None,
) -> ChainItem:
    """
    Place a card onto the chain.
    
    Rule 351: Card goes on chain → Closes State.
    Rule 356.2: Permanents finalize immediately (enter board).
    Rule 356.3: Spells linger as Finalized Chain Items.
    """
    item = ChainItem.create(
        controller_id=controller_id,
        card_instance_id=card.instance_id,
        targets=targets or [],
    )
    card.zone = ZoneType.CHAIN

    if card.card_type in (CardType.UNIT, CardType.GEAR):
        # Rule 356.2: Permanents finalize immediately — don't linger on chain.
        # They enter the board, then play-effect triggers go on chain (376.4.a.2).
        logs = _finalize_permanent(gs, card, controller_id)
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
        
        # Rule 334: After a card is finalized, controller gets priority first.
        # They may want to play additional Reactions on their own spell.
        gs.active_player_id = controller_id
        logger.info(
            "[CHAIN] Spell %s on chain. Priority → %s (controller)",
            card.name, controller_id[:6],
        )

    return item


def push_ability_to_chain(
    gs: GameState,
    source: CardInstance,
    ability_id: str,
    controller_id: str,
    targets: list[str] | None = None,
) -> ChainItem:
    """Place an activated/triggered ability onto the chain."""
    item = ChainItem.create(
        controller_id=controller_id,
        source_instance_id=source.instance_id,
        ability_id=ability_id,
        targets=targets or [],
    )
    gs.chain.push(item)
    gs.log.add(f"Ability of {source.name} placed on the chain")
    
    # Controller gets priority after adding to chain
    gs.active_player_id = controller_id
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
    
    After resolve:
    - Rule 336.2: If chain empty → Open State, turn player gets priority
    - Rule 336.4: If chain not empty → controller of newest item gets priority
    """
    if gs.chain.is_empty:
        return []

    item = gs.chain.pop()
    logs: list[str] = []

    if item.card_instance_id:
        # Resolving a spell card
        card = gs.get_instance(item.card_instance_id)
        if card and card.card_type == CardType.SPELL:
            logs.extend(_resolve_spell(gs, card, item.targets))
        elif card and card.card_type == CardType.UNIT:
            # Shouldn't normally reach here — permanents finalize in push_card_to_chain
            logs.extend(_finalize_permanent(gs, card, item.controller_id))
        elif card and card.card_type == CardType.GEAR:
            logs.extend(_finalize_permanent(gs, card, item.controller_id))
    elif item.ability_id and item.source_instance_id:
        # Resolving an ability
        source = gs.get_instance(item.source_instance_id)
        if source:
            logs.extend(_resolve_ability(gs, source, item.ability_id, item.targets))

    # Reset passes after resolution
    gs.chain.reset_passes()

    if gs.chain.is_empty:
        # Rule 336.2: Chain empty → Open State
        # Priority returns based on context:
        if gs.active_showdown:
            # Rule 343: Focus passes to next player after chain resolves during showdown
            gs.active_player_id = gs.active_showdown.focus_player_id
        else:
            # Neutral Open — turn player gets priority
            gs.active_player_id = gs.turn_player_id
        logger.info("[CHAIN] Chain empty. Priority → %s", gs.active_player_id[:6])
    else:
        # Rule 336.4: Controller of newest item gets priority
        newest = gs.chain.peek()
        if newest:
            gs.active_player_id = newest.controller_id
            logger.info(
                "[CHAIN] Chain not empty. Priority → %s (newest item controller)",
                gs.active_player_id[:6],
            )

    return logs


def _resolve_spell(gs: GameState, card: CardInstance, targets: list[str]) -> list[str]:
    """Execute a spell's effects, then send it to trash."""
    logs: list[str] = []

    for ability in card.definition.abilities:
        if ability.effect_script:
            result = resolve_effect(ability.effect_script, card, gs, targets)
            logs.extend(result)

    # Spell goes to owner's trash
    ps = gs.players[card.owner_id]
    card.zone = ZoneType.TRASH
    card.location_id = None
    ps.trash.append(card.instance_id)
    logs.append(f"{card.name} resolves and goes to trash")
    return logs


def _finalize_permanent(gs: GameState, card: CardInstance, controller_id: str) -> list[str]:
    """
    Rule 356.2: Place a unit or gear onto the board from the chain.
    Rule 376.4.a.2: Play effects (triggered abilities) go on chain after entry.
    """
    logs: list[str] = []

    if card.card_type == CardType.UNIT:
        # Rule 356.2.c: Units enter base exhausted by default
        card.zone = ZoneType.BASE
        card.location_id = controller_id
        card.exhausted = True
        card.entered_this_turn = True
        # Accelerate: if paid, unit enters ready (rule 731)
        apply_accelerate(card, card.accelerated)
        card.accelerated = False
        gs.base_units.setdefault(controller_id, []).append(card.instance_id)
        ready_str = "ready" if not card.exhausted else "exhausted"
        logs.append(f"{card.name} enters the base ({ready_str})")

    elif card.card_type == CardType.GEAR:
        # Rule 356.2.d: Gear enters base ready
        card.zone = ZoneType.BASE
        card.location_id = controller_id
        card.exhausted = False
        gs.base_gear.setdefault(controller_id, []).append(card.instance_id)
        logs.append(f"{card.name} enters the base (ready)")

    # Rule 376.4.a.2: Play effects go on chain as Pending Items
    for ability in card.definition.abilities:
        if ability.trigger_condition == "on_play" and ability.effect_script:
            # Push play-effect trigger onto chain
            play_item = ChainItem.create(
                controller_id=controller_id,
                source_instance_id=card.instance_id,
                ability_id=ability.ability_id,
            )
            gs.chain.push(play_item)
            logs.append(f"{card.name} play effect triggers")
            logger.info("[CHAIN] Play effect of %s goes on chain", card.name)

    return logs


def _resolve_ability(
    gs: GameState,
    source: CardInstance,
    ability_id: str,
    targets: list[str],
) -> list[str]:
    """Execute a triggered or activated ability."""
    for ability in source.definition.abilities:
        if ability.ability_id == ability_id and ability.effect_script:
            return resolve_effect(ability.effect_script, source, gs, targets)
    return [f"Ability {ability_id} not found on {source.name}"]
