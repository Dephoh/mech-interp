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
    """Deal N damage to target(s)."""
    amount = params.get("amount", 1)
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            target.damage += amount
            logs.append(f"{source.name} deals {amount} damage to {target.name}")
    return logs or ["No valid targets for damage"]


def prim_draw_cards(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Draw N cards for a player."""
    from .game_state import _draw_cards
    from .scoring import perform_burn_out

    count = params.get("count", 1)
    player = _get_player_id(gs, source, params.get("player", "controller"))
    logs = []

    ps = gs.players[player]
    if len(ps.main_deck) < count:
        # Burn out then draw
        perform_burn_out(gs, player)
        logs.append(f"{player} burned out")

    drawn = _draw_cards(gs, player, count)
    for did in drawn:
        card = gs.get_instance(did)
        name = card.name if card else "a card"
        logs.append(f"{gs.players[player].display_name} draws {name}")
    return logs or [f"{gs.players[player].display_name} draws {count} card(s)"]


def prim_give_might(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Give +N might to target (this turn or permanent)."""
    amount = params.get("amount", 1)
    duration = params.get("duration", "turn")
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
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
    """Place a buff counter on target (max 1 per unit, +1 Might)."""
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target and not target.buff_counter:
            target.buff_counter = True
            logs.append(f"{target.name} gains a buff (+1 Might)")
        elif target:
            logs.append(f"{target.name} already has a buff")
    return logs or ["No valid targets for buff"]


def prim_stun(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Stun target unit(s)."""
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            target.stunned = True
            logs.append(f"{target.name} is stunned")
    return logs or ["No valid targets for stun"]


def prim_heal(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Heal N damage from target (or 'all' to fully heal)."""
    amount = params.get("amount", "all")
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            if amount == "all":
                healed = target.damage
                target.damage = 0
            else:
                healed = min(target.damage, int(amount))
                target.damage -= healed
            logs.append(f"{target.name} healed {healed} damage")
    return logs or ["No valid targets for heal"]


def prim_kill(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Kill target unit(s) — set damage to lethal."""
    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            # Set damage to effective_might to trigger death in cleanup
            target.damage = max(target.effective_might, target.damage + 1)
            logs.append(f"{target.name} is killed by {source.name}")
    return logs or ["No valid targets for kill"]


def prim_move(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Move target to a destination zone/location."""
    from .enums import ZoneType

    dest = params.get("destination", {})
    dest_zone = dest.get("zone", "base")
    dest_loc = dest.get("location", "owner")
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
                target.zone = ZoneType.BATTLEFIELD
                target.location_id = bf_id
                gs.battlefields[bf_id].units.append(target.instance_id)
                logs.append(f"{target.name} moved to battlefield {bf_id}")

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
    """Banish target(s) — move to banishment zone."""
    from .enums import ZoneType

    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            _remove_from_current_zone(gs, target)
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
    """Recycle target(s) to bottom of their corresponding deck."""
    from .enums import CardType, ZoneType

    logs = []
    for tid in targets:
        target = gs.get_instance(tid)
        if target:
            _remove_from_current_zone(gs, target)
            ps = gs.players[target.owner_id]
            if target.card_type == CardType.RUNE:
                target.zone = ZoneType.RUNE_DECK
                target.exhausted = False
                ps.rune_deck.append(target.instance_id)
                logs.append(f"{target.name} recycled to rune deck")
            else:
                target.zone = ZoneType.MAIN_DECK
                target.damage = 0
                target.exhausted = False
                ps.main_deck.append(target.instance_id)
                logs.append(f"{target.name} recycled to main deck")
            target.location_id = None
    return logs or ["No valid targets for recycle"]


def prim_play_token(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Create and place token(s) on the board."""
    from .card_types import CardDefinition, CardInstance as CI, KeywordInstance
    from .enums import CardType, Keyword, SuperType, ZoneType

    name = params.get("name", "Recruit")
    might = params.get("might", 1)
    temporary = params.get("temporary", False)
    ready_on_enter = params.get("ready_on_enter", False)
    count = params.get("count", 1)
    token_type_str = params.get("token_type", "unit")
    enter_exhausted = params.get("exhausted", False)

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
        )

        token = CI.create(
            definition=token_def,
            owner_id=controller,
            zone=ZoneType.BASE,
            location_id=controller,
        )
        token.entered_this_turn = True
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
    """Channel N runes from rune deck to board."""
    from .enums import ZoneType

    count = params.get("count", 1)
    controller = source.controller_id
    ps = gs.players[controller]
    logs = []

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

    return logs or ["No runes to channel"]


def prim_attach(
    source: CardInstance, gs: GameState, params: dict, targets: list[str],
) -> list[str]:
    """Attach a gear to a unit."""
    if len(targets) < 2:
        return ["Need gear and unit targets for attach"]
    gear = gs.get_instance(targets[0])
    unit = gs.get_instance(targets[1])
    if gear and unit:
        gear.attached_to = unit.instance_id
        if gear.instance_id not in unit.attached_cards:
            unit.attached_cards.append(gear.instance_id)
        return [f"{gear.name} attached to {unit.name}"]
    return ["Invalid targets for attach"]


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
}
