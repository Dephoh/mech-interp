"""Serialize GameState into per-player views with information hiding."""

from __future__ import annotations

from typing import Any

from ..engine.card_types import CardInstance
from ..engine.enums import CardType, Phase, ZoneType
from ..engine.game_state import GameState


def serialize_for_player(gs: GameState, player_id: str) -> dict[str, Any]:
    """
    Create a client-safe view of the game state for a specific player.
    Hides opponent's hand contents, deck orders, and facedown cards
    the player can't see.
    """
    opponent_id = gs.opponent_id(player_id)

    has_priority = (gs.active_player_id == player_id)
    is_turn_player = (gs.turn_player_id == player_id)
    turn_state = gs.get_turn_state()

    return {
        "game_id": gs.game_id,
        "phase": gs.phase.value,
        "turn_number": gs.turn_number,
        "turn_player_id": gs.turn_player_id,
        "active_player_id": gs.active_player_id,
        "has_priority": has_priority,
        "is_your_turn": is_turn_player,
        "can_play_cards": _can_player_play(gs, player_id),
        "game_over": gs.game_over,
        "winner_id": gs.winner_id,
        "you": _serialize_player(gs, player_id, is_self=True),
        "opponent": _serialize_player(gs, opponent_id, is_self=False),
        "battlefields": {
            bf_id: _serialize_battlefield(gs, bf, player_id)
            for bf_id, bf in gs.battlefields.items()
        },
        "chain": _serialize_chain(gs, player_id),
        "combat": _serialize_combat(gs),
        "showdown": _serialize_showdown(gs),
        "turn_state": turn_state.value,
        "log": [e["message"] for e in gs.log.entries[-20:]],
    }


def _can_player_play(gs: GameState, player_id: str) -> bool:
    """Check if this player is allowed to play cards right now."""
    if gs.game_over or gs.phase not in (
        Phase.ACTION, Phase.SHOWDOWN, Phase.COMBAT_DAMAGE, Phase.COMBAT_RESOLUTION
    ):
        return False
    turn_state = gs.get_turn_state()
    if turn_state.value == "neutral_open":
        return player_id == gs.turn_player_id
    if turn_state.value in ("neutral_closed", "showdown_closed"):
        # Can only play Reactions, and only if you have priority
        return player_id == gs.active_player_id
    if turn_state.value == "showdown_open":
        # Can play Action cards with focus
        sd = gs.active_showdown
        return sd is not None and sd.focus_player_id == player_id
    return False


def _serialize_player(
    gs: GameState, pid: str, *, is_self: bool
) -> dict[str, Any]:
    ps = gs.players[pid]
    data: dict[str, Any] = {
        "player_id": pid,
        "display_name": ps.display_name,
        "score": ps.score,
        "main_deck_count": len(ps.main_deck),
        "rune_deck_count": len(ps.rune_deck),
        "trash_count": len(ps.trash),
        "hand_count": len(ps.hand),
    }

    if is_self:
        # Show own hand with full card info
        data["hand"] = [_serialize_card(gs.instances[iid], visible=True) for iid in ps.hand]
        data["rune_pool"] = {
            "energy": ps.rune_pool.energy,
            "power": {d.value: v for d, v in ps.rune_pool.power.items()},
        }
    else:
        # Opponent's hand: count only, no card details
        data["hand"] = [{"instance_id": iid, "facedown": True} for iid in ps.hand]
        data["rune_pool"] = {
            "energy": ps.rune_pool.energy,
            "power": {d.value: v for d, v in ps.rune_pool.power.items()},
        }

    # Legend zone
    if ps.legend_zone:
        data["legend"] = _serialize_card(gs.instances[ps.legend_zone], visible=True)
    else:
        data["legend"] = None

    # Champion zone
    if ps.champion_zone:
        data["champion"] = _serialize_card(gs.instances[ps.champion_zone], visible=True)
    else:
        data["champion"] = None

    # Base units (always visible)
    data["base_units"] = [
        _serialize_card(gs.instances[uid], visible=True)
        for uid in gs.base_units.get(pid, [])
        if uid in gs.instances
    ]

    # Base gear (always visible)
    data["base_gear"] = [
        _serialize_card(gs.instances[gid], visible=True)
        for gid in gs.base_gear.get(pid, [])
        if gid in gs.instances
    ]

    # Rune board (always visible)
    data["rune_board"] = [
        _serialize_card(gs.instances[rid], visible=True)
        for rid in gs.base_runes.get(pid, [])
        if rid in gs.instances
    ]

    # Trash (visible to both players)
    data["trash"] = [
        _serialize_card(gs.instances[tid], visible=True)
        for tid in ps.trash
        if tid in gs.instances
    ]

    return data


def _serialize_battlefield(
    gs: GameState, bf, viewer_id: str
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "battlefield_id": bf.battlefield_id,
        "control_status": bf.control_status.value,
        "controller_id": bf.controller_id,
        "contested_by": bf.contested_by,
        "showdown_staged": bf.showdown_staged,
        "combat_staged": bf.combat_staged,
    }

    # Battlefield card info
    bf_card = gs.instances.get(bf.card_instance_id)
    if bf_card:
        data["name"] = bf_card.name
        data["card_id"] = bf_card.card_id

    # Units at battlefield (always visible)
    data["units"] = [
        _serialize_card(gs.instances[uid], visible=True)
        for uid in bf.units
        if uid in gs.instances
    ]

    # Facedown card: only visible to its controller
    if bf.facedown_card:
        fd_card = gs.instances.get(bf.facedown_card)
        if fd_card and fd_card.controller_id == viewer_id:
            data["facedown_card"] = _serialize_card(fd_card, visible=True)
        else:
            data["facedown_card"] = {"instance_id": bf.facedown_card, "facedown": True}
    else:
        data["facedown_card"] = None

    return data


def _serialize_chain(gs: GameState, viewer_id: str) -> dict[str, Any]:
    items = []
    for item in gs.chain.stack:
        entry: dict[str, Any] = {
            "item_id": item.item_id,
            "controller_id": item.controller_id,
        }
        if item.card_instance_id:
            card = gs.instances.get(item.card_instance_id)
            if card:
                entry["card"] = _serialize_card(card, visible=True)
        if item.ability_id:
            entry["ability_id"] = item.ability_id
            if item.source_instance_id:
                source = gs.instances.get(item.source_instance_id)
                if source:
                    entry["source_name"] = source.name
        items.append(entry)

    return {
        "items": items,
        "is_empty": gs.chain.is_empty,
    }


def _serialize_combat(gs: GameState) -> dict[str, Any] | None:
    if not gs.active_combat:
        return None
    c = gs.active_combat
    return {
        "battlefield_id": c.battlefield_id,
        "attacker_id": c.attacker_id,
        "defender_id": c.defender_id,
        "phase": c.phase,
        "attacker_assigned": c.attacker_assignment is not None,
        "defender_assigned": c.defender_assignment is not None,
    }


def _serialize_showdown(gs: GameState) -> dict[str, Any] | None:
    if not gs.active_showdown:
        return None
    sd = gs.active_showdown
    return {
        "battlefield_id": sd.battlefield_id,
        "initiator_id": sd.initiator_id,
        "focus_player_id": sd.focus_player_id,
        "is_combat_showdown": sd.is_combat_showdown,
    }


def _serialize_card(card: CardInstance, *, visible: bool) -> dict[str, Any]:
    """Serialize a card instance. If not visible, only expose instance_id."""
    if not visible:
        return {"instance_id": card.instance_id, "facedown": True}

    data: dict[str, Any] = {
        "instance_id": card.instance_id,
        "card_id": card.card_id,
        "name": card.name,
        "card_type": card.card_type.value,
        "owner_id": card.owner_id,
        "controller_id": card.controller_id,
        "zone": card.zone.value,
        "location_id": card.location_id,
        "domains": [d.value for d in card.definition.domains],
        "cost_energy": card.definition.cost_energy,
        "cost_power": {d.value: v for d, v in card.definition.cost_power},
        "text": card.definition.text,
        "facedown": card.facedown,
    }

    # Unit-specific fields
    if card.card_type == CardType.UNIT:
        data.update({
            "base_might": card.definition.base_might,
            "effective_might": card.effective_might,
            "combat_might": card.combat_might,
            "damage": card.damage,
            "exhausted": card.exhausted,
            "stunned": card.stunned,
            "buff_counter": card.buff_counter,
            "combat_role": card.combat_role.value,
            "hidden_at_battlefield": card.hidden_at_battlefield,
            "hidden_ready": card.hidden_ready,
            "entered_this_turn": card.entered_this_turn,
            "accelerated": card.accelerated,
        })

    # Gear-specific
    if card.card_type == CardType.GEAR:
        data["exhausted"] = card.exhausted
        data["might_bonus"] = card.definition.might_bonus
        data["attached_to"] = card.attached_to

    # Rune-specific
    if card.card_type == CardType.RUNE:
        data["exhausted"] = card.exhausted

    # Attachment tracking (units and gear can have attachments)
    if card.card_type in (CardType.UNIT, CardType.GEAR):
        data["attached_to"] = card.attached_to
        data["attached_cards"] = list(card.attached_cards)

    # Keywords
    data["keywords"] = [
        {"keyword": k.keyword.value, "value": k.value}
        for k in card.all_keywords()
    ]

    # Abilities (public text only)
    data["abilities"] = [
        {
            "ability_id": ab.ability_id,
            "ability_type": ab.ability_type.value,
            "timing": ab.timing,
            "text": ab.text,
        }
        for ab in card.definition.abilities
    ]

    return data
