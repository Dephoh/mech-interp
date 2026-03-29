"""Build a GameState from a ScenarioDef for testlab play."""

from __future__ import annotations

import logging

from ..engine.card_db import CardDB
from ..engine.card_types import CardInstance
from ..engine.enums import CardType, ControlStatus, Domain, ZoneType
from ..engine.game_state import GameState, create_game, _draw_cards
from ..engine.state_machine import start_game
from .scenarios import ScenarioDef

logger = logging.getLogger("riftbound.testlab.builder")

# Default filler unit to pad decks
_FILLER_UNIT = "ogn-097-298"  # Blastcone Fae
_LEGEND_ID = "unl-230-star-219"
_RUNE_IDS = [
    "ogn-007a-298", "ogn-007a-298",
    "ogn-042a-298", "ogn-042a-298",
    "ogn-089a-298", "ogn-089a-298",
    "ogn-126a-298", "ogn-126a-298",
    "ogn-166a-298", "ogn-166a-298",
    "ogn-214a-298", "ogn-214a-298",
]
_BATTLEFIELD_IDS = ["unl-205-219", "unl-206-219", "ogn-275-298"]


def _pick_champion(card_db: dict, scenario: ScenarioDef) -> str:
    """Pick a champion: first unit in the scenario or a fallback."""
    all_unit_ids = (
        scenario.p1_hand
        + scenario.p1_base_units
        + [cid for ids in scenario.p1_bf_units.values() for cid in ids]
    )
    for cid in all_unit_ids:
        defn = card_db.get(cid)
        if defn and defn.card_type == CardType.UNIT:
            return cid
    return _FILLER_UNIT


def _build_main_deck(scenario: ScenarioDef, champion_id: str) -> list[str]:
    """Build a 40-card main deck containing scenario cards plus filler."""
    cards: list[str] = []
    # Include all scenario card_ids
    cards.extend(scenario.p1_hand)
    cards.extend(scenario.p1_base_units)
    for ids in scenario.p1_bf_units.values():
        cards.extend(ids)
    # Pad to 40 with filler
    while len(cards) < 40:
        cards.append(_FILLER_UNIT)
    return cards[:40]


def build_game_from_scenario(scenario: ScenarioDef) -> GameState:
    """Create a fully-initialized GameState from a scenario definition."""
    card_db = CardDB.all_cards()
    champion_id = _pick_champion(card_db, scenario)

    p1_deck = _build_main_deck(scenario, champion_id)

    # P2 gets a simple filler deck
    p2_deck = [_FILLER_UNIT] * 40

    configs = [
        {
            "player_id": "testlab-p1",
            "display_name": "Test Player",
            "legend_id": _LEGEND_ID,
            "champion_id": champion_id,
            "main_deck": p1_deck,
            "rune_deck": list(_RUNE_IDS),
            "battlefields": list(_BATTLEFIELD_IDS),
        },
        {
            "player_id": "testlab-p2",
            "display_name": "Bot",
            "legend_id": _LEGEND_ID,
            "champion_id": _FILLER_UNIT,
            "main_deck": p2_deck,
            "rune_deck": list(_RUNE_IDS),
            "battlefields": list(_BATTLEFIELD_IDS),
        },
    ]

    gs = create_game("testlab", configs, card_db)

    # Force P1 to go first so the tester always has the active turn
    gs.player_order = ["testlab-p1", "testlab-p2"]
    gs.turn_player_id = "testlab-p1"
    gs.active_player_id = "testlab-p1"

    # Skip mulligan, start the game
    gs.mulligan_done = {"testlab-p1": True, "testlab-p2": True}
    start_game(gs)

    # Clear P1's initial draw (filled with random filler) so we can
    # populate the hand with exactly the scenario's requested cards.
    _clear_hand_to_deck(gs, "testlab-p1")

    # Set resources
    p1 = gs.players["testlab-p1"]
    p1.rune_pool.energy = scenario.energy
    for dom_str, val in scenario.power.items():
        p1.rune_pool.power[Domain(dom_str)] = val

    p2 = gs.players["testlab-p2"]
    p2.rune_pool.energy = 10  # Enough to play cards during P2's turn

    # Move requested cards from deck to hand
    _move_cards_to_hand(gs, "testlab-p1", scenario.p1_hand)

    # Place units on battlefields
    bf_list = list(gs.battlefields.values())
    _place_units_on_battlefields(gs, "testlab-p1", scenario.p1_bf_units, bf_list, card_db)
    _place_units_on_battlefields(gs, "testlab-p2", scenario.p2_bf_units, bf_list, card_db)

    # Set battlefield control based on which players have units there
    _update_battlefield_control(gs)

    return gs


def _clear_hand_to_deck(gs: GameState, player_id: str) -> None:
    """Return all cards in the player's hand back to their deck."""
    ps = gs.players[player_id]
    for iid in list(ps.hand):
        inst = gs.instances.get(iid)
        if inst:
            inst.zone = ZoneType.MAIN_DECK
            inst.location_id = player_id
        ps.main_deck.append(iid)
    ps.hand.clear()


def _move_cards_to_hand(gs: GameState, player_id: str, card_ids: list[str]) -> None:
    """Find cards by card_id in the player's deck and move them to hand."""
    ps = gs.players[player_id]
    for wanted_cid in card_ids:
        # Find an instance in the deck with the matching card_id
        found_iid: str | None = None
        for iid in ps.main_deck:
            inst = gs.instances.get(iid)
            if inst and inst.card_id == wanted_cid:
                found_iid = iid
                break
        if found_iid:
            ps.main_deck.remove(found_iid)
            inst = gs.instances[found_iid]
            inst.zone = ZoneType.HAND
            inst.location_id = player_id
            ps.hand.append(found_iid)


def _place_units_on_battlefields(
    gs: GameState,
    player_id: str,
    bf_units: dict[int, list[str]],
    bf_list: list,
    card_db: dict,
) -> None:
    """Place unit card instances on battlefields by index."""
    ps = gs.players[player_id]

    for bf_idx, card_ids in bf_units.items():
        if bf_idx >= len(bf_list):
            continue
        bf = bf_list[bf_idx]

        for cid in card_ids:
            # Try to find this card in the player's deck first
            found_iid: str | None = None
            for iid in ps.main_deck:
                inst = gs.instances.get(iid)
                if inst and inst.card_id == cid:
                    found_iid = iid
                    break

            if found_iid:
                # Move from deck to battlefield
                ps.main_deck.remove(found_iid)
                inst = gs.instances[found_iid]
                inst.zone = ZoneType.BATTLEFIELD
                inst.location_id = bf.battlefield_id
                inst.controller_id = player_id
                inst.entered_this_turn = False
                bf.units.append(found_iid)
            else:
                # Card not in deck -- create a fresh instance
                defn = card_db.get(cid)
                if not defn:
                    logger.warning("Card %s not found in DB, skipping", cid)
                    continue
                inst = CardInstance.create(defn, player_id, ZoneType.BATTLEFIELD, bf.battlefield_id)
                inst.controller_id = player_id
                inst.entered_this_turn = False
                gs.instances[inst.instance_id] = inst
                bf.units.append(inst.instance_id)


def _update_battlefield_control(gs: GameState) -> None:
    """Set control_status/controller_id based on which players have units."""
    for bf in gs.battlefields.values():
        owners = set()
        for iid in bf.units:
            inst = gs.instances.get(iid)
            if inst:
                owners.add(inst.controller_id or inst.owner_id)

        if len(owners) == 0:
            bf.control_status = ControlStatus.UNCONTROLLED
            bf.controller_id = None
        elif len(owners) == 1:
            bf.control_status = ControlStatus.CONTROLLED
            bf.controller_id = owners.pop()
        else:
            bf.control_status = ControlStatus.CONTESTED
            bf.controller_id = None
            bf.contested_by = list(owners)[0]  # the player who "moved in"
