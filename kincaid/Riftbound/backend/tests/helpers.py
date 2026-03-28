"""Shared test helpers: build minimal GameState fixtures."""

from __future__ import annotations

import sys
import os

# Ensure the app package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engine.card_db import CardDB
from app.engine.card_types import CardDefinition, CardInstance
from app.engine.enums import (
    CardType,
    CombatRole,
    ControlStatus,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.card_types import AbilityDefinition, KeywordInstance
from app.engine.game_state import (
    BattlefieldState,
    GameState,
    PlayerState,
    create_game,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cards")


def load_card_db():
    CardDB.load_all(DATA_DIR)


def make_unit_def(
    card_id: str = "test_unit",
    name: str = "Test Unit",
    might: int = 3,
    domains: tuple[Domain, ...] = (Domain.FURY,),
    keywords: tuple[KeywordInstance, ...] = (),
    cost_energy: int = 2,
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        name=name,
        card_type=CardType.UNIT,
        domains=domains,
        base_might=might,
        keywords=keywords,
        cost_energy=cost_energy,
    )


def make_spell_def(
    card_id: str = "test_spell",
    name: str = "Test Spell",
    keywords: tuple[KeywordInstance, ...] = (),
    effect_script: str | None = None,
    cost_energy: int = 1,
) -> CardDefinition:
    abilities = ()
    if effect_script:
        abilities = (
            AbilityDefinition(
                ability_id=f"{card_id}_ab",
                ability_type="activated",
                effect_script=effect_script,
            ),
        )
    return CardDefinition(
        card_id=card_id,
        name=name,
        card_type=CardType.SPELL,
        keywords=keywords,
        abilities=abilities,
        cost_energy=cost_energy,
    )


def make_game(
    num_battlefields: int = 2,
    phase: Phase = Phase.ACTION,
) -> GameState:
    """Create a minimal 2-player game state for testing."""
    gs = GameState(game_id="test")
    p1 = PlayerState(player_id="p1", display_name="Player 1")
    p2 = PlayerState(player_id="p2", display_name="Player 2")
    p1.is_first_turn = False
    p2.is_first_turn = False

    gs.players = {"p1": p1, "p2": p2}
    gs.player_order = ["p1", "p2"]
    gs.turn_player_id = "p1"
    gs.active_player_id = "p1"
    gs.phase = phase
    gs.turn_number = 1
    gs.base_units = {"p1": [], "p2": []}
    gs.base_gear = {"p1": [], "p2": []}
    gs.base_runes = {"p1": [], "p2": []}
    gs.mulligan_done = {"p1": True, "p2": True}

    # Add battlefields
    for i in range(num_battlefields):
        bf_def = CardDefinition(
            card_id=f"bf_{i}",
            name=f"Battlefield {i}",
            card_type=CardType.BATTLEFIELD,
        )
        bf_inst = CardInstance.create(bf_def, "neutral", ZoneType.BATTLEFIELD)
        gs.instances[bf_inst.instance_id] = bf_inst
        gs.battlefields[bf_inst.instance_id] = BattlefieldState(
            battlefield_id=bf_inst.instance_id,
            card_instance_id=bf_inst.instance_id,
        )

    return gs


def add_unit(
    gs: GameState,
    player_id: str,
    zone: ZoneType = ZoneType.BASE,
    might: int = 3,
    name: str = "Unit",
    keywords: tuple[KeywordInstance, ...] = (),
    bf_id: str | None = None,
) -> CardInstance:
    """Add a unit to the game and return it."""
    defn = make_unit_def(name=name, might=might, keywords=keywords)
    inst = CardInstance.create(defn, player_id, zone)
    inst.exhausted = False
    gs.instances[inst.instance_id] = inst

    if zone == ZoneType.BASE:
        gs.base_units[player_id].append(inst.instance_id)
    elif zone == ZoneType.BATTLEFIELD and bf_id:
        bf = gs.battlefields[bf_id]
        bf.units.append(inst.instance_id)
        inst.location_id = bf_id

    return inst
