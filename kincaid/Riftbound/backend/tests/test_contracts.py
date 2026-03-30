"""Contract validation: serialized game state must match the JSON Schema.

Every test serializes a GameState and validates the output against
contracts/state_update.schema.json.  If the serializer adds/removes/renames
a field, or produces a wrong type, these tests fail — catching frontend
drift before it reaches the browser.
"""

import json
from pathlib import Path

import jsonschema
import pytest

from helpers import add_unit, load_card_db, make_game, make_spell_def
from app.engine.card_types import (
    AbilityDefinition,
    CardDefinition,
    CardInstance,
    CardType,
    KeywordInstance,
)
from app.engine.enums import (
    AbilityType,
    ControlStatus,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.game_state import (
    BattlefieldState,
    ChainItem,
    CombatState,
    ShowdownState,
)
from app.protocol.serializers import serialize_for_player

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "contracts" / "state_update.schema.json"


@pytest.fixture(scope="module")
def schema():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate(state: dict, schema: dict) -> None:
    """Validate a serialized state dict against the contract schema."""
    jsonschema.validate(instance=state, schema=schema)


# ---- Basic game states ------------------------------------------------


class TestContractBasicStates:
    """Validate serialized output for common game phases."""

    def test_action_phase(self, schema):
        gs = make_game(phase=Phase.ACTION)
        validate(serialize_for_player(gs, "p1"), schema)

    def test_mulligan_phase(self, schema):
        gs = make_game(phase=Phase.SETUP_MULLIGAN)
        validate(serialize_for_player(gs, "p1"), schema)

    def test_both_players_see_valid_state(self, schema):
        gs = make_game(phase=Phase.ACTION)
        for pid in ["p1", "p2"]:
            validate(serialize_for_player(gs, pid), schema)

    def test_game_over(self, schema):
        gs = make_game()
        gs.game_over = True
        gs.winner_id = "p1"
        gs.phase = Phase.GAME_OVER
        validate(serialize_for_player(gs, "p1"), schema)

    def test_game_over_null_winner(self, schema):
        gs = make_game()
        gs.game_over = True
        gs.winner_id = None
        gs.phase = Phase.GAME_OVER
        validate(serialize_for_player(gs, "p1"), schema)


# ---- Cards on the board -----------------------------------------------


class TestContractWithCards:
    """Validate serialized output when various card types are in play."""

    def test_units_on_battlefield(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3)
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=5)
        validate(serialize_for_player(gs, "p1"), schema)

    def test_units_in_base(self, schema):
        gs = make_game()
        add_unit(gs, "p1", ZoneType.BASE, might=2)
        add_unit(gs, "p1", ZoneType.BASE, might=4)
        validate(serialize_for_player(gs, "p1"), schema)

    def test_unit_with_keywords(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        add_unit(
            gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3,
            keywords=(
                KeywordInstance(Keyword.TANK, 0),
                KeywordInstance(Keyword.ASSAULT, 2),
            ),
        )
        validate(serialize_for_player(gs, "p1"), schema)

    def test_gear_card(self, schema):
        gs = make_game()
        gear_def = CardDefinition(
            card_id="test_gear",
            name="Test Gear",
            card_type=CardType.GEAR,
            might_bonus=2,
            cost_energy=1,
        )
        gear = CardInstance.create(gear_def, "p1", ZoneType.BASE)
        gs.instances[gear.instance_id] = gear
        gs.base_gear["p1"].append(gear.instance_id)
        validate(serialize_for_player(gs, "p1"), schema)

    def test_gear_attached_to_unit(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        unit = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3)
        gear_def = CardDefinition(
            card_id="test_gear",
            name="Test Gear",
            card_type=CardType.GEAR,
            might_bonus=2,
        )
        gear = CardInstance.create(gear_def, "p1", ZoneType.BATTLEFIELD)
        gear.location_id = bf_id
        gear.attached_to = unit.instance_id
        unit.attached_cards.append(gear.instance_id)
        gs.instances[gear.instance_id] = gear
        validate(serialize_for_player(gs, "p1"), schema)

    def test_rune_card(self, schema):
        gs = make_game()
        rune_def = CardDefinition(
            card_id="test_rune",
            name="Test Rune",
            card_type=CardType.RUNE,
            domains=(Domain.FURY,),
        )
        rune = CardInstance.create(rune_def, "p1", ZoneType.RUNE_BOARD)
        gs.instances[rune.instance_id] = rune
        gs.base_runes["p1"].append(rune.instance_id)
        validate(serialize_for_player(gs, "p1"), schema)

    def test_spell_on_chain(self, schema):
        gs = make_game()
        spell_def = make_spell_def(name="Fireball", cost_energy=3)
        spell = CardInstance.create(spell_def, "p1", ZoneType.CHAIN)
        gs.instances[spell.instance_id] = spell
        item = ChainItem.create(
            controller_id="p1",
            card_instance_id=spell.instance_id,
        )
        gs.chain.push(item)
        validate(serialize_for_player(gs, "p1"), schema)

    def test_card_with_abilities(self, schema):
        gs = make_game()
        ab = AbilityDefinition(
            ability_id="test_ab",
            ability_type=AbilityType.ACTIVATED,
            timing="action",
            text="Deal 2 damage to target unit.",
            targets_required=1,
            target_type="unit",
        )
        unit_def = CardDefinition(
            card_id="ab_unit",
            name="Ability Unit",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(ab,),
        )
        unit = CardInstance.create(unit_def, "p1", ZoneType.BASE)
        gs.instances[unit.instance_id] = unit
        gs.base_units["p1"].append(unit.instance_id)
        validate(serialize_for_player(gs, "p1"), schema)

    def test_cards_in_hand_visible_to_owner(self, schema):
        gs = make_game()
        unit_def = CardDefinition(
            card_id="hand_unit",
            name="Hand Unit",
            card_type=CardType.UNIT,
            base_might=2,
            cost_energy=1,
        )
        inst = CardInstance.create(unit_def, "p1", ZoneType.HAND)
        inst.location_id = "p1"
        gs.instances[inst.instance_id] = inst
        gs.players["p1"].hand.append(inst.instance_id)
        # Owner sees full card info
        state = serialize_for_player(gs, "p1")
        validate(state, schema)
        hand_card = state["you"]["hand"][-1]
        assert "card_id" in hand_card
        assert hand_card["card_id"] == "hand_unit"

    def test_opponent_hand_is_facedown(self, schema):
        gs = make_game()
        unit_def = CardDefinition(
            card_id="opp_unit",
            name="Opponent Unit",
            card_type=CardType.UNIT,
            base_might=2,
        )
        inst = CardInstance.create(unit_def, "p2", ZoneType.HAND)
        inst.location_id = "p2"
        gs.instances[inst.instance_id] = inst
        gs.players["p2"].hand.append(inst.instance_id)
        # p1 sees opponent hand as facedown
        state = serialize_for_player(gs, "p1")
        validate(state, schema)
        opp_card = state["opponent"]["hand"][-1]
        assert opp_card["facedown"] is True
        assert "card_id" not in opp_card


# ---- Combat, showdown, and chain states --------------------------------


class TestContractCombatShowdown:
    """Validate serialized output during combat and showdown phases."""

    def test_active_combat(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3)
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=4)
        gs.active_combat = CombatState(
            battlefield_id=bf_id,
            attacker_id="p1",
            defender_id="p2",
        )
        gs.phase = Phase.COMBAT_DAMAGE
        state = serialize_for_player(gs, "p1")
        validate(state, schema)
        assert state["combat"] is not None
        assert state["combat"]["battlefield_id"] == bf_id

    def test_active_showdown(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        gs.active_showdown = ShowdownState(
            battlefield_id=bf_id,
            initiator_id="p1",
            focus_player_id="p2",
        )
        gs.phase = Phase.SHOWDOWN
        state = serialize_for_player(gs, "p1")
        validate(state, schema)
        assert state["showdown"] is not None

    def test_combat_showdown(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        gs.active_showdown = ShowdownState(
            battlefield_id=bf_id,
            initiator_id="p1",
            focus_player_id="p1",
            is_combat_showdown=True,
        )
        state = serialize_for_player(gs, "p1")
        validate(state, schema)

    def test_combat_with_assignments(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        gs.active_combat = CombatState(
            battlefield_id=bf_id,
            attacker_id="p1",
            defender_id="p2",
            phase="waiting_assignments",
            attacker_assignment={"some_unit": 3},
        )
        state = serialize_for_player(gs, "p1")
        validate(state, schema)
        assert state["combat"]["attacker_assigned"] is True
        assert state["combat"]["defender_assigned"] is False

    def test_chain_with_ability(self, schema):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3)
        item = ChainItem.create(
            controller_id="p1",
            source_instance_id=unit.instance_id,
            ability_id="test_ability",
        )
        gs.chain.push(item)
        state = serialize_for_player(gs, "p1")
        validate(state, schema)
        assert not state["chain"]["is_empty"]


# ---- Battlefield edge cases -------------------------------------------


class TestContractBattlefield:
    """Validate battlefield serialization edge cases."""

    def test_controlled_battlefield(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"
        validate(serialize_for_player(gs, "p1"), schema)

    def test_contested_battlefield(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTESTED
        bf.contested_by = "p2"
        validate(serialize_for_player(gs, "p1"), schema)

    def test_facedown_card_visible_to_controller(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        unit_def = CardDefinition(
            card_id="hidden_unit",
            name="Hidden Unit",
            card_type=CardType.UNIT,
            base_might=2,
        )
        hidden = CardInstance.create(unit_def, "p1", ZoneType.FACEDOWN_ZONE)
        hidden.facedown = True
        hidden.controller_id = "p1"
        gs.instances[hidden.instance_id] = hidden
        bf.facedown_card = hidden.instance_id
        # Controller sees full card
        state = serialize_for_player(gs, "p1")
        validate(state, schema)
        bf_view = list(state["battlefields"].values())[0]
        assert bf_view["facedown_card"] is not None
        assert "card_id" in bf_view["facedown_card"]

    def test_facedown_card_hidden_from_opponent(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        unit_def = CardDefinition(
            card_id="hidden_unit",
            name="Hidden Unit",
            card_type=CardType.UNIT,
            base_might=2,
        )
        hidden = CardInstance.create(unit_def, "p1", ZoneType.FACEDOWN_ZONE)
        hidden.facedown = True
        hidden.controller_id = "p1"
        gs.instances[hidden.instance_id] = hidden
        bf.facedown_card = hidden.instance_id
        # Opponent sees facedown stub
        state = serialize_for_player(gs, "p2")
        validate(state, schema)
        bf_view = list(state["battlefields"].values())[0]
        fd = bf_view["facedown_card"]
        assert fd["facedown"] is True
        assert "card_id" not in fd

    def test_staged_flags(self, schema):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.showdown_staged = True
        bf.combat_staged = True
        validate(serialize_for_player(gs, "p1"), schema)


# ---- Full game with real card DB --------------------------------------


class TestContractRealCards:
    """Validate with real card definitions loaded from card_definitions.json."""

    @pytest.fixture(autouse=True, scope="class")
    def _load_db(self):
        load_card_db()

    def test_full_game_creation(self, schema):
        from app.engine.card_db import CardDB
        from app.engine.game_state import create_game

        card_db = CardDB.all_cards()
        deck_config = {
            "legend_id": "unl-230-star-219",
            "champion_id": "ogn-097-298",
            "main_deck": ["ogn-097-298"] * 40,
            "rune_deck": ["ogn-007a-298"] * 12,
            "battlefields": ["unl-205-219", "unl-206-219", "ogn-275-298"],
        }
        configs = [
            {"player_id": "cp1", "display_name": "P1", **deck_config},
            {"player_id": "cp2", "display_name": "P2", **deck_config},
        ]
        gs = create_game("contract-test", configs, card_db)
        for pid in ["cp1", "cp2"]:
            state = serialize_for_player(gs, pid)
            validate(state, schema)

    def test_full_game_after_mulligan(self, schema):
        from app.engine.card_db import CardDB
        from app.engine.game_state import create_game
        from app.engine.state_machine import start_game

        card_db = CardDB.all_cards()
        deck_config = {
            "legend_id": "unl-230-star-219",
            "champion_id": "ogn-097-298",
            "main_deck": ["ogn-097-298"] * 40,
            "rune_deck": ["ogn-007a-298"] * 12,
            "battlefields": ["unl-205-219", "unl-206-219", "ogn-275-298"],
        }
        configs = [
            {"player_id": "cp1", "display_name": "P1", **deck_config},
            {"player_id": "cp2", "display_name": "P2", **deck_config},
        ]
        gs = create_game("contract-test2", configs, card_db)
        gs.mulligan_done = {"cp1": True, "cp2": True}
        start_game(gs)
        for pid in ["cp1", "cp2"]:
            state = serialize_for_player(gs, pid)
            validate(state, schema)
