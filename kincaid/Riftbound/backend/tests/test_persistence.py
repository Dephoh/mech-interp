"""Tests for game state persistence: serialization round-trip and file I/O."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from helpers import (
    add_unit,
    load_card_db,
    make_game,
    make_unit_def,
)
from app.engine.card_types import CardDefinition, CardInstance, KeywordInstance
from app.engine.enums import (
    CardType,
    CombatRole,
    ControlStatus,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.game_state import (
    ActiveModifier,
    ActiveReplacement,
    BattlefieldState,
    ChainItem,
    ChainState,
    CombatState,
    GameLog,
    GameState,
    PlayerState,
    RunePool,
    ShowdownState,
)
from app.engine.persistence import (
    delete_game_state,
    game_state_from_dict,
    game_state_to_dict,
    list_saved_rooms,
    load_game_state,
    save_game_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_card_db() -> dict[str, CardDefinition]:
    """Build a minimal card DB for tests (avoids needing full JSON data)."""
    defs = {}
    for cid, name, ctype, might in [
        ("test_unit", "Test Unit", CardType.UNIT, 3),
        ("bf_0", "Battlefield 0", CardType.BATTLEFIELD, 0),
        ("bf_1", "Battlefield 1", CardType.BATTLEFIELD, 0),
        ("test_spell", "Test Spell", CardType.SPELL, 0),
        ("test_rune", "Test Rune", CardType.RUNE, 0),
        ("test_gear", "Test Gear", CardType.GEAR, 0),
        ("test_legend", "Test Legend", CardType.LEGEND, 0),
    ]:
        defs[cid] = CardDefinition(
            card_id=cid, name=name, card_type=ctype, base_might=might,
        )
    return defs


CARD_DB = _make_card_db()


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Test that game_state_to_dict -> game_state_from_dict preserves all data."""

    def test_minimal_game_round_trip(self):
        """A basic game with 2 players and 2 battlefields survives round-trip."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        gs.turn_number = 5
        gs.turn_player_id = "p1"
        gs.active_player_id = "p2"

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert gs2.game_id == gs.game_id
        assert gs2.phase == Phase.ACTION
        assert gs2.turn_number == 5
        assert gs2.turn_player_id == "p1"
        assert gs2.active_player_id == "p2"
        assert set(gs2.players.keys()) == {"p1", "p2"}
        assert gs2.player_order == ["p1", "p2"]
        assert len(gs2.battlefields) == 2
        assert gs2.game_over is False

    def test_player_state_preserved(self):
        """Player score, rune pool, hand, deck, and tracking survive round-trip."""
        gs = make_game()
        p1 = gs.players["p1"]
        p1.score = 4
        p1.rune_pool.energy = 7
        p1.rune_pool.power = {Domain.FURY: 3, Domain.CALM: 1}
        p1.is_first_turn = False
        p1.goes_second = True
        p1.played_main_deck_this_turn = 2
        p1.battlefields_scored_this_turn = {"bf_a", "bf_b"}

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)
        p1r = gs2.players["p1"]

        assert p1r.score == 4
        assert p1r.rune_pool.energy == 7
        assert p1r.rune_pool.power[Domain.FURY] == 3
        assert p1r.rune_pool.power[Domain.CALM] == 1
        assert p1r.is_first_turn is False
        assert p1r.goes_second is True
        assert p1r.played_main_deck_this_turn == 2
        assert p1r.battlefields_scored_this_turn == {"bf_a", "bf_b"}

    def test_card_instance_preserved(self):
        """Card instance mutable state (damage, buffs, etc.) survives round-trip."""
        gs = make_game()
        unit = add_unit(gs, "p1", zone=ZoneType.BASE, might=5, name="Unit")
        unit.damage = 2
        unit.buff_counter = True
        unit.exhausted = True
        unit.stunned = True
        unit.combat_role = CombatRole.ATTACKER
        unit.facedown = True
        unit.entered_this_turn = False
        unit.hidden_at_battlefield = "bf_123"
        unit.hidden_ready = True
        unit.accelerated = True
        unit.aura_might_bonus = 2
        unit.gear_might_bonus = 1
        unit.trait_might_set = 10
        unit.might_modifiers = [1, -2, 3]
        unit.granted_keywords = [
            KeywordInstance(keyword=Keyword.AMBUSH, value=0),
        ]
        unit.aura_keywords = [
            KeywordInstance(keyword=Keyword.SHIELD, value=2),
        ]

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)
        u2 = gs2.instances[unit.instance_id]

        assert u2.damage == 2
        assert u2.buff_counter is True
        assert u2.exhausted is True
        assert u2.stunned is True
        assert u2.combat_role == CombatRole.ATTACKER
        assert u2.facedown is True
        assert u2.entered_this_turn is False
        assert u2.hidden_at_battlefield == "bf_123"
        assert u2.hidden_ready is True
        assert u2.accelerated is True
        assert u2.aura_might_bonus == 2
        assert u2.gear_might_bonus == 1
        assert u2.trait_might_set == 10
        assert u2.might_modifiers == [1, -2, 3]
        assert len(u2.granted_keywords) == 1
        assert u2.granted_keywords[0].keyword == Keyword.AMBUSH
        assert len(u2.aura_keywords) == 1
        assert u2.aura_keywords[0].keyword == Keyword.SHIELD
        assert u2.aura_keywords[0].value == 2

    def test_battlefield_state_preserved(self):
        """Battlefield control status and unit references survive round-trip."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTESTED
        bf.controller_id = "p1"
        bf.contested_by = "p2"
        bf.scored_this_turn_by = "p1"
        bf.showdown_staged = True
        bf.combat_staged = True

        # Add a unit to the battlefield
        unit = add_unit(gs, "p1", zone=ZoneType.BATTLEFIELD, bf_id=bf_id, might=4)

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)
        bf2 = gs2.battlefields[bf_id]

        assert bf2.control_status == ControlStatus.CONTESTED
        assert bf2.controller_id == "p1"
        assert bf2.contested_by == "p2"
        assert bf2.scored_this_turn_by == "p1"
        assert bf2.showdown_staged is True
        assert bf2.combat_staged is True
        assert unit.instance_id in bf2.units

    def test_chain_state_preserved(self):
        """Chain stack and passed players survive round-trip."""
        gs = make_game()
        item = ChainItem.create(
            controller_id="p1",
            source_instance_id="src_123",
            ability_id="ab_001",
            card_instance_id="card_456",
            targets=["t1", "t2"],
        )
        gs.chain.push(item)
        gs.chain.passed_players = {"p2"}

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert len(gs2.chain.stack) == 1
        i2 = gs2.chain.stack[0]
        assert i2.item_id == item.item_id
        assert i2.controller_id == "p1"
        assert i2.source_instance_id == "src_123"
        assert i2.ability_id == "ab_001"
        assert i2.card_instance_id == "card_456"
        assert i2.targets == ["t1", "t2"]
        assert i2.pending is True
        assert gs2.chain.passed_players == {"p2"}

    def test_combat_state_preserved(self):
        """Active combat state survives round-trip."""
        gs = make_game()
        gs.active_combat = CombatState(
            battlefield_id="bf_001",
            attacker_id="p1",
            defender_id="p2",
            phase="waiting_assignments",
            attacker_assignment={"u1": 3, "u2": 1},
            defender_assignment=None,
        )

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert gs2.active_combat is not None
        c = gs2.active_combat
        assert c.battlefield_id == "bf_001"
        assert c.attacker_id == "p1"
        assert c.defender_id == "p2"
        assert c.phase == "waiting_assignments"
        assert c.attacker_assignment == {"u1": 3, "u2": 1}
        assert c.defender_assignment is None

    def test_showdown_state_preserved(self):
        """Active showdown state survives round-trip."""
        gs = make_game()
        gs.active_showdown = ShowdownState(
            battlefield_id="bf_002",
            initiator_id="p2",
            focus_player_id="p1",
            passed_players={"p2"},
            is_combat_showdown=True,
        )

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert gs2.active_showdown is not None
        sd = gs2.active_showdown
        assert sd.battlefield_id == "bf_002"
        assert sd.initiator_id == "p2"
        assert sd.focus_player_id == "p1"
        assert sd.passed_players == {"p2"}
        assert sd.is_combat_showdown is True

    def test_no_combat_no_showdown(self):
        """None combat/showdown stays None through round-trip."""
        gs = make_game()
        assert gs.active_combat is None
        assert gs.active_showdown is None

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert gs2.active_combat is None
        assert gs2.active_showdown is None

    def test_game_over_preserved(self):
        """Game over state and winner survive round-trip."""
        gs = make_game()
        gs.game_over = True
        gs.winner_id = "p1"

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert gs2.game_over is True
        assert gs2.winner_id == "p1"

    def test_active_modifiers_preserved(self):
        """Active modifier list survives round-trip."""
        gs = make_game()
        gs.active_modifiers = [
            ActiveModifier(
                source_instance_id="src_1",
                ability_id="ab_1",
                stat="might",
                amount=2,
                target_spec={"obj_type": "unit", "scope": "friendly"},
                exclude_self=True,
                duration="continuous",
            ),
        ]

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert len(gs2.active_modifiers) == 1
        m = gs2.active_modifiers[0]
        assert m.source_instance_id == "src_1"
        assert m.ability_id == "ab_1"
        assert m.stat == "might"
        assert m.amount == 2
        assert m.target_spec == {"obj_type": "unit", "scope": "friendly"}
        assert m.exclude_self is True

    def test_active_replacements_preserved(self):
        """Active replacement list survives round-trip."""
        gs = make_game()
        gs.active_replacements = [
            ActiveReplacement(
                source_instance_id="src_2",
                ability_id="ab_2",
                watch_event="kill",
                condition={"cond_type": "always"},
                alternative_ir={"type": "heal", "amount": 1},
                target_spec={"obj_type": "unit"},
            ),
        ]

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert len(gs2.active_replacements) == 1
        r = gs2.active_replacements[0]
        assert r.source_instance_id == "src_2"
        assert r.watch_event == "kill"
        assert r.alternative_ir == {"type": "heal", "amount": 1}

    def test_pending_choice_preserved(self):
        """Pending choice dict survives round-trip."""
        gs = make_game()
        gs.pending_choice = {
            "controller_id": "p1",
            "prompt": "Choose a target",
            "options": ["opt1", "opt2"],
            "min_choices": 1,
            "max_choices": 1,
        }

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert gs2.pending_choice is not None
        assert gs2.pending_choice["controller_id"] == "p1"
        assert gs2.pending_choice["options"] == ["opt1", "opt2"]

    def test_log_preserved(self):
        """Game log entries survive round-trip."""
        gs = make_game()
        gs.log.add("Turn 1 started")
        gs.log.add("Player 1 plays a unit", player="p1")

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert len(gs2.log.entries) == 2
        assert gs2.log.entries[0]["message"] == "Turn 1 started"
        assert gs2.log.entries[1]["message"] == "Player 1 plays a unit"
        assert gs2.log.entries[1]["player"] == "p1"

    def test_json_serializable(self):
        """The dict output can be serialized to JSON and back without loss."""
        gs = make_game()
        gs.players["p1"].rune_pool.power = {Domain.FURY: 2, Domain.MIND: 1}
        add_unit(gs, "p1", zone=ZoneType.BASE, might=4)

        data = game_state_to_dict(gs)
        json_str = json.dumps(data)
        data2 = json.loads(json_str)
        gs2 = game_state_from_dict(data2, card_db=CARD_DB)

        assert gs2.game_id == gs.game_id
        assert gs2.players["p1"].rune_pool.power[Domain.FURY] == 2

    def test_mulligan_done_preserved(self):
        """Mulligan tracking survives round-trip."""
        gs = make_game()
        gs.mulligan_done = {"p1": True, "p2": False}

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert gs2.mulligan_done == {"p1": True, "p2": False}

    def test_base_zones_preserved(self):
        """base_units, base_gear, base_runes survive round-trip."""
        gs = make_game()
        unit = add_unit(gs, "p1", zone=ZoneType.BASE, might=3)

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        assert unit.instance_id in gs2.base_units["p1"]

    def test_attachment_references_preserved(self):
        """Attached cards references survive round-trip."""
        gs = make_game()
        unit = add_unit(gs, "p1", zone=ZoneType.BASE, might=3)
        # Simulate a gear attached to the unit
        gear_def = CardDefinition(
            card_id="test_gear", name="Test Gear", card_type=CardType.GEAR
        )
        gear = CardInstance.create(gear_def, "p1", ZoneType.BASE)
        gs.instances[gear.instance_id] = gear
        gear.attached_to = unit.instance_id
        unit.attached_cards.append(gear.instance_id)

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)

        u2 = gs2.instances[unit.instance_id]
        g2 = gs2.instances[gear.instance_id]
        assert g2.attached_to == unit.instance_id
        assert gear.instance_id in u2.attached_cards


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------


class TestFilePersistence:
    """Test save/load/delete/list operations with a temp directory."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        # Clean up temp files
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load(self):
        """Save a game state, then load it back and verify."""
        gs = make_game()
        gs.turn_number = 3
        gs.players["p1"].score = 5

        path = save_game_state("room-abc", gs, save_dir=self.tmp_dir)
        assert os.path.exists(path)
        assert path.endswith("room-abc.json")

        gs2 = load_game_state("room-abc", save_dir=self.tmp_dir, card_db=CARD_DB)
        assert gs2 is not None
        assert gs2.game_id == gs.game_id
        assert gs2.turn_number == 3
        assert gs2.players["p1"].score == 5

    def test_load_nonexistent_returns_none(self):
        """Loading a room that was never saved returns None."""
        gs = load_game_state("nonexistent", save_dir=self.tmp_dir, card_db=CARD_DB)
        assert gs is None

    def test_delete_game_state(self):
        """Deleting a saved game state removes the file."""
        gs = make_game()
        save_game_state("room-del", gs, save_dir=self.tmp_dir)
        assert delete_game_state("room-del", save_dir=self.tmp_dir) is True
        assert load_game_state("room-del", save_dir=self.tmp_dir, card_db=CARD_DB) is None
        # Second delete returns False (file already gone)
        assert delete_game_state("room-del", save_dir=self.tmp_dir) is False

    def test_list_saved_rooms(self):
        """list_saved_rooms returns all room_ids with saved state files."""
        gs = make_game()
        save_game_state("room-1", gs, save_dir=self.tmp_dir)
        save_game_state("room-2", gs, save_dir=self.tmp_dir)
        save_game_state("room-3", gs, save_dir=self.tmp_dir)

        rooms = list_saved_rooms(save_dir=self.tmp_dir)
        assert set(rooms) == {"room-1", "room-2", "room-3"}

    def test_list_saved_rooms_empty_dir(self):
        """list_saved_rooms returns empty list for empty/nonexistent dir."""
        rooms = list_saved_rooms(save_dir=self.tmp_dir)
        assert rooms == []

        rooms2 = list_saved_rooms(save_dir=os.path.join(self.tmp_dir, "nope"))
        assert rooms2 == []

    def test_save_creates_directory(self):
        """Save creates the game_states directory if it doesn't exist."""
        nested = os.path.join(self.tmp_dir, "sub", "game_states")
        gs = make_game()
        save_game_state("room-x", gs, save_dir=nested)
        assert os.path.isdir(nested)
        assert os.path.exists(os.path.join(nested, "room-x.json"))

    def test_save_file_is_valid_json(self):
        """The saved file is valid JSON and contains game_id."""
        gs = make_game()
        path = save_game_state("room-json", gs, save_dir=self.tmp_dir)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["game_id"] == gs.game_id
        assert data["_room_id"] == "room-json"

    def test_overwrite_on_save(self):
        """Saving the same room_id overwrites the previous file."""
        gs = make_game()
        gs.turn_number = 1
        save_game_state("room-ow", gs, save_dir=self.tmp_dir)

        gs.turn_number = 10
        save_game_state("room-ow", gs, save_dir=self.tmp_dir)

        gs2 = load_game_state("room-ow", save_dir=self.tmp_dir, card_db=CARD_DB)
        assert gs2.turn_number == 10

    def test_corrupted_file_returns_none(self):
        """Loading a corrupted JSON file returns None (graceful failure)."""
        path = os.path.join(self.tmp_dir, "room-bad.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        gs = load_game_state("room-bad", save_dir=self.tmp_dir, card_db=CARD_DB)
        assert gs is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and special scenarios."""

    def test_empty_game_state(self):
        """Minimal game state (no players) survives round-trip."""
        gs = GameState(game_id="empty")
        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)
        assert gs2.game_id == "empty"
        assert gs2.players == {}
        assert gs2.player_order == []

    def test_victory_score_preserved(self):
        """Custom victory score survives round-trip."""
        gs = make_game()
        gs.victory_score = 11

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)
        assert gs2.victory_score == 11

    def test_teams_preserved(self):
        """2v2 team assignments survive round-trip."""
        gs = make_game()
        gs.teams = {"p1": 0, "p2": 1}

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)
        assert gs2.teams == {"p1": 0, "p2": 1}

    def test_pending_combats_and_showdowns(self):
        """pending_combats and pending_showdowns lists survive round-trip."""
        gs = make_game()
        gs.pending_combats = ["bf_001", "bf_002"]
        gs.pending_showdowns = ["bf_003"]

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)
        assert gs2.pending_combats == ["bf_001", "bf_002"]
        assert gs2.pending_showdowns == ["bf_003"]

    def test_last_effect_succeeded(self):
        """last_effect_succeeded flag survives round-trip."""
        gs = make_game()
        gs.last_effect_succeeded = True

        data = game_state_to_dict(gs)
        gs2 = game_state_from_dict(data, card_db=CARD_DB)
        assert gs2.last_effect_succeeded is True
