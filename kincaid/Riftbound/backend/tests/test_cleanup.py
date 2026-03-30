"""Tests for the cleanup loop."""

import pytest
from helpers import make_game, add_unit

from app.engine.card_types import KeywordInstance
from app.engine.cleanup import run_cleanup, kill_temporary_units
from app.engine.enums import (
    ControlStatus,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.game_state import CombatState, ShowdownState


class TestKillDeadUnits:
    def test_unit_with_lethal_damage_dies(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=3)
        unit.damage = 3  # exactly lethal

        logs = run_cleanup(gs)
        assert unit.zone == ZoneType.TRASH
        assert unit.instance_id in gs.players["p1"].trash

    def test_unit_below_lethal_survives(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=3)
        unit.damage = 2

        logs = run_cleanup(gs)
        assert unit.zone == ZoneType.BASE

    def test_unit_with_zero_might_not_killed(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=0)
        unit.damage = 0

        run_cleanup(gs)
        assert unit.zone == ZoneType.BASE


class TestResetEmptyBattlefields:
    def test_empty_controlled_battlefield_becomes_uncontrolled(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"
        # No units at battlefield

        run_cleanup(gs)
        assert bf.control_status == ControlStatus.UNCONTROLLED
        assert bf.controller_id is None

    def test_battlefield_with_units_stays_controlled(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id)

        run_cleanup(gs)
        assert bf.control_status == ControlStatus.CONTROLLED


class TestStageCombats:
    def test_two_players_at_contested_bf_stages_combat(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id)
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id)

        run_cleanup(gs)
        # Combat should have been staged and then opened
        # (since we're in Action phase with empty chain)
        assert gs.active_combat is not None or bf.combat_staged


class TestKillTemporary:
    def test_temporary_unit_killed_at_beginning(self):
        gs = make_game(phase=Phase.ACTION)
        gs.turn_player_id = "p1"
        temp_kw = (KeywordInstance(keyword=Keyword.TEMPORARY),)
        unit = add_unit(gs, "p1", keywords=temp_kw)
        unit.entered_this_turn = False  # survived at least one turn

        logs = kill_temporary_units(gs)
        assert unit.zone == ZoneType.TRASH

    def test_temporary_unit_safe_on_entry_turn(self):
        gs = make_game(phase=Phase.ACTION)
        gs.turn_player_id = "p1"
        temp_kw = (KeywordInstance(keyword=Keyword.TEMPORARY),)
        unit = add_unit(gs, "p1", keywords=temp_kw)
        unit.entered_this_turn = True  # just entered

        logs = kill_temporary_units(gs)
        assert unit.zone == ZoneType.BASE


class TestCombatStagingInterference:
    """Regression tests: cleanup must not re-stage combat at a battlefield
    that already has an active combat or showdown in progress."""

    def test_no_restage_during_active_combat(self):
        """Playing a spell during active combat should NOT produce 'Combat staged'."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id)
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id)

        # Simulate combat already opened: combat_staged cleared, active_combat set
        bf.combat_staged = False
        gs.active_combat = CombatState(
            battlefield_id=bf_id,
            attacker_id="p1",
            defender_id="p2",
            phase="showdown",
        )

        logs = run_cleanup(gs)
        staged_msgs = [l for l in logs if "Combat staged" in l]
        assert staged_msgs == [], (
            f"Should not re-stage combat during active combat, got: {staged_msgs}"
        )

    def test_no_restage_during_active_showdown(self):
        """Cleanup should not stage combat at a BF with an active showdown."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id)
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id)

        bf.combat_staged = False
        gs.active_showdown = ShowdownState(
            battlefield_id=bf_id,
            initiator_id="p1",
            focus_player_id="p1",
            is_combat_showdown=True,
        )

        logs = run_cleanup(gs)
        staged_msgs = [l for l in logs if "Combat staged" in l]
        assert staged_msgs == [], (
            f"Should not stage combat during active showdown, got: {staged_msgs}"
        )

    def test_still_stages_combat_at_different_battlefield(self):
        """A second contested BF should still get combat staged normally."""
        gs = make_game(num_battlefields=2)
        bf_ids = list(gs.battlefields.keys())
        bf0 = gs.battlefields[bf_ids[0]]
        bf1 = gs.battlefields[bf_ids[1]]

        # BF0: active combat in progress
        bf0.contested_by = "p1"
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_ids[0])
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_ids[0])
        bf0.combat_staged = False
        gs.active_combat = CombatState(
            battlefield_id=bf_ids[0],
            attacker_id="p1",
            defender_id="p2",
            phase="showdown",
        )

        # BF1: newly contested, no combat yet
        bf1.contested_by = "p2"
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_ids[1])
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_ids[1])

        logs = run_cleanup(gs)
        staged_at_bf0 = [l for l in logs if "Combat staged" in l and bf_ids[0] in l]
        staged_at_bf1 = [l for l in logs if "Combat staged" in l and bf_ids[1] in l]

        assert staged_at_bf0 == [], "Should NOT re-stage at active-combat BF"
        assert len(staged_at_bf1) == 1, "Should stage combat at the other BF"

    def test_stages_combat_when_no_active_combat(self):
        """Normal case: contested BF with no active combat should still stage."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id)
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id)

        # No active combat or showdown — staging should proceed
        logs = run_cleanup(gs)
        staged_msgs = [l for l in logs if "Combat staged" in l]
        assert len(staged_msgs) >= 1, "Should stage combat when no active combat exists"
