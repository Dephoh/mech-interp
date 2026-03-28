"""Tests for the cleanup loop."""

import pytest
from helpers import make_game, add_unit

from app.engine.card_types import KeywordInstance
from app.engine.cleanup import kill_temporary_units, run_cleanup
from app.engine.enums import (
    ControlStatus,
    Keyword,
    Phase,
    ZoneType,
)


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
