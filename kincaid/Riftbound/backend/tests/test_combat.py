"""Tests for combat and showdown mechanics."""

import pytest
from helpers import make_game, add_unit

from app.engine.combat import (
    apply_combat_damage,
    open_showdown,
    pass_focus,
    start_combat,
    validate_assignment,
)
from app.engine.card_types import KeywordInstance
from app.engine.enums import (
    CombatRole,
    ControlStatus,
    Keyword,
    ZoneType,
)


class TestShowdown:
    def test_open_showdown(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        logs = open_showdown(gs, bf_id, "p1")
        assert gs.active_showdown is not None
        assert gs.active_showdown.battlefield_id == bf_id
        assert gs.active_showdown.focus_player_id == "p1"

    def test_pass_focus_alternates(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        open_showdown(gs, bf_id, "p1")

        logs = pass_focus(gs, "p1")
        assert gs.active_showdown is not None  # not closed yet
        assert gs.active_showdown.focus_player_id == "p2"

    def test_both_pass_closes_showdown(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id)

        open_showdown(gs, bf_id, "p1")
        pass_focus(gs, "p1")
        pass_focus(gs, "p2")
        assert gs.active_showdown is None


class TestStartCombat:
    def test_combat_initialization(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id)
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id)

        logs = start_combat(gs, bf_id)
        assert gs.active_combat is not None
        assert gs.active_combat.attacker_id == "p1"
        assert gs.active_combat.defender_id == "p2"
        # Showdown opens as part of combat
        assert gs.active_showdown is not None


class TestValidateAssignment:
    def test_valid_assignment(self):
        gs = make_game()
        unit = add_unit(gs, "p2", might=3, name="Target")
        result = validate_assignment(3, {unit.instance_id: 3}, [unit])
        assert result is None

    def test_wrong_total(self):
        gs = make_game()
        unit = add_unit(gs, "p2", might=3)
        result = validate_assignment(3, {unit.instance_id: 2}, [unit])
        assert result is not None
        assert "exactly" in result.lower()

    def test_tank_ordering_enforced(self):
        gs = make_game()
        tank_kw = (KeywordInstance(keyword=Keyword.TANK),)
        tank = add_unit(gs, "p2", might=3, name="Tank", keywords=tank_kw)
        squishy = add_unit(gs, "p2", might=2, name="Squishy")

        # Assign damage to squishy before tank has lethal — invalid
        result = validate_assignment(
            4,
            {tank.instance_id: 1, squishy.instance_id: 3},
            [tank, squishy],
        )
        assert result is not None
        assert "tank" in result.lower()

    def test_tank_ordering_valid(self):
        gs = make_game()
        tank_kw = (KeywordInstance(keyword=Keyword.TANK),)
        tank = add_unit(gs, "p2", might=3, name="Tank", keywords=tank_kw)
        squishy = add_unit(gs, "p2", might=2, name="Squishy")

        # Assign lethal to tank first, then rest to squishy — valid
        result = validate_assignment(
            5,
            {tank.instance_id: 3, squishy.instance_id: 2},
            [tank, squishy],
        )
        assert result is None

    def test_lethal_before_next(self):
        gs = make_game()
        u1 = add_unit(gs, "p2", might=3, name="U1")
        u2 = add_unit(gs, "p2", might=3, name="U2")

        # Split damage without giving lethal to first — invalid
        result = validate_assignment(
            4,
            {u1.instance_id: 2, u2.instance_id: 2},
            [u1, u2],
        )
        assert result is not None
        assert "lethal" in result.lower()
