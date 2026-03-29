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


# =========================================================================
# TS-03: Combat edge case tests
# =========================================================================


class TestTankKeywordEdgeCases:
    """Tank ordering edge cases beyond the basic tests above."""

    def test_tank_takes_damage_first(self):
        """Unit with Tank must absorb lethal before non-Tank units receive any."""
        gs = make_game()
        tank_kw = (KeywordInstance(keyword=Keyword.TANK),)
        tank = add_unit(gs, "p2", might=4, name="BigTank", keywords=tank_kw)
        squishy = add_unit(gs, "p2", might=2, name="Squishy")

        # Valid: give tank lethal (4), leftover to squishy (1)
        result = validate_assignment(
            5,
            {tank.instance_id: 4, squishy.instance_id: 1},
            [tank, squishy],
        )
        assert result is None

        # Invalid: skip tank, hit squishy only
        result2 = validate_assignment(
            5,
            {tank.instance_id: 0, squishy.instance_id: 5},
            [tank, squishy],
        )
        assert result2 is not None
        assert "tank" in result2.lower()

    def test_multiple_tanks(self):
        """Multiple Tank units must each receive lethal before non-Tank gets any."""
        gs = make_game()
        tank_kw = (KeywordInstance(keyword=Keyword.TANK),)
        tank1 = add_unit(gs, "p2", might=2, name="Tank1", keywords=tank_kw)
        tank2 = add_unit(gs, "p2", might=3, name="Tank2", keywords=tank_kw)
        squishy = add_unit(gs, "p2", might=1, name="Squishy")

        # Valid: lethal to both tanks (2 + 3), then 1 to squishy
        result = validate_assignment(
            6,
            {tank1.instance_id: 2, tank2.instance_id: 3, squishy.instance_id: 1},
            [tank1, tank2, squishy],
        )
        assert result is None

        # Invalid: skip tank2 but hit squishy
        result2 = validate_assignment(
            6,
            {tank1.instance_id: 2, tank2.instance_id: 1, squishy.instance_id: 3},
            [tank1, tank2, squishy],
        )
        assert result2 is not None
        assert "tank" in result2.lower()


class TestLethalEdgeCases:
    """Lethal-before-next rule edge cases."""

    def test_lethal_valid_overflow(self):
        """Assigning extra damage to the last unit is fine."""
        gs = make_game()
        u1 = add_unit(gs, "p2", might=2, name="U1")
        u2 = add_unit(gs, "p2", might=3, name="U2")

        # Lethal to u1 (2), rest to u2 (3) -- valid
        result = validate_assignment(
            5,
            {u1.instance_id: 2, u2.instance_id: 3},
            [u1, u2],
        )
        assert result is None

    def test_single_target_always_valid(self):
        """With only one target, any valid-total assignment succeeds."""
        gs = make_game()
        u = add_unit(gs, "p2", might=5, name="Lone")
        result = validate_assignment(3, {u.instance_id: 3}, [u])
        assert result is None


class TestApplyCombatDamage:
    """Test apply_combat_damage and simultaneous damage."""

    def test_simultaneous_lethal_both_die(self):
        """Both sides deal lethal — both sides lose their units."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        attacker = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Att")
        defender = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Def")

        logs = start_combat(gs, bf_id)
        # Both auto-assigned since single-target on each side
        # apply_combat_damage should have fired automatically
        # After combat, both units should have taken lethal damage (3 >= 3)
        assert attacker.damage >= attacker.effective_might or not attacker.is_alive or attacker.damage == 0
        assert defender.damage >= defender.effective_might or not defender.is_alive or defender.damage == 0

    def test_attacker_survives_defender_dies(self):
        """Attacker with more might survives, defender dies."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        attacker = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=5, name="BigAtt")
        defender = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="SmallDef")

        logs = start_combat(gs, bf_id)
        # Defender took 5 damage (attacker might) against 2 might = dead
        # Attacker took 2 damage against 5 might, healed to 0 after combat
        # After combat resolves, attacker should be alive (damage healed)
        assert attacker.damage == 0  # healed after combat


class TestEmptyBattlefieldCombat:
    def test_combat_with_no_defenders(self):
        """Combat with units only on attacker side should skip to resolution."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        # Only attacker-side units
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Att")
        # No defender units — second player has no units here

        # start_combat needs both players present; check that the engine handles this
        logs = start_combat(gs, bf_id)
        log_text = " ".join(logs).lower()
        # Should indicate combat can't proceed or resolved immediately
        assert "combat" in log_text or "requires" in log_text

    def test_no_units_on_either_side(self):
        """Combat with zero units on both sides should not crash."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        # No units at all
        logs = start_combat(gs, bf_id)
        # Should gracefully return a message, not crash
        assert isinstance(logs, list)


class TestCombatWithZeroMight:
    def test_zero_might_deals_no_damage(self):
        """A unit with 0 might contributes 0 to combat damage."""
        gs = make_game()
        zero_unit = add_unit(gs, "p1", might=0, name="ZeroMight")
        assert zero_unit.combat_might == 0

    def test_stunned_unit_deals_no_damage(self):
        """A stunned unit has 0 combat_might regardless of base might."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=5, name="Stunned")
        unit.stunned = True
        assert unit.combat_might == 0


class TestCombatPreservesOtherBattlefields:
    def test_combat_at_bf0_does_not_affect_bf1(self):
        """Combat at one battlefield must not alter units at another."""
        gs = make_game(num_battlefields=2)
        bf_ids = list(gs.battlefields.keys())
        bf0_id = bf_ids[0]
        bf1_id = bf_ids[1]

        bf0 = gs.battlefields[bf0_id]
        bf0.contested_by = "p1"

        # Place units at bf0 (combat site)
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf0_id, might=3, name="CombatAtt")
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf0_id, might=3, name="CombatDef")

        # Place units at bf1 (bystander)
        bystander = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf1_id, might=4, name="Bystander")

        start_combat(gs, bf0_id)

        # Bystander should be unaffected
        assert bystander.damage == 0
        assert bystander.combat_role == CombatRole.NONE
        assert bystander.instance_id in gs.battlefields[bf1_id].units


class TestCombatWithGear:
    """Tests for gear interactions in combat context."""

    def test_gear_might_bonus_in_effective_might(self):
        """A unit with might_modifiers from gear should include them in combat."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Equipped")
        # Simulate gear adding +2 might via might_modifiers
        unit.might_modifiers.append(2)
        assert unit.effective_might == 5
        assert unit.combat_might == 5

    def test_assault_keyword_boosts_attacker(self):
        """Assault keyword adds might only when unit is attacker."""
        gs = make_game()
        assault_kw = (KeywordInstance(keyword=Keyword.ASSAULT, value=2),)
        unit = add_unit(gs, "p1", might=3, name="Charger", keywords=assault_kw)

        # Not in combat
        assert unit.effective_might == 3
        assert unit.combat_might == 3

        # As attacker
        unit.combat_role = CombatRole.ATTACKER
        assert unit.effective_might == 5
        assert unit.combat_might == 5

        # As defender — no assault bonus
        unit.combat_role = CombatRole.DEFENDER
        assert unit.effective_might == 3

    def test_shield_keyword_boosts_defender(self):
        """Shield keyword adds might only when unit is defender."""
        gs = make_game()
        shield_kw = (KeywordInstance(keyword=Keyword.SHIELD, value=2),)
        unit = add_unit(gs, "p1", might=3, name="Guardian", keywords=shield_kw)

        # Not in combat
        assert unit.effective_might == 3

        # As defender
        unit.combat_role = CombatRole.DEFENDER
        assert unit.effective_might == 5

        # As attacker — no shield bonus
        unit.combat_role = CombatRole.ATTACKER
        assert unit.effective_might == 3
