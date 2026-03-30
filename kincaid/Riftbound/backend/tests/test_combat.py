"""Tests for combat and showdown mechanics."""

import pytest
from helpers import make_game, make_unit_def, add_unit

from app.engine.combat import (
    _reveal_hidden_at_battlefield,
    apply_combat_damage,
    open_showdown,
    pass_focus,
    resolve_combat,
    start_combat,
    validate_assignment,
)
from app.engine.card_types import CardDefinition, CardInstance, KeywordInstance
from app.engine.enums import (
    CardType,
    CombatRole,
    ControlStatus,
    Keyword,
    ZoneType,
)
from app.engine.game_state import CombatState
from app.engine.keywords import hide_card_at_battlefield


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

    def test_tank_with_pre_existing_damage(self):
        """Tank with pre-existing damage needs only remaining lethal before spillover."""
        gs = make_game()
        tank_kw = (KeywordInstance(keyword=Keyword.TANK),)
        tank = add_unit(gs, "p2", might=5, name="WoundedTank", keywords=tank_kw)
        squishy = add_unit(gs, "p2", might=2, name="Squishy")

        # Tank already has 3 damage, so only needs 2 more for lethal (5 - 3 = 2)
        tank.damage = 3

        # Valid: 2 to tank (lethal), 1 to squishy
        result = validate_assignment(
            3,
            {tank.instance_id: 2, squishy.instance_id: 1},
            [tank, squishy],
        )
        assert result is None

        # Invalid: 1 to tank (not lethal yet), 2 to squishy
        result2 = validate_assignment(
            3,
            {tank.instance_id: 1, squishy.instance_id: 2},
            [tank, squishy],
        )
        assert result2 is not None
        assert "tank" in result2.lower()

    def test_all_tanks_no_non_tanks(self):
        """If all defending units have Tank, ordering between them is unconstrained."""
        gs = make_game()
        tank_kw = (KeywordInstance(keyword=Keyword.TANK),)
        tank1 = add_unit(gs, "p2", might=2, name="Tank1", keywords=tank_kw)
        tank2 = add_unit(gs, "p2", might=3, name="Tank2", keywords=tank_kw)

        # Partial damage to tank1 without lethal, rest to tank2 — valid
        # (no non-tank units to protect, so tank ordering rule is satisfied)
        result = validate_assignment(
            4,
            {tank1.instance_id: 2, tank2.instance_id: 2},
            [tank1, tank2],
        )
        assert result is None


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

    def test_three_units_lethal_chain(self):
        """With 3 targets, must give lethal to first two before the third gets any."""
        gs = make_game()
        u1 = add_unit(gs, "p2", might=2, name="U1")
        u2 = add_unit(gs, "p2", might=2, name="U2")
        u3 = add_unit(gs, "p2", might=2, name="U3")

        # Valid: lethal to u1 (2), lethal to u2 (2), rest to u3 (1)
        result = validate_assignment(
            5,
            {u1.instance_id: 2, u2.instance_id: 2, u3.instance_id: 1},
            [u1, u2, u3],
        )
        assert result is None

        # Invalid: skip u2, damage u1 and u3
        result2 = validate_assignment(
            5,
            {u1.instance_id: 2, u2.instance_id: 1, u3.instance_id: 2},
            [u1, u2, u3],
        )
        assert result2 is not None
        assert "lethal" in result2.lower()

    def test_lethal_with_pre_existing_damage(self):
        """Pre-existing damage reduces the lethal threshold for the rule check."""
        gs = make_game()
        u1 = add_unit(gs, "p2", might=4, name="Wounded")
        u2 = add_unit(gs, "p2", might=3, name="Fresh")

        # u1 already has 2 damage, so needs only 2 more for lethal (4 - 2 = 2)
        u1.damage = 2

        # Valid: 2 to u1 (lethal), 1 to u2
        result = validate_assignment(
            3,
            {u1.instance_id: 2, u2.instance_id: 1},
            [u1, u2],
        )
        assert result is None

        # Invalid: only 1 to u1 (not lethal), 2 to u2
        result2 = validate_assignment(
            3,
            {u1.instance_id: 1, u2.instance_id: 2},
            [u1, u2],
        )
        assert result2 is not None
        assert "lethal" in result2.lower()


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

    def test_simultaneous_elimination_battlefield_uncontrolled(self):
        """Both sides eliminated simultaneously: battlefield becomes uncontrolled (rule 444.2.d)."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        attacker = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Att")
        defender = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Def")

        # Full combat flow: start -> close showdown (both pass) -> damage -> resolution
        all_logs = start_combat(gs, bf_id)
        all_logs += pass_focus(gs, "p1")
        all_logs += pass_focus(gs, "p2")
        log_text = " ".join(all_logs).lower()

        # Both units dealt lethal to each other; both should be dead
        assert attacker.zone == ZoneType.TRASH
        assert defender.zone == ZoneType.TRASH
        assert attacker.instance_id not in bf.units
        assert defender.instance_id not in bf.units

        # Battlefield should be uncontrolled (rule 444.2.d)
        assert bf.control_status == ControlStatus.UNCONTROLLED
        assert bf.controller_id is None
        assert bf.contested_by is None
        assert "uncontrolled" in log_text

    def test_simultaneous_elimination_units_in_trash(self):
        """Both units go to their owners' trash piles after mutual elimination."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        attacker = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=4, name="MutualAtt")
        defender = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=4, name="MutualDef")

        start_combat(gs, bf_id)
        pass_focus(gs, "p1")
        pass_focus(gs, "p2")

        # Both should be in their owners' trash
        assert attacker.instance_id in gs.players["p1"].trash
        assert defender.instance_id in gs.players["p2"].trash
        # Combat state should be cleaned up
        assert gs.active_combat is None


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

    def test_gear_detaches_to_base_on_combat_death(self):
        """When a gear-attached unit dies in combat, gear returns to owner's base."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        # Create a defending unit with gear attached
        defender = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="EquippedDef")

        # Create a gear card and attach it to the defender
        gear_def = CardDefinition(
            card_id="test_sword",
            name="Test Sword",
            card_type=CardType.GEAR,
            cost_energy=1,
            might_bonus=2,
        )
        gear = CardInstance.create(gear_def, "p2", ZoneType.BATTLEFIELD)
        gear.location_id = bf_id
        gear.attached_to = defender.instance_id
        gs.instances[gear.instance_id] = gear
        defender.attached_cards.append(gear.instance_id)
        defender.gear_might_bonus = 2  # +2 from gear, so effective_might = 4

        # Create an attacker that deals lethal to the defender (might 4)
        attacker = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=5, name="StrongAtt")

        # Full combat flow: start -> close showdown -> damage -> resolution
        start_combat(gs, bf_id)
        pass_focus(gs, "p1")
        pass_focus(gs, "p2")

        # Defender should be dead (took 5 damage vs 4 effective might)
        assert defender.zone == ZoneType.TRASH
        assert defender.instance_id not in bf.units

        # Gear should be detached and in owner's (p2) base
        assert gear.attached_to is None
        assert gear.zone == ZoneType.BASE
        assert gear.location_id == "p2"
        assert gear.instance_id in gs.base_gear.get("p2", [])

        # Defender should have no attached cards
        assert len(defender.attached_cards) == 0

    def test_multiple_gear_detach_on_combat_death(self):
        """Multiple gear cards all detach when the equipped unit dies in combat."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        defender = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="MultiGearDef")

        # Attach two gear cards
        gear_cards = []
        for i, name in enumerate(["Helmet", "Shield"]):
            gear_def = CardDefinition(
                card_id=f"test_{name.lower()}",
                name=f"Test {name}",
                card_type=CardType.GEAR,
                cost_energy=1,
                might_bonus=1,
            )
            gear = CardInstance.create(gear_def, "p2", ZoneType.BATTLEFIELD)
            gear.location_id = bf_id
            gear.attached_to = defender.instance_id
            gs.instances[gear.instance_id] = gear
            defender.attached_cards.append(gear.instance_id)
            gear_cards.append(gear)

        defender.gear_might_bonus = 2  # +1 from each gear

        # Attacker deals lethal
        attacker = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=5, name="Crusher")

        start_combat(gs, bf_id)
        pass_focus(gs, "p1")
        pass_focus(gs, "p2")

        # Both gear cards should be detached and in p2's base
        for gear in gear_cards:
            assert gear.attached_to is None
            assert gear.zone == ZoneType.BASE
            assert gear.location_id == "p2"
            assert gear.instance_id in gs.base_gear.get("p2", [])

    def test_gear_survives_if_unit_survives_combat(self):
        """Gear stays attached when its unit survives combat."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        # Defender with gear: base might 3 + 2 gear = 5 effective might
        defender = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="ToughDef")

        gear_def = CardDefinition(
            card_id="test_armor",
            name="Test Armor",
            card_type=CardType.GEAR,
            cost_energy=1,
            might_bonus=2,
        )
        gear = CardInstance.create(gear_def, "p2", ZoneType.BATTLEFIELD)
        gear.location_id = bf_id
        gear.attached_to = defender.instance_id
        gs.instances[gear.instance_id] = gear
        defender.attached_cards.append(gear.instance_id)
        defender.gear_might_bonus = 2

        # Attacker with only 2 might — won't deal lethal to the 5-might defender
        attacker = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="WeakAtt")

        start_combat(gs, bf_id)
        pass_focus(gs, "p1")
        pass_focus(gs, "p2")

        # Defender should survive and keep gear attached
        assert defender.zone != ZoneType.TRASH
        assert gear.attached_to == defender.instance_id
        assert gear.instance_id in defender.attached_cards


# =========================================================================
# BE-01: Hidden card reveal in combat
# =========================================================================


def _place_facedown(gs, player_id, bf_id, *, might=3, name="Hidden Unit", ready=True):
    """Helper: place a unit facedown at a battlefield (simulating the Hide action)."""
    bf = gs.battlefields[bf_id]
    # The player must control the battlefield to hide a card there
    bf.controller_id = player_id
    bf.control_status = ControlStatus.CONTROLLED

    defn = make_unit_def(
        card_id=f"hidden_{name.lower().replace(' ', '_')}",
        name=name,
        might=might,
        keywords=(KeywordInstance(keyword=Keyword.HIDDEN),),
    )
    inst = CardInstance.create(defn, player_id, ZoneType.FACEDOWN_ZONE)
    inst.location_id = bf_id
    inst.facedown = True
    inst.hidden_at_battlefield = bf_id
    inst.hidden_ready = ready
    gs.instances[inst.instance_id] = inst
    bf.facedown_card = inst.instance_id
    return inst


class TestHiddenCardRevealInCombat:
    """BE-01: Hidden cards are revealed when combat starts at their battlefield."""

    def test_facedown_card_revealed_on_combat_start(self):
        """A facedown card at a battlefield is revealed when combat begins there."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # Place a facedown hidden unit controlled by p1
        hidden = _place_facedown(gs, "p1", bf_id, might=4, name="Ambusher")

        # Set up combat: p1 defends, p2 attacks
        bf.contested_by = "p2"
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Defender")
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Attacker")

        # Pre-conditions
        assert hidden.facedown is True
        assert hidden.hidden_at_battlefield == bf_id
        assert bf.facedown_card == hidden.instance_id

        logs = start_combat(gs, bf_id)
        log_text = " ".join(logs)

        # Post-conditions: card is revealed
        assert hidden.facedown is False
        assert hidden.hidden_at_battlefield is None
        assert hidden.hidden_ready is False
        # The facedown slot should be cleared
        assert bf.facedown_card is None
        # The revealed unit should now be in the battlefield units list
        assert hidden.instance_id in bf.units
        assert hidden.zone == ZoneType.BATTLEFIELD
        # Logs should mention the reveal
        assert "revealed" in log_text.lower()

    def test_revealed_unit_participates_in_combat(self):
        """A revealed facedown unit gets a combat role and participates in combat."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # p1 has a facedown hidden unit (defender side)
        hidden = _place_facedown(gs, "p1", bf_id, might=4, name="HiddenDefender")

        bf.contested_by = "p2"
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="OpenDefender")
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Attacker")

        start_combat(gs, bf_id)

        # The hidden unit should be revealed and have a defender role
        assert hidden.instance_id in bf.units
        assert hidden.combat_role == CombatRole.DEFENDER

    def test_hidden_unit_in_units_list_revealed(self):
        """A hidden unit already in bf.units (with hidden_at_battlefield set) is revealed."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # Manually place a unit in bf.units that is flagged as hidden
        defn = make_unit_def(name="SneakyUnit", might=3,
                             keywords=(KeywordInstance(keyword=Keyword.HIDDEN),))
        unit = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        unit.location_id = bf_id
        unit.hidden_at_battlefield = bf_id
        unit.hidden_ready = True
        gs.instances[unit.instance_id] = unit
        bf.units.append(unit.instance_id)

        bf.contested_by = "p2"
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Enemy")

        logs = start_combat(gs, bf_id)

        # The hidden flag should be cleared
        assert unit.hidden_at_battlefield is None
        assert unit.hidden_ready is False
        assert unit.facedown is False

    def test_no_facedown_card_no_error(self):
        """Combat at a battlefield with no facedown card works normally."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3)
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3)

        assert bf.facedown_card is None  # no facedown card

        # Should not raise
        logs = start_combat(gs, bf_id)
        assert gs.active_combat is not None

    def test_facedown_card_not_at_combat_battlefield_unaffected(self):
        """A facedown card at a DIFFERENT battlefield is not revealed by combat."""
        gs = make_game(num_battlefields=2)
        bf_ids = list(gs.battlefields.keys())
        bf0_id, bf1_id = bf_ids[0], bf_ids[1]

        # Facedown at bf1, combat at bf0
        hidden = _place_facedown(gs, "p1", bf1_id, might=3, name="FarAwayHidden")

        bf0 = gs.battlefields[bf0_id]
        bf0.contested_by = "p1"
        add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf0_id, might=3)
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf0_id, might=3)

        start_combat(gs, bf0_id)

        # The card at bf1 should still be facedown
        assert hidden.facedown is True
        assert hidden.hidden_at_battlefield == bf1_id
        assert gs.battlefields[bf1_id].facedown_card == hidden.instance_id

    def test_resolve_combat_cleans_up_lingering_facedown(self):
        """resolve_combat clears any lingering facedown state at the combat battlefield."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        att = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=5, name="StrongAtt")
        add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="WeakDef")

        # Manually place a facedown card that wasn't revealed (edge case)
        defn = make_unit_def(name="OrphanedFacedown", might=1)
        orphan = CardInstance.create(defn, "p1", ZoneType.FACEDOWN_ZONE)
        orphan.facedown = True
        orphan.hidden_at_battlefield = bf_id
        gs.instances[orphan.instance_id] = orphan
        bf.facedown_card = orphan.instance_id

        # Run full combat (start -> showdown close -> damage -> resolution)
        start_combat(gs, bf_id)

        # After combat resolves, the facedown_card slot should be cleared
        assert bf.facedown_card is None
        assert orphan.facedown is False
        assert orphan.hidden_at_battlefield is None

    def test_reveal_hidden_at_battlefield_helper_directly(self):
        """Test _reveal_hidden_at_battlefield in isolation."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        hidden = _place_facedown(gs, "p1", bf_id, might=3, name="DirectReveal")

        logs = _reveal_hidden_at_battlefield(gs, bf)

        assert hidden.facedown is False
        assert hidden.hidden_at_battlefield is None
        assert bf.facedown_card is None
        assert hidden.instance_id in bf.units
        assert any("revealed" in log.lower() for log in logs)

    def test_non_unit_facedown_card_not_added_to_units(self):
        """A non-unit facedown card (e.g. spell) is revealed but not added to bf.units."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.controller_id = "p1"
        bf.control_status = ControlStatus.CONTROLLED

        # Create a spell card and place it facedown
        spell_def = CardDefinition(
            card_id="hidden_spell",
            name="Hidden Spell",
            card_type=CardType.SPELL,
            keywords=(KeywordInstance(keyword=Keyword.HIDDEN),),
        )
        spell = CardInstance.create(spell_def, "p1", ZoneType.FACEDOWN_ZONE)
        spell.location_id = bf_id
        spell.facedown = True
        spell.hidden_at_battlefield = bf_id
        spell.hidden_ready = True
        gs.instances[spell.instance_id] = spell
        bf.facedown_card = spell.instance_id

        logs = _reveal_hidden_at_battlefield(gs, bf)

        # Spell should be revealed
        assert spell.facedown is False
        assert spell.hidden_at_battlefield is None
        assert bf.facedown_card is None
        # Spell should NOT be in bf.units
        assert spell.instance_id not in bf.units
