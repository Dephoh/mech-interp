"""Tests for gear attachment, might bonus, and detachment on death."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engine.card_types import CardDefinition, CardInstance
from app.engine.enums import CardType, ZoneType
from app.engine.effect_primitives import prim_attach, prim_detach, prim_kill

from helpers import add_unit, make_game


def _add_gear(gs, player_id, name="Sword", might_bonus=3):
    """Add a gear card to a player's base and return it."""
    defn = CardDefinition(
        card_id=f"gear_{name.lower()}",
        name=name,
        card_type=CardType.GEAR,
        might_bonus=might_bonus,
    )
    inst = CardInstance.create(defn, player_id, ZoneType.BASE)
    gs.instances[inst.instance_id] = inst
    gs.base_gear[player_id].append(inst.instance_id)
    return inst


class TestGearMightBonus:
    def test_attached_gear_boosts_effective_might(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _add_gear(gs, "p1", "B.F. Sword", might_bonus=3)

        # Attach
        prim_attach(unit, gs, {}, [gear.instance_id, unit.instance_id])
        gs.recalculate_modifiers()

        assert unit.gear_might_bonus == 3
        assert unit.effective_might == 6  # base 3 + gear 3

    def test_multiple_gear_stack(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=2, name="Soldier")
        g1 = _add_gear(gs, "p1", "Sword", might_bonus=2)
        g2 = _add_gear(gs, "p1", "Shield", might_bonus=1)

        prim_attach(unit, gs, {}, [g1.instance_id, unit.instance_id])
        prim_attach(unit, gs, {}, [g2.instance_id, unit.instance_id])
        gs.recalculate_modifiers()

        assert unit.gear_might_bonus == 3  # 2 + 1
        assert unit.effective_might == 5  # base 2 + gear 3

    def test_no_bonus_without_attachment(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        _add_gear(gs, "p1", "Sword", might_bonus=3)  # gear exists but not attached

        gs.recalculate_modifiers()
        assert unit.gear_might_bonus == 0
        assert unit.effective_might == 3


class TestGearDetach:
    def test_detach_removes_bonus(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _add_gear(gs, "p1", "Sword", might_bonus=3)

        prim_attach(unit, gs, {}, [gear.instance_id, unit.instance_id])
        gs.recalculate_modifiers()
        assert unit.effective_might == 6

        prim_detach(unit, gs, {}, [gear.instance_id])
        gs.recalculate_modifiers()
        assert unit.gear_might_bonus == 0
        assert unit.effective_might == 3


class TestGearDetachOnDeath:
    def test_gear_returns_to_base_when_unit_dies_in_cleanup(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        unit = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=3, name="Soldier", bf_id=bf_id)
        gear = _add_gear(gs, "p1", "Sword", might_bonus=3)

        prim_attach(unit, gs, {}, [gear.instance_id, unit.instance_id])
        # Remove gear from base_gear since it's now attached
        gs.base_gear["p1"].remove(gear.instance_id)

        # Deal lethal damage
        unit.damage = 6

        from app.engine.cleanup import run_cleanup
        logs = run_cleanup(gs)

        # Gear should be back in base
        assert gear.attached_to is None
        assert gear.zone == ZoneType.BASE
        assert gear.instance_id in gs.base_gear["p1"]
        assert any("detached" in l for l in logs)

    def test_gear_returns_to_base_when_unit_dies_in_combat(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        unit = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=3, name="Soldier", bf_id=bf_id)
        enemy = add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=5, name="Enemy", bf_id=bf_id)
        gear = _add_gear(gs, "p1", "Sword", might_bonus=2)

        prim_attach(unit, gs, {}, [gear.instance_id, unit.instance_id])
        gs.base_gear["p1"].remove(gear.instance_id)

        from app.engine.combat import start_combat
        from app.engine.enums import CombatRole
        from app.engine.game_state import CombatState

        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        # Start combat and run through
        logs = start_combat(gs, bf_id)

        # Both pass showdown
        from app.engine.combat import pass_focus
        pass_focus(gs, "p1")
        logs = pass_focus(gs, "p2")

        # Unit with 3 might vs enemy with 5 might — unit dies
        # Check gear is detached
        assert gear.attached_to is None
        assert gear.zone == ZoneType.BASE
        assert gear.instance_id in gs.base_gear["p1"]


class TestKillGear:
    def test_prim_kill_destroys_gear_in_base(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Attacker")
        gear = _add_gear(gs, "p1", "Sword", might_bonus=3)

        logs = prim_kill(source, gs, {}, [gear.instance_id])
        assert gear.zone == ZoneType.TRASH
        assert gear.instance_id not in gs.base_gear["p1"]
        assert gear.instance_id in gs.players["p1"].trash
        assert any("destroyed" in l for l in logs)

    def test_prim_kill_detaches_gear_from_unit(self):
        gs = make_game()
        source = add_unit(gs, "p2", ZoneType.BASE, might=3, name="Attacker")
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _add_gear(gs, "p1", "Sword", might_bonus=3)

        prim_attach(unit, gs, {}, [gear.instance_id, unit.instance_id])
        gs.base_gear["p1"].remove(gear.instance_id)

        logs = prim_kill(source, gs, {}, [gear.instance_id])
        assert gear.zone == ZoneType.TRASH
        assert gear.attached_to is None
        assert gear.instance_id not in unit.attached_cards

    def test_prim_kill_still_works_on_units(self):
        gs = make_game()
        source = add_unit(gs, "p2", ZoneType.BASE, might=3, name="Attacker")
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Target")

        logs = prim_kill(source, gs, {}, [unit.instance_id])
        # Unit gets lethal damage (cleanup will kill it)
        assert unit.damage >= unit.effective_might
        assert any("killed" in l for l in logs)
