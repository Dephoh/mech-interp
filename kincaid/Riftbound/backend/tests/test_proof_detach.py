"""Proof scenarios: gear detachment on unit death (rule 719.5).

Rule 719.5: When a Top-Most Card changes zones from a board zone to a
non-board zone, all Attached cards Detach from it, remaining in their
current zones.

These tests verify that when a unit with attached gear dies (via kill spell
or lethal damage), the gear detaches correctly and remains on the board.
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.card_types import CardDefinition, CardInstance, CardType
from app.engine.effect_ir import deal_damage, kill, TargetSpec
from app.engine.enums import ActionType, Phase, ZoneType


def _attach_gear(s: Scenario, unit: CardInstance, player_id: str,
                 name: str = "Sword", might_bonus: int = 2) -> CardInstance:
    """Create a gear card in the player's base and attach it to a unit."""
    gear_def = CardDefinition(
        card_id=f"test_{name.lower().replace(' ', '_')}",
        name=name,
        card_type=CardType.GEAR,
        might_bonus=might_bonus,
    )
    gear = CardInstance.create(gear_def, player_id, ZoneType.BASE)
    s.gs.instances[gear.instance_id] = gear
    s.gs.base_gear[player_id].append(gear.instance_id)
    # Attach
    gear.attached_to = unit.instance_id
    unit.attached_cards.append(gear.instance_id)
    s.gs.recalculate_modifiers()
    return gear


class TestGearDetachOnDeath:
    """Proof: gear detaches when its host unit dies (rule 719.5)."""

    def test_gear_detaches_when_unit_killed(self):
        """Proof: a kill spell targeting a unit with attached gear moves
        the unit to trash while the gear remains on the board in base_gear
        with attached_to=None, visible in the frontend view."""
        s = Scenario()
        unit = add_unit(s.gs, "p2", ZoneType.BASE, might=3, name="Equipped")
        gear = _attach_gear(s, unit, "p2", name="Sword", might_bonus=2)

        # Verify gear is attached before the kill
        assert gear.attached_to == unit.instance_id
        assert gear.instance_id in unit.attached_cards

        # Play a kill spell targeting the unit
        spell = s.add_hand_spell(
            "p1", name="Execute", cost_energy=2,
            effect_ir=kill(TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [unit.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: unit is in trash
        assert unit.zone == ZoneType.TRASH

        # Game state: gear detached and remains on board
        assert gear.attached_to is None
        assert gear.instance_id in s.gs.base_gear["p2"]
        assert gear.zone == ZoneType.BASE

        # Frontend view: gear visible in p2's base_gear with attached_to=None
        view = s.view("p2")
        gear_views = view["you"]["base_gear"]
        gear_ids = [g["instance_id"] for g in gear_views]
        assert gear.instance_id in gear_ids
        gear_view = next(g for g in gear_views if g["instance_id"] == gear.instance_id)
        assert gear_view["attached_to"] is None

        # Frontend view: unit is gone from base_units, present in trash
        base_unit_ids = [c["instance_id"] for c in view["you"]["base_units"]]
        assert unit.instance_id not in base_unit_ids
        trash_ids = [c["instance_id"] for c in view["you"]["trash"]]
        assert unit.instance_id in trash_ids

    def test_gear_might_bonus_removed_after_detach(self):
        """Proof: after the host unit dies and gear detaches, the gear's
        might bonus no longer applies. A second unit does not inherit
        the detached gear's bonus."""
        s = Scenario()
        unit = add_unit(s.gs, "p2", ZoneType.BASE, might=3, name="Equipped")
        gear = _attach_gear(s, unit, "p2", name="Blade", might_bonus=2)

        # Before kill: unit has gear bonus (3 base + 2 gear = 5)
        assert unit.effective_might == 5

        # Second unit -- should never get the bonus
        bystander = add_unit(s.gs, "p2", ZoneType.BASE, might=4, name="Bystander")
        assert bystander.effective_might == 4

        # Kill the equipped unit
        spell = s.add_hand_spell(
            "p1", name="Smite", cost_energy=2,
            effect_ir=kill(TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [unit.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: gear detached
        assert gear.attached_to is None

        # Game state: bystander still has only its base might (no stray bonus)
        assert bystander.effective_might == 4
        assert bystander.gear_might_bonus == 0

        # Frontend view: bystander shows correct might
        view = s.view("p2")
        bystander_view = Scenario.find_card_in_view(view, bystander.instance_id)
        assert bystander_view is not None
        assert bystander_view["effective_might"] == 4

    def test_lethal_damage_triggers_detach(self):
        """Proof: dealing lethal damage (not a kill spell) to a unit with
        gear causes the gear to detach after cleanup processes the death.
        The gear remains on the board and the unit moves to trash."""
        s = Scenario()
        unit = add_unit(s.gs, "p2", ZoneType.BASE, might=3, name="Armored")
        gear = _attach_gear(s, unit, "p2", name="Shield", might_bonus=2)

        # With gear: effective might = 3 + 2 = 5
        assert unit.effective_might == 5
        assert gear.attached_to == unit.instance_id

        # Deal 5 damage (lethal: damage >= effective_might)
        spell = s.add_hand_spell(
            "p1", name="Fireball", cost_energy=3,
            effect_ir=deal_damage(5, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [unit.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: unit died and moved to trash
        assert unit.zone == ZoneType.TRASH

        # Game state: gear detached and remains on the board
        assert gear.attached_to is None
        assert gear.instance_id in s.gs.base_gear["p2"]
        assert gear.zone == ZoneType.BASE

        # Frontend view: gear in base_gear, detached
        view = s.view("p2")
        gear_views = view["you"]["base_gear"]
        gear_ids = [g["instance_id"] for g in gear_views]
        assert gear.instance_id in gear_ids
        gear_view = next(g for g in gear_views if g["instance_id"] == gear.instance_id)
        assert gear_view["attached_to"] is None

        # Frontend view: unit in trash
        trash_ids = [c["instance_id"] for c in view["you"]["trash"]]
        assert unit.instance_id in trash_ids
