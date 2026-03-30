"""Proof scenarios for AoE (area-of-effect) damage spells.

Each test proves that spells dealing damage to multiple units at once
produce the correct frontend-visible results:
  - All targeted units receive damage marks
  - Units with lethal damage are killed and moved to trash by cleanup
  - scope="enemy" hits only opponent units; scope="any" hits both sides
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.effect_ir import TargetSpec, deal_damage
from app.engine.enums import ActionType, ZoneType


class TestAoEDamageProof:
    """Proof: AoE damage spells apply damage to the correct set of units."""

    def test_aoe_damages_all_enemies(self):
        """Proof: a spell with deal_damage(2, scope='enemy', count=-1) deals 2
        damage to ALL enemy units and 0 damage to friendly units."""
        s = Scenario()

        # Place 3 enemy units in p2's base
        enemy1 = add_unit(s.gs, "p2", ZoneType.BASE, might=4, name="Enemy A")
        enemy2 = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Enemy B")
        enemy3 = add_unit(s.gs, "p2", ZoneType.BASE, might=6, name="Enemy C")

        # Place 1 friendly unit to verify it is NOT hit
        friendly = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Friendly")

        spell = s.add_hand_spell(
            "p1", name="Firestorm", cost_energy=4,
            effect_ir=deal_damage(2, TargetSpec(obj_type="unit", scope="enemy", count=-1)),
        )
        s.give_resources("p1", energy=10)

        # Play spell (goes on chain)
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        # Resolve: both players pass priority
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: all 3 enemies have damage=2, friendly has damage=0
        assert enemy1.damage == 2
        assert enemy2.damage == 2
        assert enemy3.damage == 2
        assert friendly.damage == 0

        # Frontend view: all enemy units show damage=2
        for enemy in (enemy1, enemy2, enemy3):
            card_view = Scenario.find_card_in_view(view, enemy.instance_id)
            assert card_view is not None, f"{enemy.name} not found in view"
            assert card_view["damage"] == 2, f"{enemy.name} expected damage=2"

        # Frontend view: friendly unit shows damage=0
        friendly_view = Scenario.find_card_in_view(view, friendly.instance_id)
        assert friendly_view is not None
        assert friendly_view["damage"] == 0

    def test_aoe_kills_weak_units(self):
        """Proof: AoE damage that exceeds a unit's might kills it (moved to
        trash by cleanup), while tougher units survive with damage marks."""
        s = Scenario()

        # Weak unit: might=2 — will die to 3 damage
        weak = add_unit(s.gs, "p2", ZoneType.BASE, might=2, name="Weak")
        # Tough unit: might=5 — survives 3 damage
        tough = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Tough")

        spell = s.add_hand_spell(
            "p1", name="Inferno", cost_energy=3,
            effect_ir=deal_damage(3, TargetSpec(obj_type="unit", scope="enemy", count=-1)),
        )
        s.give_resources("p1", energy=10)

        # Play and resolve
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: weak unit is dead (damage >= might), tough survived
        assert weak.zone == ZoneType.TRASH
        assert tough.zone == ZoneType.BASE
        assert tough.damage == 3

        # Frontend: weak unit is in opponent's trash, not in base
        p1_view = s.view("p1")
        opp_base_ids = [c["instance_id"] for c in p1_view["opponent"]["base_units"]]
        assert weak.instance_id not in opp_base_ids
        opp_trash_ids = [c["instance_id"] for c in p1_view["opponent"]["trash"]]
        assert weak.instance_id in opp_trash_ids

        # Frontend: tough unit is still in base with damage=3
        tough_view = Scenario.find_card_in_view(p1_view, tough.instance_id)
        assert tough_view is not None
        assert tough_view["damage"] == 3
        assert tough_view["effective_might"] == 5

    def test_aoe_damages_both_sides(self):
        """Proof: a spell with scope='any' and count=-1 deals damage to ALL
        units on both sides of the board."""
        s = Scenario()

        # 2 friendly units
        ally1 = add_unit(s.gs, "p1", ZoneType.BASE, might=4, name="Ally A")
        ally2 = add_unit(s.gs, "p1", ZoneType.BASE, might=5, name="Ally B")

        # 2 enemy units
        foe1 = add_unit(s.gs, "p2", ZoneType.BASE, might=4, name="Foe A")
        foe2 = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Foe B")

        spell = s.add_hand_spell(
            "p1", name="Earthquake", cost_energy=5,
            effect_ir=deal_damage(2, TargetSpec(obj_type="unit", scope="any", count=-1)),
        )
        s.give_resources("p1", energy=10)

        # Play and resolve
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: all 4 units have damage=2
        for unit in (ally1, ally2, foe1, foe2):
            assert unit.damage == 2, f"{unit.name} expected damage=2, got {unit.damage}"

        # Frontend: verify via p1's view — friendly units show damage
        p1_view = s.view("p1")
        for unit in (ally1, ally2, foe1, foe2):
            card_view = Scenario.find_card_in_view(p1_view, unit.instance_id)
            assert card_view is not None, f"{unit.name} not found in p1 view"
            assert card_view["damage"] == 2, (
                f"{unit.name} expected damage=2 in view, got {card_view['damage']}"
            )
