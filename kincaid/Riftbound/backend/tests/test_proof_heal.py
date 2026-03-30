"""Proof scenarios for the HEAL effect.

Each test proves that heal spells produce correct frontend-visible results:
  1. Set up a game state with a damaged unit
  2. Play a heal spell through the full validate->execute->cleanup pipeline
  3. Assert on BOTH the game state AND the serialized frontend view

If any test here fails, either:
  - The engine computed wrong state (bug in heal logic), or
  - The serializer produced wrong output (bug in the contract boundary), or
  - The contract schema is wrong (update contracts/state_update.schema.json)
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.effect_ir import TargetSpec, heal
from app.engine.enums import ActionType, ZoneType


class TestHealProof:
    """Proof: heal effects reduce damage correctly and results are frontend-visible."""

    def test_heal_reduces_damage(self):
        """Proof: a heal(3) spell on a unit with damage=4 reduces damage
        to 1, visible in both game state and frontend view."""
        s = Scenario()
        target = add_unit(s.gs, "p1", ZoneType.BASE, might=6, name="Wounded")
        target.damage = 4

        spell = s.add_hand_spell(
            "p1", name="Mend", cost_energy=1,
            effect_ir=heal(3, TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play spell
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        # Resolve (both pass priority)
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: damage reduced from 4 to 1
        assert target.damage == 1

        # Frontend view: damage is 1, effective_might unchanged at 6
        target_view = Scenario.find_card_in_view(s.view("p1"), target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 1
        assert target_view["effective_might"] == 6

    def test_heal_all_removes_all_damage(self):
        """Proof: a heal('all') spell on a unit with damage=5 removes all
        damage, visible in both game state and frontend view."""
        s = Scenario()
        target = add_unit(s.gs, "p1", ZoneType.BASE, might=6, name="Battered")
        target.damage = 5

        spell = s.add_hand_spell(
            "p1", name="Full Restore", cost_energy=1,
            effect_ir=heal("all", TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play spell
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        # Resolve (both pass priority)
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: damage fully removed
        assert target.damage == 0

        # Frontend view: damage is 0, effective_might unchanged at 6
        target_view = Scenario.find_card_in_view(s.view("p1"), target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 0
        assert target_view["effective_might"] == 6

    def test_heal_cannot_go_below_zero(self):
        """Proof: healing for more than current damage clamps damage to 0
        (not negative), visible in the frontend view."""
        s = Scenario()
        target = add_unit(s.gs, "p1", ZoneType.BASE, might=6, name="Scratched")
        target.damage = 2

        spell = s.add_hand_spell(
            "p1", name="Overheal", cost_energy=1,
            effect_ir=heal(5, TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play spell
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        # Resolve (both pass priority)
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: damage is 0, not negative
        assert target.damage == 0

        # Frontend view: damage clamped to 0
        target_view = Scenario.find_card_in_view(s.view("p1"), target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 0
        assert target_view["effective_might"] == 6
