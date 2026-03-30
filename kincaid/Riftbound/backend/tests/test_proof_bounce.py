"""Proof scenarios for RETURN TO HAND (bounce) effects.

Each test proves that bouncing a unit via a spell:
  1. Moves the unit to the owner's hand in game state
  2. Clears transient state (damage, exhaustion, stun)
  3. Updates the serialized frontend view correctly

If any test fails, either:
  - The return_to_hand primitive computed wrong state, or
  - The serializer produced wrong output for zone transitions, or
  - The contract schema is wrong
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.effect_ir import TargetSpec, return_to_hand
from app.engine.enums import ActionType, ZoneType


class TestBounceProof:
    """Proof: return_to_hand moves a unit back to the owner's hand."""

    def test_bounce_moves_unit_to_hand(self):
        """Proof: a bounce spell targeting a friendly unit in base removes it
        from base and places it in the owner's hand, visible in both game
        state and the frontend view."""
        s = Scenario()
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=4, name="Recalled")
        spell = s.add_hand_spell(
            "p1", name="Bounce", cost_energy=1,
            effect_ir=return_to_hand(TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play the bounce spell targeting the unit
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [unit.instance_id],
        })
        # Resolve: both players pass priority
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: unit is in hand
        assert unit.zone == ZoneType.HAND
        assert unit.instance_id in s.gs.players["p1"].hand

        # Game state: unit is no longer in base
        assert unit.instance_id not in s.gs.base_units["p1"]

        # Frontend view (p1): unit appears in hand with full details
        p1_view = s.view("p1")
        hand_ids = [c["instance_id"] for c in p1_view["you"]["hand"]]
        assert unit.instance_id in hand_ids

        hand_card = next(c for c in p1_view["you"]["hand"]
                         if c["instance_id"] == unit.instance_id)
        assert hand_card["name"] == "Recalled"
        assert hand_card["base_might"] == 4

        # Frontend view (p1): unit no longer in base_units
        base_ids = [c["instance_id"] for c in p1_view["you"]["base_units"]]
        assert unit.instance_id not in base_ids

    def test_bounce_clears_damage(self):
        """Proof: a unit with 3 damage returns to hand with 0 damage.
        The return_to_hand primitive resets transient state (damage,
        exhaustion, stun) when moving the unit."""
        s = Scenario()
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=5, name="Wounded")
        # Simulate combat damage and exhaustion
        unit.damage = 3
        unit.exhausted = True
        unit.stunned = True

        spell = s.add_hand_spell(
            "p1", name="Recall", cost_energy=1,
            effect_ir=return_to_hand(TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [unit.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: all transient state is cleared
        assert unit.zone == ZoneType.HAND
        assert unit.damage == 0
        assert unit.exhausted is False
        assert unit.stunned is False
        assert unit.buff_counter is False

        # Frontend view: card in hand shows 0 damage
        p1_view = s.view("p1")
        hand_card = next(c for c in p1_view["you"]["hand"]
                         if c["instance_id"] == unit.instance_id)
        assert hand_card["damage"] == 0

    def test_bounced_unit_leaves_battlefield(self):
        """Proof: a unit on a battlefield disappears from the battlefield
        view and appears in the owner's hand after being bounced."""
        s = Scenario()
        unit = add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=s.bf0,
                        might=3, name="Deployed")

        spell = s.add_hand_spell(
            "p1", name="Withdraw", cost_energy=1,
            effect_ir=return_to_hand(TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Verify unit is on battlefield before bounce
        pre_view = s.view("p1")
        bf_unit_ids_before = [c["instance_id"]
                              for c in pre_view["battlefields"][s.bf0]["units"]]
        assert unit.instance_id in bf_unit_ids_before

        # Play and resolve the bounce spell
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [unit.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: unit moved from battlefield to hand
        assert unit.zone == ZoneType.HAND
        assert unit.instance_id not in s.gs.battlefields[s.bf0].units
        assert unit.instance_id in s.gs.players["p1"].hand

        # Frontend view: unit is gone from battlefield
        p1_view = s.view("p1")
        bf_unit_ids_after = [c["instance_id"]
                             for c in p1_view["battlefields"][s.bf0]["units"]]
        assert unit.instance_id not in bf_unit_ids_after

        # Frontend view: unit appears in hand
        hand_ids = [c["instance_id"] for c in p1_view["you"]["hand"]]
        assert unit.instance_id in hand_ids
