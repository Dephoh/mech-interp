"""Proof scenarios for the STUN effect.

Each test proves that stunning a unit produces correct game state AND
correct frontend-visible results through the serialize pipeline.

Stun rules (from card_types.py):
  - ``combat_might`` returns 0 when ``stunned is True``
  - ``effective_might`` is NOT affected by stun
  - ``stunned`` clears at end of turn (clear_turn_state)
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.effect_ir import stun, TargetSpec
from app.engine.enums import ActionType, ZoneType


# =========================================================================
# Stun effect
# =========================================================================


class TestStunProof:
    """Proof: stunning a unit zeroes combat_might but preserves effective_might,
    and the stunned state is visible to both players in the frontend."""

    def test_stun_sets_combat_might_zero(self):
        """Proof: playing a stun spell on an enemy unit sets stunned=True,
        which makes combat_might=0 in both the game state and the
        serialized frontend view."""
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Stunnable")
        spell = s.add_hand_spell(
            "p1", name="Freeze", cost_energy=1,
            effect_ir=stun(TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play the stun spell targeting the enemy unit
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        # Resolve: both players pass priority
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: target is stunned, combat_might is 0
        assert target.stunned is True
        assert target.combat_might == 0

        # Frontend view: stunned flag and combat_might visible
        target_view = Scenario.find_card_in_view(view, target.instance_id)
        assert target_view is not None
        assert target_view["stunned"] is True
        assert target_view["combat_might"] == 0

    def test_stunned_unit_keeps_effective_might(self):
        """Proof: stun only zeroes combat_might; effective_might stays at
        the unit's actual might value (base + modifiers)."""
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=4, name="Sturdy")
        spell = s.add_hand_spell(
            "p1", name="Ice Bolt", cost_energy=1,
            effect_ir=stun(TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play and resolve the stun spell
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: effective_might unchanged, combat_might zeroed
        assert target.stunned is True
        assert target.effective_might == 4
        assert target.combat_might == 0

        # Frontend view: effective_might still shows the real value
        p1_view = s.view("p1")
        target_view = Scenario.find_card_in_view(p1_view, target.instance_id)
        assert target_view is not None
        assert target_view["effective_might"] == 4
        assert target_view["combat_might"] == 0

    def test_stun_visible_to_both_players(self):
        """Proof: after stunning an enemy unit, both the caster's view
        and the target owner's view show stunned=True."""
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=3, name="Visible")
        spell = s.add_hand_spell(
            "p1", name="Frost Touch", cost_energy=1,
            effect_ir=stun(TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play and resolve the stun spell
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Caster's view (p1): sees stunned on opponent's unit
        p1_view = s.view("p1")
        target_in_p1 = Scenario.find_card_in_view(p1_view, target.instance_id)
        assert target_in_p1 is not None
        assert target_in_p1["stunned"] is True

        # Target owner's view (p2): sees stunned on own unit
        p2_view = s.view("p2")
        target_in_p2 = Scenario.find_card_in_view(p2_view, target.instance_id)
        assert target_in_p2 is not None
        assert target_in_p2["stunned"] is True
