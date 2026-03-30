"""Proof scenarios for CONDITIONAL effects.

Each test proves that a conditional IR node evaluates its condition
and executes the correct branch (then vs else), with results visible
in the serialized frontend view.

Conditions tested:
  - unit_count_gte (true when the controller has >= N friendly units)
  - The same condition forced false via an unreachable threshold

If any test here fails, either:
  - The condition evaluator returned the wrong boolean, or
  - The wrong branch (then/else) was executed, or
  - The serializer produced wrong output for the resulting state
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.effect_ir import (
    ConditionSpec,
    TargetSpec,
    conditional,
    deal_damage,
    draw_cards,
)
from app.engine.enums import ActionType, ZoneType


class TestConditionalProof:
    """Proof: conditional effects choose the correct branch."""

    def test_conditional_true_branch(self):
        """Proof: when the condition is true, the THEN branch executes.

        Setup: p1 has a friendly unit in base (unit_count_gte threshold=1
        is satisfied). Spell with conditional deals 5 damage on true.
        Assert: target takes 5 damage, visible in frontend view.
        """
        s = Scenario()

        # Place a friendly unit so unit_count_gte(1, friendly) is true
        add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Ally")

        # Enemy target to receive damage
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=8, name="Target")

        ir = conditional(
            ConditionSpec(
                cond_type="unit_count_gte",
                params={"threshold": 1, "scope": "friendly"},
            ),
            then=deal_damage(5, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            else_=deal_damage(1, TargetSpec(obj_type="unit", scope="enemy", count=1)),
        )
        spell = s.add_hand_spell(
            "p1", name="Conditional Bolt", cost_energy=2,
            effect_ir=ir,
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play spell, resolve chain
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: THEN branch dealt 5 damage (not 1)
        assert target.damage == 5

        # Frontend: damage visible on the target
        target_view = Scenario.find_card_in_view(view, target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 5

    def test_conditional_false_branch(self):
        """Proof: when the condition is false, the ELSE branch executes.

        Setup: p1 has no units meeting threshold (unit_count_gte with
        threshold=99 is impossible). Spell with conditional draws 1 card
        on the else branch.
        Assert: p1 draws a card, visible in frontend hand count.
        """
        s = Scenario()

        # Deck cards so draw can succeed
        s.add_deck_cards("p1", 5)

        # Enemy target (won't be damaged if else branch fires)
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=8, name="Target")

        ir = conditional(
            ConditionSpec(
                cond_type="unit_count_gte",
                params={"threshold": 99, "scope": "friendly"},
            ),
            then=deal_damage(5, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            else_=draw_cards(1),
        )
        spell = s.add_hand_spell(
            "p1", name="Conditional Study", cost_energy=2,
            effect_ir=ir,
            targets_required=1,
        )
        s.give_resources("p1", energy=5)
        hand_before = len(s.gs.players["p1"].hand)  # includes the spell

        # Play spell, resolve chain
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: ELSE branch fired — drew 1 card, no damage
        assert target.damage == 0
        # Hand: lost spell (-1), drew 1 (+1) = net 0 change
        expected_hand = hand_before - 1 + 1
        assert len(s.gs.players["p1"].hand) == expected_hand

        # Frontend: target undamaged, hand count correct
        target_view = Scenario.find_card_in_view(view, target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 0

        p1_view = s.view("p1")
        assert p1_view["you"]["hand_count"] == expected_hand

    def test_conditional_no_else_false(self):
        """Proof: when the condition is false and there is no else branch,
        nothing happens.

        Setup: p1 has no units meeting threshold (unit_count_gte with
        threshold=99). Conditional has no else branch.
        Assert: target takes 0 damage, game state unchanged.
        """
        s = Scenario()

        target = add_unit(s.gs, "p2", ZoneType.BASE, might=8, name="Target")

        ir = conditional(
            ConditionSpec(
                cond_type="unit_count_gte",
                params={"threshold": 99, "scope": "friendly"},
            ),
            then=deal_damage(5, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            # no else_ branch
        )
        spell = s.add_hand_spell(
            "p1", name="Conditional Whiff", cost_energy=2,
            effect_ir=ir,
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play spell, resolve chain
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: no branch executed — target undamaged
        assert target.damage == 0

        # Frontend: target shows 0 damage
        target_view = Scenario.find_card_in_view(view, target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 0
