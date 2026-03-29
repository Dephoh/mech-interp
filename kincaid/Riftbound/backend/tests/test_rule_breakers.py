"""Tests for Task 1–3 rule-breaker nodes: modifier, replacement, player_choice."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engine.action_executor import execute_action
from app.engine.card_types import CardInstance
from app.engine.effect_ir import (
    MODIFIER,
    PLAYER_CHOICE,
    REPLACEMENT,
    TargetSpec,
    modifier,
    player_choice,
    replacement,
    sequence,
    deal_damage,
    heal,
)
from app.engine.effect_resolver import resolve_effect_ir
from app.engine.enums import ActionType, Phase, ZoneType
from app.engine.game_state import ActiveModifier, ActiveReplacement
from app.protocol.messages import (
    ChoiceRequired,
    SubmitChoice,
    parse_client_message,
)

from helpers import add_unit, make_game


# ---------------------------------------------------------------------------
# Modifier (aura) tests
# ---------------------------------------------------------------------------


class TestModifierNode:
    def test_modifier_registers_on_game_state(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=3, name="Warlord",
                          bf_id=list(gs.battlefields.keys())[0])

        ir = modifier("might", 1, target=TargetSpec(scope="friendly"), exclude_self=True)
        logs = resolve_effect_ir(ir, source, gs)

        assert len(gs.active_modifiers) == 1
        assert gs.active_modifiers[0].stat == "might"
        assert gs.active_modifiers[0].amount == 1
        assert "+1 might aura active" in logs[0]

    def test_modifier_boosts_friendly_units(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        source = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=3, name="Warlord", bf_id=bf_id)
        ally = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=2, name="Soldier", bf_id=bf_id)
        enemy = add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=4, name="Enemy", bf_id=bf_id)

        ir = modifier("might", 1, target=TargetSpec(scope="friendly"), exclude_self=True)
        resolve_effect_ir(ir, source, gs)

        # Ally gets +1, source excluded, enemy unaffected
        assert ally.aura_might_bonus == 1
        assert ally.effective_might == 3  # base 2 + 1 aura
        assert source.aura_might_bonus == 0
        assert enemy.aura_might_bonus == 0

    def test_modifier_removed_when_source_leaves(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        source = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=3, name="Warlord", bf_id=bf_id)
        ally = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=2, name="Soldier", bf_id=bf_id)

        ir = modifier("might", 1, target=TargetSpec(scope="friendly"), exclude_self=True)
        resolve_effect_ir(ir, source, gs)
        assert ally.aura_might_bonus == 1

        # Simulate source leaving the board
        gs.unregister_card(source.instance_id)
        gs.recalculate_modifiers()
        assert ally.aura_might_bonus == 0

    def test_modifier_does_not_double_register(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Warlord")
        ir = modifier("might", 1, target=TargetSpec(scope="friendly"))
        resolve_effect_ir(ir, source, gs)
        resolve_effect_ir(ir, source, gs)  # second call
        assert len(gs.active_modifiers) == 1

    def test_modifier_self_only(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Buffed")
        ir = modifier("might", 2, target=TargetSpec(scope="self"))
        resolve_effect_ir(ir, source, gs)
        assert source.aura_might_bonus == 2
        assert source.effective_might == 5  # base 3 + 2 aura


# ---------------------------------------------------------------------------
# Replacement (interception) tests
# ---------------------------------------------------------------------------


class TestReplacementNode:
    def test_replacement_registers_on_game_state(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Guardian")
        ir = replacement(
            "deal_damage",
            condition={"cond_type": "always"},
            target=TargetSpec(scope="friendly").to_dict(),
        )
        logs = resolve_effect_ir(ir, source, gs)
        assert len(gs.active_replacements) == 1
        assert "replacement effect active" in logs[0]

    def test_replacement_intercepts_damage(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        guardian = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=5, name="Guardian", bf_id=bf_id)
        protectee = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=3, name="Protectee", bf_id=bf_id)
        attacker = add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=4, name="Attacker", bf_id=bf_id)

        # Register replacement: when friendly unit would take damage, heal it instead
        rep_ir = replacement(
            "deal_damage",
            condition={"cond_type": "always"},
            alternative_ir=heal(amount="all"),
            target=TargetSpec(scope="friendly").to_dict(),
        )
        resolve_effect_ir(rep_ir, guardian, gs)

        # Now try to deal damage to protectee — should be intercepted
        damage_ir = deal_damage(3, target=TargetSpec(scope="friendly"))
        logs = resolve_effect_ir(damage_ir, attacker, gs, [protectee.instance_id])

        # Protectee should not have taken damage (replacement healed instead)
        # The heal on a fresh unit with 0 damage heals 0
        assert protectee.damage == 0

    def test_replacement_does_not_intercept_enemy(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        guardian = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=5, name="Guardian", bf_id=bf_id)
        enemy = add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=3, name="Enemy", bf_id=bf_id)

        # Replacement only protects friendly units
        rep_ir = replacement(
            "deal_damage",
            condition={"cond_type": "always"},
            alternative_ir=heal(amount="all"),
            target=TargetSpec(scope="friendly").to_dict(),
        )
        resolve_effect_ir(rep_ir, guardian, gs)

        # Deal damage to enemy — should NOT be intercepted
        damage_ir = deal_damage(3)
        logs = resolve_effect_ir(damage_ir, guardian, gs, [enemy.instance_id])
        assert enemy.damage == 3

    def test_replacement_removed_when_source_leaves(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Guardian")
        rep_ir = replacement("deal_damage")
        resolve_effect_ir(rep_ir, source, gs)
        assert len(gs.active_replacements) == 1

        gs.unregister_card(source.instance_id)
        assert len(gs.active_replacements) == 0


# ---------------------------------------------------------------------------
# Player Choice tests
# ---------------------------------------------------------------------------


class TestPlayerChoiceNode:
    def test_choice_halts_with_pending(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Chooser")

        ir = player_choice(
            prompt="Choose a target",
            options=[
                {"label": "Deal 2 damage", "effect": deal_damage(2)},
                {"label": "Heal all", "effect": heal(amount="all")},
            ],
        )
        logs = resolve_effect_ir(ir, source, gs)

        assert gs.pending_choice is not None
        assert gs.pending_choice["prompt"] == "Choose a target"
        assert len(gs.pending_choice["options"]) == 2
        assert "CHOICE_PENDING" in logs[0]

    def test_choice_auto_play_when_already_pending(self):
        """If a choice is already pending, auto-play picks first option."""
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Chooser")
        enemy = add_unit(gs, "p2", ZoneType.BASE, might=5, name="Enemy")

        # Simulate an already-pending choice
        gs.pending_choice = {"prompt": "existing"}

        ir = player_choice(
            prompt="Another choice",
            options=[
                {"label": "Deal 2 damage", "effect": deal_damage(2)},
            ],
        )
        logs = resolve_effect_ir(ir, source, gs, [enemy.instance_id])
        # Should auto-play first option since pending_choice was already set
        assert enemy.damage == 2

    def test_choice_target_selection(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Chooser")

        ir = player_choice(
            prompt="Choose a friendly unit",
            target=TargetSpec(scope="friendly").to_dict(),
            min_choices=1,
            max_choices=1,
        )
        logs = resolve_effect_ir(ir, source, gs)
        assert gs.pending_choice is not None
        assert gs.pending_choice["target"] is not None


# ---------------------------------------------------------------------------
# Choice State Machine: halt → message → resume
# ---------------------------------------------------------------------------


class TestChoiceStateMachine:
    def test_submit_choice_resumes_with_option(self):
        """Full round trip: halt on choice → submit option → effect resolves."""
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Chooser")
        enemy = add_unit(gs, "p2", ZoneType.BASE, might=5, name="Enemy")

        # Step 1: Resolver halts on player_choice
        ir = player_choice(
            prompt="Pick your action",
            options=[
                {"label": "Deal 2 damage", "effect": deal_damage(2)},
                {"label": "Heal self", "effect": heal(amount="all")},
            ],
        )
        logs = resolve_effect_ir(ir, source, gs, [enemy.instance_id])
        assert gs.pending_choice is not None
        assert "CHOICE_PENDING" in logs[0]

        # Step 2: Build ChoiceRequired server message
        choice_msg = ChoiceRequired(
            prompt=gs.pending_choice["prompt"],
            options=gs.pending_choice["options"],
            controller_id=gs.pending_choice["controller_id"],
        )
        assert choice_msg.type == "CHOICE_REQUIRED"
        assert choice_msg.prompt == "Pick your action"

        # Step 3: Client submits choice (option 0 = Deal 2 damage)
        submit_logs = execute_action(gs, "p1", ActionType.SUBMIT_CHOICE, {
            "chosen_option_index": 0,
        })
        assert gs.pending_choice is None
        assert enemy.damage == 2
        assert any("option 0" in l for l in submit_logs)

    def test_submit_choice_wrong_player_rejected(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Chooser")
        ir = player_choice(prompt="Choose", options=[
            {"label": "A", "effect": deal_damage(1)},
        ])
        resolve_effect_ir(ir, source, gs)

        # Player 2 tries to submit — should be rejected
        logs = execute_action(gs, "p2", ActionType.SUBMIT_CHOICE, {
            "chosen_option_index": 0,
        })
        assert gs.pending_choice is not None  # still pending
        assert any("Not your choice" in l for l in logs)

    def test_submit_choice_no_pending(self):
        gs = make_game()
        logs = execute_action(gs, "p1", ActionType.SUBMIT_CHOICE, {
            "chosen_option_index": 0,
        })
        assert any("No pending choice" in l for l in logs)

    def test_submit_choice_second_option(self):
        gs = make_game()
        source = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Chooser")
        source.damage = 2  # has damage to heal

        ir = player_choice(
            prompt="Pick",
            options=[
                {"label": "Deal 3 damage", "effect": deal_damage(3)},
                {"label": "Heal self", "effect": heal(amount="all")},
            ],
        )
        resolve_effect_ir(ir, source, gs, [source.instance_id])
        assert gs.pending_choice is not None

        # Pick option 1 (heal)
        execute_action(gs, "p1", ActionType.SUBMIT_CHOICE, {
            "chosen_option_index": 1,
        })
        assert gs.pending_choice is None
        assert source.damage == 0  # fully healed

    def test_choice_required_message_serializes(self):
        msg = ChoiceRequired(
            prompt="Choose a unit to destroy",
            options=[{"label": "Destroy", "effect": {"type": "kill"}}],
            target={"obj_type": "unit", "scope": "enemy"},
            min_choices=1,
            max_choices=1,
            controller_id="p1",
        )
        d = msg.model_dump()
        assert d["type"] == "CHOICE_REQUIRED"
        assert d["prompt"] == "Choose a unit to destroy"
        assert len(d["options"]) == 1

    def test_submit_choice_message_parses(self):
        msg = parse_client_message({
            "type": "SUBMIT_CHOICE",
            "chosen_option_index": 1,
            "chosen_target_ids": [],
        })
        assert isinstance(msg, SubmitChoice)
        assert msg.chosen_option_index == 1
