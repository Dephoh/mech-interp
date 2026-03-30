"""Proof scenarios for SCORE_POINTS effects.

Each test proves that score_points effects produce correct frontend-visible
results through the full validate -> execute -> cleanup pipeline.
"""

from __future__ import annotations

import pytest
from helpers import Scenario

from app.engine.effect_ir import score_points
from app.engine.enums import ActionType, Phase


class TestScorePointsProof:
    """Proof: score_points effects update scores and trigger victory."""

    def test_score_points_increases_score(self):
        """Proof: a spell with score_points(2) increases the caster's score
        by 2, visible in both game state and frontend view."""
        s = Scenario()
        spell = s.add_hand_spell(
            "p1", name="Score", cost_energy=1,
            effect_ir=score_points(2),
        )
        s.give_resources("p1", energy=5)

        # Play the spell
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        # Resolve: both players pass priority
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: score increased
        assert s.gs.players["p1"].score == 2

        # Frontend view: score visible to the scoring player
        view = s.view("p1")
        assert view["you"]["score"] == 2

    def test_score_visible_to_opponent(self):
        """Proof: after scoring 2 points, the opponent's frontend view
        shows the correct score."""
        s = Scenario()
        spell = s.add_hand_spell(
            "p1", name="Score", cost_energy=1,
            effect_ir=score_points(2),
        )
        s.give_resources("p1", energy=5)

        # Play and resolve
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Frontend view: opponent sees P1's score
        opp_view = s.view("p2")
        assert opp_view["opponent"]["score"] == 2

    def test_game_over_on_victory_score(self):
        """Proof: scoring to >= victory_score (default 8) triggers game over
        during cleanup, visible in the frontend view."""
        s = Scenario()
        # Set P1 one point below victory threshold
        s.gs.players["p1"].score = 7

        spell = s.add_hand_spell(
            "p1", name="Score", cost_energy=1,
            effect_ir=score_points(1),
        )
        s.give_resources("p1", energy=5)

        # Play and resolve
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: score reached victory threshold
        assert s.gs.players["p1"].score == 8

        # Game state: game over triggered by cleanup
        assert s.gs.game_over is True
        assert s.gs.winner_id == "p1"

        # Frontend view: game_over and winner visible
        view = s.view("p1")
        assert view["game_over"] is True
        assert view["winner_id"] == "p1"
