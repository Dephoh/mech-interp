"""Proof scenarios for CONCEDE and GAME OVER conditions.

Tests that conceding ends the game immediately with the opponent as winner,
that conceding works cleanly during combat, and that no actions are accepted
after the game is over.
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.enums import (
    ActionType,
    Phase,
    ZoneType,
)


# =========================================================================
# Concede and Game Over
# =========================================================================


class TestConcedeProof:
    """Proof: conceding ends the game and produces correct frontend state."""

    def test_concede_sets_game_over(self):
        """Proof: when P1 concedes, game_over=True and winner_id='p2'
        in both the game state and BOTH players' frontend views."""
        s = Scenario()

        view_p1 = s.act("p1", ActionType.CONCEDE)

        # Game state
        assert s.gs.game_over is True
        assert s.gs.winner_id == "p2"

        # Frontend view for P1 (the conceding player)
        assert view_p1["game_over"] is True
        assert view_p1["winner_id"] == "p2"

        # Frontend view for P2 (the winner)
        view_p2 = s.view("p2")
        assert view_p2["game_over"] is True
        assert view_p2["winner_id"] == "p2"

    def test_concede_during_combat(self):
        """Proof: conceding during an active showdown/combat ends the game
        cleanly regardless of phase. The game_over flag is set even though
        combat is in progress."""
        s = Scenario()

        # Set up a contested battlefield with units from both players
        bf = s.gs.battlefields[s.bf0]
        add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=s.bf0, might=3, name="Attacker")
        add_unit(s.gs, "p2", ZoneType.BATTLEFIELD, bf_id=s.bf0, might=4, name="Defender")
        bf.contested_by = "p1"

        # Open a showdown to simulate mid-combat state
        from app.engine.combat import open_showdown
        open_showdown(s.gs, s.bf0, "p1")

        # Verify we are in showdown phase before conceding
        assert s.gs.active_showdown is not None

        # P1 concedes mid-combat
        view_p1 = s.act("p1", ActionType.CONCEDE)

        # Game state: game over despite active showdown
        assert s.gs.game_over is True
        assert s.gs.winner_id == "p2"

        # Frontend views reflect game over
        assert view_p1["game_over"] is True
        assert view_p1["winner_id"] == "p2"

        view_p2 = s.view("p2")
        assert view_p2["game_over"] is True
        assert view_p2["winner_id"] == "p2"

    def test_actions_rejected_after_game_over(self):
        """Proof: after the game is over (via concede), all subsequent
        actions are rejected. The validator returns ok=False for any
        action when game_over is True, which causes Scenario.act to
        raise AssertionError."""
        s = Scenario()
        s.add_deck_cards("p1", 5)
        s.add_deck_cards("p2", 5)

        # Give P2 a card in hand so they could try to play it
        unit = s.add_hand_unit("p2", might=3, cost_energy=2, name="Too Late")
        s.give_resources("p2", energy=5)

        # P1 concedes
        s.act("p1", ActionType.CONCEDE)
        assert s.gs.game_over is True

        # P2 tries to play a card -- rejected
        with pytest.raises(AssertionError, match="rejected"):
            s.act("p2", ActionType.PLAY_CARD, {
                "instance_id": unit.instance_id,
                "targets": [],
                "destination": {"zone": "base"},
            })

        # P2 tries to advance phase -- rejected
        with pytest.raises(AssertionError, match="rejected"):
            s.act("p2", ActionType.ADVANCE_PHASE)

        # Even P1 (the loser) cannot take actions
        with pytest.raises(AssertionError, match="rejected"):
            s.act("p1", ActionType.ADVANCE_PHASE)
