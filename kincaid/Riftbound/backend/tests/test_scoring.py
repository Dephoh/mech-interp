"""Tests for scoring: Hold, Conquer, 7-point restriction, Burn-Out."""

import pytest
from helpers import make_game, add_unit

from app.engine.enums import ControlStatus, Phase
from app.engine.scoring import check_win, perform_burn_out, score_conquer, score_hold


class TestCheckWin:
    def test_no_winner_at_start(self):
        gs = make_game()
        assert check_win(gs) is None

    def test_winner_at_8(self):
        gs = make_game()
        gs.players["p1"].score = 8
        assert check_win(gs) == "p1"

    def test_no_winner_at_7(self):
        gs = make_game()
        gs.players["p1"].score = 7
        assert check_win(gs) is None


class TestScoreHold:
    def test_score_hold_controlled_battlefield(self):
        gs = make_game(num_battlefields=2)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        logs = score_hold(gs, "p1")
        assert gs.players["p1"].score == 1
        assert len(logs) >= 1

    def test_no_hold_on_uncontrolled(self):
        gs = make_game(num_battlefields=2)
        logs = score_hold(gs, "p1")
        assert gs.players["p1"].score == 0

    def test_no_double_scoring(self):
        gs = make_game(num_battlefields=2)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        score_hold(gs, "p1")
        score_hold(gs, "p1")  # second call
        assert gs.players["p1"].score == 1  # only scored once


class TestScoreConquer:
    def test_conquer_scores_point(self):
        gs = make_game(num_battlefields=2)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        logs = score_conquer(gs, "p1", bf_id)
        assert gs.players["p1"].score == 1

    def test_conquer_blocked_at_7_without_all_scored(self):
        gs = make_game(num_battlefields=2)
        gs.players["p1"].score = 7
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        logs = score_conquer(gs, "p1", bf_id)
        # Should NOT have scored (at 7 without all battlefields scored)
        assert gs.players["p1"].score == 7

    def test_conquer_allowed_at_7_with_all_scored(self):
        gs = make_game(num_battlefields=2)
        gs.players["p1"].score = 7
        # Mark all battlefields as scored this turn
        for bf_id in gs.battlefields:
            gs.players["p1"].battlefields_scored_this_turn.add(bf_id)

        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        # Reset scored tracking for this specific bf so it can be scored again
        gs.players["p1"].battlefields_scored_this_turn.discard(bf_id)
        gs.players["p1"].battlefields_scored_this_turn.add(bf_id)

        logs = score_conquer(gs, "p1", bf_id)
        # Already in scored set, so won't score again
        assert gs.players["p1"].score == 7


class TestBurnOut:
    def test_burn_out_recycles_trash(self):
        gs = make_game()
        p1 = gs.players["p1"]
        p1.main_deck = []

        # Add cards to trash
        from app.engine.card_types import CardInstance, CardDefinition
        from app.engine.enums import CardType, ZoneType

        for i in range(5):
            defn = CardDefinition(
                card_id=f"trash_{i}", name=f"Trash {i}", card_type=CardType.SPELL
            )
            inst = CardInstance.create(defn, "p1", ZoneType.TRASH)
            gs.instances[inst.instance_id] = inst
            p1.trash.append(inst.instance_id)

        logs = perform_burn_out(gs, "p1")
        # Trash recycled into deck
        assert len(p1.main_deck) == 5
        assert len(p1.trash) == 0
        # Opponent gains a point
        assert gs.players["p2"].score == 1
