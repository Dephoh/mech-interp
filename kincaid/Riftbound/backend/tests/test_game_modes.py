"""Tests for multi-player game modes: FFA3, FFA4, 2v2."""

import pytest
from helpers import make_game, add_unit

from app.engine.enums import ControlStatus, GameMode, Phase
from app.engine.game_state import GameState, PlayerState, _mode_config
from app.engine.scoring import check_win, perform_burn_out, score_conquer, score_hold


# ---------------------------------------------------------------------------
# GameMode enum & _mode_config
# ---------------------------------------------------------------------------


class TestGameModeEnum:
    def test_all_modes_exist(self):
        assert GameMode.STANDARD_1V1.value == "standard_1v1"
        assert GameMode.FFA3.value == "ffa3"
        assert GameMode.FFA4.value == "ffa4"
        assert GameMode.TWO_VS_TWO.value == "2v2"

    def test_mode_config_player_counts(self):
        assert _mode_config(GameMode.STANDARD_1V1)["players"] == 2
        assert _mode_config(GameMode.FFA3)["players"] == 3
        assert _mode_config(GameMode.FFA4)["players"] == 4
        assert _mode_config(GameMode.TWO_VS_TWO)["players"] == 4

    def test_mode_config_victory_scores(self):
        assert _mode_config(GameMode.STANDARD_1V1)["victory"] == 8
        assert _mode_config(GameMode.FFA3)["victory"] == 8
        assert _mode_config(GameMode.FFA4)["victory"] == 8
        assert _mode_config(GameMode.TWO_VS_TWO)["victory"] == 11

    def test_mode_config_battlefield_counts(self):
        assert _mode_config(GameMode.STANDARD_1V1)["battlefields"] == 2
        assert _mode_config(GameMode.FFA3)["battlefields"] == 3
        assert _mode_config(GameMode.FFA4)["battlefields"] == 3
        assert _mode_config(GameMode.TWO_VS_TWO)["battlefields"] == 3


# ---------------------------------------------------------------------------
# Game creation with make_game helper
# ---------------------------------------------------------------------------


class TestMultiPlayerCreation:
    def test_ffa3_creates_3_players(self):
        gs = make_game(game_mode=GameMode.FFA3, num_battlefields=3)
        assert len(gs.players) == 3
        assert len(gs.player_order) == 3
        assert gs.victory_score == 8

    def test_ffa4_creates_4_players(self):
        gs = make_game(game_mode=GameMode.FFA4, num_battlefields=3)
        assert len(gs.players) == 4
        assert len(gs.player_order) == 4
        assert gs.victory_score == 8

    def test_2v2_creates_4_players_with_teams(self):
        gs = make_game(game_mode=GameMode.TWO_VS_TWO, num_battlefields=3)
        assert len(gs.players) == 4
        assert len(gs.player_order) == 4
        assert gs.victory_score == 11
        # Teams assigned
        assert len(gs.teams) == 4
        team_values = set(gs.teams.values())
        assert team_values == {0, 1}

    def test_2v2_default_team_assignment(self):
        gs = make_game(game_mode=GameMode.TWO_VS_TWO, num_battlefields=3)
        # Default interleaving: p1=team0, p2=team1, p3=team0, p4=team1
        assert gs.teams["p1"] == 0
        assert gs.teams["p2"] == 1
        assert gs.teams["p3"] == 0
        assert gs.teams["p4"] == 1

    def test_standard_1v1_backward_compatible(self):
        gs = make_game()
        assert len(gs.players) == 2
        assert gs.game_mode == GameMode.STANDARD_1V1
        assert gs.victory_score == 8
        assert len(gs.teams) == 0


# ---------------------------------------------------------------------------
# opponent_ids / teammate_id / next_player
# ---------------------------------------------------------------------------


class TestMultiPlayerNavigation:
    def test_opponent_ids_ffa3(self):
        gs = make_game(game_mode=GameMode.FFA3, num_battlefields=3)
        opps = gs.opponent_ids("p1")
        assert set(opps) == {"p2", "p3"}

    def test_opponent_ids_ffa4(self):
        gs = make_game(game_mode=GameMode.FFA4, num_battlefields=3)
        opps = gs.opponent_ids("p1")
        assert set(opps) == {"p2", "p3", "p4"}

    def test_opponent_ids_2v2_excludes_teammate(self):
        gs = make_game(game_mode=GameMode.TWO_VS_TWO, num_battlefields=3)
        # p1 is team 0, p3 is team 0 => they are teammates
        opps = gs.opponent_ids("p1")
        assert "p3" not in opps  # teammate excluded
        assert set(opps) == {"p2", "p4"}

    def test_teammate_id_2v2(self):
        gs = make_game(game_mode=GameMode.TWO_VS_TWO, num_battlefields=3)
        assert gs.teammate_id("p1") == "p3"
        assert gs.teammate_id("p2") == "p4"
        assert gs.teammate_id("p3") == "p1"
        assert gs.teammate_id("p4") == "p2"

    def test_teammate_id_none_in_ffa(self):
        gs = make_game(game_mode=GameMode.FFA3, num_battlefields=3)
        assert gs.teammate_id("p1") is None

    def test_next_player_wraps(self):
        gs = make_game(game_mode=GameMode.FFA3, num_battlefields=3)
        assert gs.next_player("p1") == "p2"
        assert gs.next_player("p2") == "p3"
        assert gs.next_player("p3") == "p1"

    def test_opponent_id_still_works_for_2p(self):
        gs = make_game()
        assert gs.opponent_id("p1") == "p2"
        assert gs.opponent_id("p2") == "p1"


# ---------------------------------------------------------------------------
# Team scoring (2v2)
# ---------------------------------------------------------------------------


class TestTeamScoring:
    def test_team_score_sums_teammates(self):
        gs = make_game(game_mode=GameMode.TWO_VS_TWO, num_battlefields=3)
        gs.players["p1"].score = 5
        gs.players["p3"].score = 3  # p1's teammate
        assert gs.team_score("p1") == 8
        assert gs.team_score("p3") == 8

    def test_team_score_independent_teams(self):
        gs = make_game(game_mode=GameMode.TWO_VS_TWO, num_battlefields=3)
        gs.players["p1"].score = 5
        gs.players["p3"].score = 3
        gs.players["p2"].score = 2
        gs.players["p4"].score = 1
        assert gs.team_score("p1") == 8
        assert gs.team_score("p2") == 3

    def test_2v2_check_win_uses_team_score(self):
        gs = make_game(game_mode=GameMode.TWO_VS_TWO, num_battlefields=3)
        gs.players["p1"].score = 6
        gs.players["p3"].score = 5  # combined = 11
        winner = check_win(gs)
        assert winner in ("p1", "p3")

    def test_2v2_no_win_below_11(self):
        gs = make_game(game_mode=GameMode.TWO_VS_TWO, num_battlefields=3)
        gs.players["p1"].score = 6
        gs.players["p3"].score = 4  # combined = 10
        assert check_win(gs) is None

    def test_team_score_returns_individual_for_ffa(self):
        gs = make_game(game_mode=GameMode.FFA3, num_battlefields=3)
        gs.players["p1"].score = 5
        assert gs.team_score("p1") == 5


# ---------------------------------------------------------------------------
# Scoring: Hold / Conquer in multi-player
# ---------------------------------------------------------------------------


class TestMultiPlayerScoreHold:
    def test_ffa3_score_hold_multiple_battlefields(self):
        gs = make_game(game_mode=GameMode.FFA3, num_battlefields=3)
        bf_ids = list(gs.battlefields.keys())

        # p1 controls two battlefields
        for bf_id in bf_ids[:2]:
            bf = gs.battlefields[bf_id]
            bf.control_status = ControlStatus.CONTROLLED
            bf.controller_id = "p1"

        logs = score_hold(gs, "p1")
        assert gs.players["p1"].score == 2


class TestMultiPlayerScoreConquer:
    def test_ffa3_conquer_scores_point(self):
        gs = make_game(game_mode=GameMode.FFA3, num_battlefields=3)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p2"

        logs = score_conquer(gs, "p2", bf_id)
        assert gs.players["p2"].score == 1


# ---------------------------------------------------------------------------
# Burn Out in multi-player
# ---------------------------------------------------------------------------


class TestMultiPlayerBurnOut:
    def test_burn_out_gives_all_opponents_a_point(self):
        gs = make_game(game_mode=GameMode.FFA3, num_battlefields=3)
        p1 = gs.players["p1"]
        p1.main_deck = []

        from app.engine.card_types import CardDefinition, CardInstance
        from app.engine.enums import CardType, ZoneType

        for i in range(3):
            defn = CardDefinition(
                card_id=f"trash_{i}", name=f"Trash {i}", card_type=CardType.SPELL
            )
            inst = CardInstance.create(defn, "p1", ZoneType.TRASH)
            gs.instances[inst.instance_id] = inst
            p1.trash.append(inst.instance_id)

        logs = perform_burn_out(gs, "p1")
        # Both opponents gain 1 point each
        assert gs.players["p2"].score == 1
        assert gs.players["p3"].score == 1
        # p1 does not gain a point
        assert gs.players["p1"].score == 0

    def test_burn_out_2v2_only_opponents_gain(self):
        gs = make_game(game_mode=GameMode.TWO_VS_TWO, num_battlefields=3)
        p1 = gs.players["p1"]
        p1.main_deck = []

        from app.engine.card_types import CardDefinition, CardInstance
        from app.engine.enums import CardType, ZoneType

        for i in range(3):
            defn = CardDefinition(
                card_id=f"trash_{i}", name=f"Trash {i}", card_type=CardType.SPELL
            )
            inst = CardInstance.create(defn, "p1", ZoneType.TRASH)
            gs.instances[inst.instance_id] = inst
            p1.trash.append(inst.instance_id)

        logs = perform_burn_out(gs, "p1")
        # Only opponents (team 1) gain points, not teammate p3
        assert gs.players["p2"].score == 1
        assert gs.players["p4"].score == 1
        assert gs.players["p3"].score == 0  # teammate
        assert gs.players["p1"].score == 0


# ---------------------------------------------------------------------------
# check_win in FFA
# ---------------------------------------------------------------------------


class TestFFACheckWin:
    def test_ffa3_first_to_8_wins(self):
        gs = make_game(game_mode=GameMode.FFA3, num_battlefields=3)
        gs.players["p2"].score = 8
        assert check_win(gs) == "p2"

    def test_ffa4_no_winner_below_8(self):
        gs = make_game(game_mode=GameMode.FFA4, num_battlefields=3)
        gs.players["p1"].score = 7
        gs.players["p2"].score = 7
        gs.players["p3"].score = 7
        gs.players["p4"].score = 7
        assert check_win(gs) is None
