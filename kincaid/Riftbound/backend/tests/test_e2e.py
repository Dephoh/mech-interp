"""End-to-end 2-player game tests via direct engine calls.

Simulates a full game lifecycle: mulligan, awaken/channel/draw, action phase
(play cards, move units), combat, scoring, and game-over detection.

Uses the helpers infrastructure from existing tests and drives everything
through validate -> execute -> cleanup (same pipeline as the real server,
minus WebSocket).
"""

from __future__ import annotations

import pytest
from helpers import (
    Scenario,
    add_unit,
    make_game,
    make_unit_def,
)

from app.engine.action_executor import execute_action
from app.engine.action_validator import validate_action
from app.engine.card_types import CardDefinition, CardInstance
from app.engine.cleanup import run_cleanup
from app.engine.combat import start_combat
from app.engine.enums import (
    ActionType,
    CardType,
    ControlStatus,
    Domain,
    Phase,
    ZoneType,
)
from app.engine.game_state import (
    BattlefieldState,
    GameState,
    PlayerState,
    create_game,
    _draw_cards,
)
from app.engine.scoring import check_win, score_hold
from app.engine.state_machine import advance_phase, start_game


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_card_db() -> dict[str, CardDefinition]:
    """Build a minimal card database with known cards for e2e testing."""
    db: dict[str, CardDefinition] = {}

    # Legend
    db["legend_alpha"] = CardDefinition(
        card_id="legend_alpha",
        name="Alpha Legend",
        card_type=CardType.LEGEND,
    )
    db["legend_beta"] = CardDefinition(
        card_id="legend_beta",
        name="Beta Legend",
        card_type=CardType.LEGEND,
    )

    # Champions
    db["champion_a"] = CardDefinition(
        card_id="champion_a",
        name="Champion A",
        card_type=CardType.UNIT,
        base_might=4,
        cost_energy=3,
    )
    db["champion_b"] = CardDefinition(
        card_id="champion_b",
        name="Champion B",
        card_type=CardType.UNIT,
        base_might=4,
        cost_energy=3,
    )

    # Main deck units (need 40 per player)
    for i in range(20):
        db[f"unit_{i}"] = CardDefinition(
            card_id=f"unit_{i}",
            name=f"Unit {i}",
            card_type=CardType.UNIT,
            base_might=2 + (i % 3),  # might cycles 2, 3, 4
            cost_energy=1 + (i % 2),
            domains=(Domain.FURY,),
        )

    # Main deck spells
    for i in range(20):
        db[f"spell_{i}"] = CardDefinition(
            card_id=f"spell_{i}",
            name=f"Spell {i}",
            card_type=CardType.SPELL,
            cost_energy=1,
            domains=(Domain.FURY,),
        )

    # Rune cards (need 12 per player)
    for i in range(12):
        db[f"rune_{i}"] = CardDefinition(
            card_id=f"rune_{i}",
            name=f"Rune {i}",
            card_type=CardType.RUNE,
            domains=(Domain.FURY,),
        )

    # Battlefields (need 3 per player)
    for i in range(6):
        db[f"bf_{i}"] = CardDefinition(
            card_id=f"bf_{i}",
            name=f"Battlefield {i}",
            card_type=CardType.BATTLEFIELD,
        )

    return db


def _make_player_config(
    player_id: str,
    display_name: str,
    legend_id: str,
    champion_id: str,
) -> dict:
    """Build a player config dict for create_game with deterministic deck lists."""
    # 40-card main deck: 20 units + 20 spells
    main_deck = [f"unit_{i}" for i in range(20)] + [f"spell_{i}" for i in range(20)]
    # 12-card rune deck
    rune_deck = [f"rune_{i}" for i in range(12)]
    # 3 battlefield candidates
    bf_offset = 0 if player_id == "p1" else 3
    battlefields = [f"bf_{bf_offset + i}" for i in range(3)]

    return {
        "player_id": player_id,
        "display_name": display_name,
        "legend_id": legend_id,
        "champion_id": champion_id,
        "main_deck": main_deck,
        "rune_deck": rune_deck,
        "battlefields": battlefields,
    }


def _create_e2e_game() -> GameState:
    """Create a full 2-player game via create_game with known cards.

    Returns a GameState in SETUP_MULLIGAN phase, ready for mulligan choices.
    """
    card_db = _make_card_db()
    configs = [
        _make_player_config("p1", "Player One", "legend_alpha", "champion_a"),
        _make_player_config("p2", "Player Two", "legend_beta", "champion_b"),
    ]
    gs = create_game("e2e_test", configs, card_db)
    # Fix turn order for deterministic tests
    gs.player_order = ["p1", "p2"]
    gs.turn_player_id = "p1"
    gs.active_player_id = "p1"
    gs.players["p1"].goes_second = False
    gs.players["p2"].goes_second = True
    return gs


def _do_action(gs: GameState, player_id: str, action_type: ActionType, payload: dict) -> list[str]:
    """Validate and execute an action, asserting validation passes."""
    result = validate_action(gs, player_id, action_type, payload)
    assert result.ok, (
        f"Action {action_type.value} by {player_id} rejected: {result.error} "
        f"(phase={gs.phase.value}, active={gs.active_player_id})"
    )
    return execute_action(gs, player_id, action_type, payload)


def _add_deck_cards(gs: GameState, player_id: str, count: int = 10) -> None:
    """Add filler cards to a player's main deck to avoid burn-out."""
    ps = gs.players[player_id]
    for i in range(count):
        defn = make_unit_def(name=f"Filler {i}")
        inst = CardInstance.create(defn, player_id, ZoneType.MAIN_DECK)
        gs.instances[inst.instance_id] = inst
        ps.main_deck.append(inst.instance_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestE2EFullGameFlow:
    """End-to-end test: complete 2-player game flow through all phases."""

    def test_mulligan_to_action_phase(self):
        """Both players complete mulligan, which transitions the game through
        awaken -> beginning -> channel -> draw -> action phase."""
        gs = _create_e2e_game()

        # Verify initial state
        assert gs.phase == Phase.SETUP_MULLIGAN
        assert len(gs.players["p1"].hand) == 4
        assert len(gs.players["p2"].hand) == 4

        # P1 keeps all cards (keep_indices = [0,1,2,3])
        _do_action(gs, "p1", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})
        assert gs.mulligan_done["p1"] is True
        # Game does not start yet - P2 hasn't mulliganed
        assert gs.phase == Phase.SETUP_MULLIGAN

        # P2 mulligans 2 cards (keep indices [0, 1] keeps first 2)
        _do_action(gs, "p2", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1]})
        assert gs.mulligan_done["p2"] is True

        # After both mulligans: game should auto-start and reach Action phase
        assert gs.phase == Phase.ACTION
        assert gs.turn_number == 1
        assert gs.turn_player_id == "p1"
        assert gs.active_player_id == "p1"

        # P1 drew 1 card during draw phase (started with 4, kept all)
        assert len(gs.players["p1"].hand) == 5

        # P2 mulliganed 2, drew 2 replacements during mulligan.
        # P2 hasn't had their turn yet so no draw phase card.
        assert len(gs.players["p2"].hand) == 4

        # P1 goes first: channels 2 runes
        assert len(gs.base_runes["p1"]) == 2
        # P2 hasn't had their turn yet
        assert len(gs.base_runes["p2"]) == 0

    def test_play_unit_and_move_to_battlefield(self):
        """Play a unit from hand to base, then move it to a battlefield."""
        gs = _create_e2e_game()

        # Fast-forward through mulligan
        _do_action(gs, "p1", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})
        _do_action(gs, "p2", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})

        assert gs.phase == Phase.ACTION
        p1 = gs.players["p1"]

        # Give P1 energy to play a card
        p1.rune_pool.energy = 10

        # Find a unit in P1's hand
        unit_id = None
        for iid in p1.hand:
            inst = gs.instances[iid]
            if inst.card_type == CardType.UNIT:
                unit_id = iid
                break
        assert unit_id is not None, "P1 should have at least one unit in hand"
        unit = gs.instances[unit_id]
        unit_cost = unit.definition.cost_energy

        # Play the unit to base (default destination)
        hand_size_before = len(p1.hand)
        _do_action(gs, "p1", ActionType.PLAY_CARD, {
            "instance_id": unit_id,
            "targets": [],
            "destination": {"zone": "base"},
        })

        # Unit should be in P1's base
        assert unit.zone == ZoneType.BASE
        assert unit_id in gs.base_units["p1"]
        assert len(p1.hand) == hand_size_before - 1
        assert p1.rune_pool.energy == 10 - unit_cost

        # The unit enters exhausted - manually ready it for movement test.
        unit.exhausted = False

        # Move the unit to a battlefield
        bf_id = list(gs.battlefields.keys())[0]
        _do_action(gs, "p1", ActionType.MOVE_UNIT, {
            "instance_id": unit_id,
            "destination": {"zone": "battlefield", "id": bf_id},
        })

        # Unit should now be at the battlefield
        assert unit.zone == ZoneType.BATTLEFIELD
        assert unit_id in gs.battlefields[bf_id].units
        assert unit_id not in gs.base_units["p1"]
        # Unit is exhausted after moving
        assert unit.exhausted is True

    def test_end_turn_and_next_player(self):
        """End turn transitions to the next player's action phase."""
        gs = _create_e2e_game()

        # Fast-forward through mulligan
        _do_action(gs, "p1", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})
        _do_action(gs, "p2", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})
        assert gs.phase == Phase.ACTION
        assert gs.turn_player_id == "p1"
        assert gs.turn_number == 1

        # End P1's turn (advance phase from ACTION -> end of turn -> next turn)
        _do_action(gs, "p1", ActionType.ADVANCE_PHASE, {})

        # Should now be P2's turn
        assert gs.turn_player_id == "p2"
        assert gs.active_player_id == "p2"
        assert gs.turn_number == 2
        assert gs.phase == Phase.ACTION

        # P2 goes second on first turn so should have channeled 3 runes
        assert len(gs.base_runes["p2"]) == 3

    def test_full_turn_cycle(self):
        """Both players take a turn, verifying turn counting and phase transitions."""
        gs = _create_e2e_game()

        # Mulligan
        _do_action(gs, "p1", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})
        _do_action(gs, "p2", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})

        assert gs.turn_number == 1
        assert gs.turn_player_id == "p1"

        # P1 ends turn
        _do_action(gs, "p1", ActionType.ADVANCE_PHASE, {})
        assert gs.turn_number == 2
        assert gs.turn_player_id == "p2"

        # P2 ends turn
        _do_action(gs, "p2", ActionType.ADVANCE_PHASE, {})
        assert gs.turn_number == 3
        assert gs.turn_player_id == "p1"

        # Back to P1 - verify rune channels accumulate
        # P1 channeled 2 on turn 1, another 2 on turn 3
        assert len(gs.base_runes["p1"]) == 4


class TestE2EScoring:
    """End-to-end tests: score changes via battlefield control."""

    def test_score_from_holding_battlefield(self):
        """Controlling a battlefield at the start of your turn scores 1 point (Hold)."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # P1 controls the battlefield with a unit there
        u1 = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Holder")
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        # Give both players deck cards to avoid burn-out during draw phase
        _add_deck_cards(gs, "p1", 10)
        _add_deck_cards(gs, "p2", 10)

        assert gs.players["p1"].score == 0

        # End P1's turn: advance_phase from ACTION transitions to next turn
        _do_action(gs, "p1", ActionType.ADVANCE_PHASE, {})

        # Now it's P2's turn. P2 ends turn.
        _do_action(gs, "p2", ActionType.ADVANCE_PHASE, {})

        # Now it's P1's turn 3. Beginning phase auto-executed Hold scoring.
        assert gs.players["p1"].score == 1

    def test_score_from_conquering_battlefield(self):
        """Winning combat at a battlefield scores 1 point (Conquer)."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        # P1 has a strong unit, P2 has a weak unit at same battlefield
        u1 = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=5, name="Attacker")
        u2 = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="Defender")

        assert gs.players["p1"].score == 0

        # Start combat
        start_combat(gs, bf_id)

        # Both pass focus to close the combat showdown
        _do_action(gs, "p1", ActionType.PASS_FOCUS, {})
        _do_action(gs, "p2", ActionType.PASS_FOCUS, {})

        # P1 should have conquered the battlefield and scored 1 point
        assert gs.players["p1"].score == 1
        assert bf.control_status == ControlStatus.CONTROLLED
        assert bf.controller_id == "p1"

    def test_multiple_battlefield_scoring(self):
        """Controlling multiple battlefields scores 1 point per battlefield from Hold."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        bf_ids = list(gs.battlefields.keys())
        bf0 = gs.battlefields[bf_ids[0]]
        bf1 = gs.battlefields[bf_ids[1]]

        # P1 controls both battlefields
        u1 = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_ids[0], might=3, name="Holder A")
        u2 = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_ids[1], might=3, name="Holder B")
        bf0.control_status = ControlStatus.CONTROLLED
        bf0.controller_id = "p1"
        bf1.control_status = ControlStatus.CONTROLLED
        bf1.controller_id = "p1"

        _add_deck_cards(gs, "p1", 10)
        _add_deck_cards(gs, "p2", 10)

        assert gs.players["p1"].score == 0

        # P1 ends turn
        _do_action(gs, "p1", ActionType.ADVANCE_PHASE, {})
        # P2 ends turn -> P1's next turn -> beginning phase -> Hold scoring
        _do_action(gs, "p2", ActionType.ADVANCE_PHASE, {})

        # P1 should have scored 2 points (1 per controlled battlefield)
        assert gs.players["p1"].score == 2


class TestE2ECombatFlow:
    """End-to-end tests: full combat triggered via game flow."""

    def test_move_triggers_combat(self):
        """Moving a unit to an opponent's battlefield triggers combat via cleanup."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # P2 controls the battlefield
        u2 = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="Sentry")
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p2"

        # P1 has a unit in base, ready to move
        u1 = add_unit(gs, "p1", ZoneType.BASE, might=5, name="Raider")
        u1.exhausted = False

        # Move P1's unit to the battlefield (will contest it)
        _do_action(gs, "p1", ActionType.MOVE_UNIT, {
            "instance_id": u1.instance_id,
            "destination": {"zone": "battlefield", "id": bf_id},
        })

        # Cleanup should have staged and opened combat
        assert gs.active_showdown is not None or gs.active_combat is not None

        # If a showdown is open, pass focus to resolve it
        if gs.active_showdown:
            focus = gs.active_showdown.focus_player_id
            other = "p2" if focus == "p1" else "p1"
            _do_action(gs, focus, ActionType.PASS_FOCUS, {})
            _do_action(gs, other, ActionType.PASS_FOCUS, {})

        # After combat resolves: P1's stronger unit wins
        assert gs.active_combat is None
        assert u2.zone == ZoneType.TRASH
        assert u1.zone == ZoneType.BATTLEFIELD
        assert bf.control_status == ControlStatus.CONTROLLED
        assert bf.controller_id == "p1"

    def test_combat_defender_wins(self):
        """When the defender's unit is stronger, attackers are recalled and
        the defender keeps control."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p2"

        # P1 weak attacker (might=2), P2 strong defender (might=5)
        u1 = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="Weak Raider")
        u2 = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=5, name="Strong Guard")

        start_combat(gs, bf_id)
        _do_action(gs, "p1", ActionType.PASS_FOCUS, {})
        _do_action(gs, "p2", ActionType.PASS_FOCUS, {})

        # P1's unit destroyed, P2's survives
        assert u1.zone == ZoneType.TRASH
        assert u2.zone == ZoneType.BATTLEFIELD
        assert u2.damage == 0  # healed after combat
        assert bf.control_status == ControlStatus.CONTROLLED
        assert bf.controller_id == "p2"


class TestE2EGameOver:
    """End-to-end tests: game-over conditions."""

    def test_game_over_at_victory_score(self):
        """Score reaching victory_score (default 8) ends the game."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # Set P1 to 7 points (one away from victory)
        gs.players["p1"].score = 7

        # P1 controls a battlefield
        u1 = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Victor")
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        _add_deck_cards(gs, "p1", 10)
        _add_deck_cards(gs, "p2", 10)

        # End P1's turn -> P2's turn -> end P2's turn -> P1's next turn
        _do_action(gs, "p1", ActionType.ADVANCE_PHASE, {})
        # P1's score should reach 8 during their next beginning phase.
        # That beginning phase triggers when P2 ends their turn.
        if not gs.game_over:
            _do_action(gs, "p2", ActionType.ADVANCE_PHASE, {})

        # P1 should have hit 8 from Hold scoring
        assert gs.players["p1"].score >= 8
        assert gs.game_over is True
        assert gs.winner_id == "p1"

    def test_concede_ends_game(self):
        """A player conceding immediately ends the game with the opponent winning."""
        gs = _create_e2e_game()

        # Fast-forward through mulligan
        _do_action(gs, "p1", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})
        _do_action(gs, "p2", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})

        assert gs.phase == Phase.ACTION
        assert not gs.game_over

        # P2 concedes
        _do_action(gs, "p2", ActionType.CONCEDE, {})

        assert gs.game_over is True
        assert gs.winner_id == "p1"

    def test_conquer_for_game_winning_point(self):
        """Conquering a battlefield can win the game when at victory_score - 1."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        bf_ids = list(gs.battlefields.keys())
        bf0 = gs.battlefields[bf_ids[0]]
        bf1 = gs.battlefields[bf_ids[1]]

        # P1 at 7 points with the second battlefield already scored
        gs.players["p1"].score = 7
        gs.players["p1"].battlefields_scored_this_turn = {bf_ids[1]}

        # P1 conquers bf0 via combat
        bf0.contested_by = "p1"
        u1 = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_ids[0], might=5, name="Finisher")
        u2 = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_ids[0], might=2, name="Last Stand")

        start_combat(gs, bf_ids[0])
        _do_action(gs, "p1", ActionType.PASS_FOCUS, {})
        _do_action(gs, "p2", ActionType.PASS_FOCUS, {})

        # P1 conquered bf0; together with bf1 already scored, all battlefields
        # have been scored this turn -> final point allowed
        assert gs.players["p1"].score == 8
        assert gs.game_over is True
        assert gs.winner_id == "p1"


class TestE2EResourceManagement:
    """End-to-end tests: rune exhausting and resource flow."""

    def test_exhaust_rune_for_energy(self):
        """Exhausting a rune during action phase gives energy to spend."""
        gs = _create_e2e_game()

        # Mulligan
        _do_action(gs, "p1", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})
        _do_action(gs, "p2", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 1, 2, 3]})

        assert gs.phase == Phase.ACTION
        p1 = gs.players["p1"]

        # P1 should have channeled runes
        assert len(gs.base_runes["p1"]) > 0

        # Energy pool starts empty (emptied at end of draw phase per rule 315.4.d)
        assert p1.rune_pool.energy == 0

        # Exhaust first rune
        rune_id = gs.base_runes["p1"][0]
        rune = gs.instances[rune_id]
        assert not rune.exhausted

        _do_action(gs, "p1", ActionType.EXHAUST_RUNE, {"instance_id": rune_id})

        # Rune should be exhausted and player gained 1 energy
        assert rune.exhausted is True
        assert p1.rune_pool.energy == 1

    def test_play_card_costs_energy(self):
        """Playing a card from hand deducts its energy cost from the rune pool."""
        s = Scenario()
        s.give_resources("p1", energy=5)

        unit = s.add_hand_unit("p1", might=3, cost_energy=2, name="Soldier")
        assert s.gs.players["p1"].rune_pool.energy == 5

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": unit.instance_id,
            "targets": [],
            "destination": {"zone": "base"},
        })

        assert s.gs.players["p1"].rune_pool.energy == 3  # 5 - 2
        assert unit.zone == ZoneType.BASE


class TestE2EComprehensiveGame:
    """A comprehensive end-to-end test simulating multiple turns of gameplay."""

    def test_multi_turn_game_to_victory(self):
        """Simulate a multi-turn game where P1 builds board control and
        wins by reaching the victory score through repeated Hold scoring."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        bf_ids = list(gs.battlefields.keys())
        bf0 = gs.battlefields[bf_ids[0]]
        bf1 = gs.battlefields[bf_ids[1]]

        # P1 controls both battlefields
        u1 = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_ids[0], might=3, name="Guard A")
        u2 = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_ids[1], might=3, name="Guard B")
        bf0.control_status = ControlStatus.CONTROLLED
        bf0.controller_id = "p1"
        bf1.control_status = ControlStatus.CONTROLLED
        bf1.controller_id = "p1"

        # Give both players plenty of deck cards to avoid burn-out
        _add_deck_cards(gs, "p1", 30)
        _add_deck_cards(gs, "p2", 30)

        # Simulate turns: each P1 turn scores 2 points (Hold on 2 battlefields)
        # Need 4 full cycles (P1 turn + P2 turn) to reach 8 points
        turn = 0
        while not gs.game_over and turn < 20:
            turn += 1
            current = gs.turn_player_id
            _do_action(gs, current, ActionType.ADVANCE_PHASE, {})
            if gs.game_over:
                break

        # P1 should have won through Hold scoring
        # 2 points per P1 turn beginning phase, so after 4 of P1's turns: 8 points
        assert gs.game_over is True
        assert gs.winner_id == "p1"
        assert gs.players["p1"].score >= 8
