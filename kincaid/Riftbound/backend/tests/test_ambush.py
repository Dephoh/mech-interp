"""Tests for Ambush keyword (rule 755).

Ambush means a unit "may be played during either player's turn as though
it had Reaction."  Normal units can only be played during the controlling
player's turn; Ambush units can also be played during the opponent's turn
when you have priority (e.g., during chain resolution).

The unit still goes through normal play resolution (costs, chain, enters board).
"""

from __future__ import annotations

import pytest
from helpers import add_unit, load_card_db, make_game, make_unit_def

from app.engine.card_db import CardDB
from app.engine.card_types import CardInstance, KeywordInstance
from app.engine.action_executor import execute_action
from app.engine.action_validator import validate_action, ValidationResult
from app.engine.enums import (
    ActionType,
    CardType,
    ControlStatus,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.game_state import ChainItem, GameState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AMBUSH_CARD_ID = "unl-002-219"      # Inferna -- ambush+reaction+assault 2, 2E fury
NON_AMBUSH_CARD_ID = "ogn-097-298"  # Blastcone Fae -- hidden, 2E 1P(mind)
CHAKRAM_DANCER_ID = "unl-071-219"   # Chakram Dancer -- ambush, on_play: ready others here


@pytest.fixture(scope="module", autouse=True)
def _load_db():
    load_card_db()


def _add_card_to_hand(gs: GameState, player_id: str, card_id: str) -> CardInstance:
    """Create a card instance from the DB and put it in the player's hand."""
    defn = CardDB.get(card_id)
    inst = CardInstance.create(defn, player_id, ZoneType.HAND)
    gs.instances[inst.instance_id] = inst
    gs.players[player_id].hand.append(inst.instance_id)
    return inst


def _give_resources(gs: GameState, player_id: str, energy: int, power: dict[Domain, int] | None = None):
    """Give a player enough resources to pay a cost."""
    ps = gs.players[player_id]
    ps.rune_pool.energy = energy
    if power:
        for dom, amt in power.items():
            ps.rune_pool.power[dom] = amt


def _setup_opponent_turn_with_chain(gs: GameState):
    """Set up: P2's turn, P2 plays something on the chain, P1 has priority.

    This simulates a real gameplay scenario where P2 played a card/ability,
    opening the chain, and now P1 has priority to respond (Neutral Closed).
    """
    gs.turn_player_id = "p2"
    gs.phase = Phase.ACTION

    # P2 plays something that opens the chain
    dummy = ChainItem.create(controller_id="p2", source_instance_id=None)
    gs.chain.push(dummy)

    # After push, priority goes to the controller (p2). Then p2 passes priority
    # to p1. We simulate this by setting active_player_id directly.
    gs.active_player_id = "p1"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAmbush:
    """Rule 755: Ambush units can be played on the opponent's turn."""

    def test_ambush_unit_plays_on_opponent_turn_via_chain(self):
        """Full gameplay scenario: P2's turn, P2 plays something (chain opens),
        P1 has priority, P1 plays an ambush unit.

        Verify: validation passes, card enters the chain pipeline, and after
        resolution the unit is on the board.
        """
        gs = make_game()
        card = _add_card_to_hand(gs, "p1", AMBUSH_CARD_ID)

        # Inferna costs 2E, fury domain, no power cost
        _give_resources(gs, "p1", energy=5, power={Domain.FURY: 3})

        # Set up P2's turn with chain open, P1 has priority
        _setup_opponent_turn_with_chain(gs)

        # Validate: P1 should be able to play this ambush unit
        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": card.instance_id},
        )
        assert result.ok, f"Ambush unit should be playable on opponent's turn: {result.error}"

        # Execute the play
        logs = execute_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": card.instance_id},
        )

        # After execution, permanent finalizes immediately (rule 356.2).
        # The unit should be on the board (base by default).
        assert card.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD), (
            f"Ambush unit should be on the board after play, but is in {card.zone.value}"
        )
        assert card.controller_id == "p1"
        assert card.instance_id not in gs.players["p1"].hand, (
            "Card should have been removed from hand"
        )

    def test_ambush_unit_enters_board_exhausted(self):
        """Ambush does not grant readiness -- units still enter exhausted
        (unless they also have Accelerate). Verify the unit is exhausted
        after entering from an ambush play.
        """
        gs = make_game()
        card = _add_card_to_hand(gs, "p1", AMBUSH_CARD_ID)

        _give_resources(gs, "p1", energy=5, power={Domain.FURY: 3})
        _setup_opponent_turn_with_chain(gs)

        # Execute the play
        execute_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": card.instance_id},
        )

        # Unit should be on the board and exhausted
        assert card.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD)
        assert card.exhausted, (
            "Ambush unit should enter exhausted (Ambush does not grant readiness)"
        )

    def test_non_ambush_unit_cant_play_on_opponent_turn(self):
        """Same setup as test 1, but P1 tries to play a regular unit (no Ambush).
        Should be rejected by the validator.
        """
        gs = make_game()
        card = _add_card_to_hand(gs, "p1", NON_AMBUSH_CARD_ID)

        # Blastcone Fae costs 2E + 1P(mind)
        _give_resources(gs, "p1", energy=5, power={Domain.MIND: 3})

        # Set up P2's turn with chain open, P1 has priority
        _setup_opponent_turn_with_chain(gs)

        # Validate: should be rejected -- non-ambush unit in Neutral Closed
        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": card.instance_id},
        )
        assert not result.ok, "Non-ambush unit should NOT be playable on opponent's turn"
        # The error should mention the card cannot be played in current state
        assert "cannot be played" in result.error.lower() or "state" in result.error.lower(), (
            f"Expected state-related rejection, got: {result.error}"
        )

    def test_ambush_unit_plays_normally_on_own_turn(self):
        """On P1's own turn, play the ambush unit normally.
        Should work just like any normal unit play.
        """
        gs = make_game()
        card = _add_card_to_hand(gs, "p1", AMBUSH_CARD_ID)

        _give_resources(gs, "p1", energy=5, power={Domain.FURY: 3})

        # P1's turn, Neutral Open -- normal play conditions
        gs.turn_player_id = "p1"
        gs.active_player_id = "p1"
        gs.phase = Phase.ACTION

        # Validate
        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": card.instance_id},
        )
        assert result.ok, f"Ambush unit should be playable on own turn: {result.error}"

        # Execute
        logs = execute_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": card.instance_id},
        )

        # Unit should be on the board
        assert card.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD), (
            f"Unit should be on the board, but is in {card.zone.value}"
        )
        assert card.controller_id == "p1"
        assert card.instance_id not in gs.players["p1"].hand

    def test_ambush_with_enter_play_trigger(self):
        """Play Chakram Dancer (unl-071-219) during opponent's turn.

        Chakram Dancer has ambush and an on_play trigger:
        "When you play me, [ready] your other units here."
        (The card text says "give shield" but the effect_ir is "ready".)

        Place a friendly unit at a battlefield, then play Chakram Dancer
        to the same battlefield. Verify the on_play trigger fires and
        readies the other friendly unit.
        """
        gs = make_game()

        # Pick the first battlefield and give P1 control of it
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        # Place a friendly unit at the battlefield (starts exhausted)
        ally = add_unit(gs, "p1", zone=ZoneType.BATTLEFIELD, name="Ally", bf_id=bf_id)
        ally.exhausted = True  # explicitly exhausted

        # Put Chakram Dancer in hand
        dancer = _add_card_to_hand(gs, "p1", CHAKRAM_DANCER_ID)

        # Chakram Dancer costs 3E, mind domain, no power cost
        _give_resources(gs, "p1", energy=10, power={Domain.MIND: 5})

        # Set up P2's turn with chain open, P1 has priority
        _setup_opponent_turn_with_chain(gs)

        # Validate: dancer has ambush, should be playable
        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": dancer.instance_id,
             "destination": {"zone": "battlefield", "id": bf_id}},
        )
        assert result.ok, f"Chakram Dancer should be playable on opponent's turn: {result.error}"

        # Execute the play
        logs = execute_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": dancer.instance_id,
             "destination": {"zone": "battlefield", "id": bf_id}},
        )

        # Dancer should be on the battlefield
        assert dancer.zone == ZoneType.BATTLEFIELD, (
            f"Dancer should be on the battlefield, but is in {dancer.zone.value}"
        )
        assert dancer.instance_id in bf.units, (
            "Dancer should be in the battlefield's unit list"
        )

        # Check that the on_play trigger fired.
        # The trigger pushes an ability to the chain. We need to resolve it.
        # Look for evidence the trigger fired in the logs.
        trigger_fired = any("trigger" in log.lower() for log in logs)

        # The trigger's effect is "ready" on friendly units at this location.
        # Since the chain may still have items, we may need to pass priority
        # to resolve them. Let's check if the chain has trigger items and
        # resolve them.
        resolve_logs = []
        safety = 0
        while not gs.chain.is_empty and safety < 20:
            # Both players pass priority to resolve
            p_logs = execute_action(gs, gs.active_player_id, ActionType.PASS_PRIORITY, {})
            resolve_logs.extend(p_logs)
            safety += 1

        all_logs = logs + resolve_logs

        # After full resolution, the ally should have been readied
        # (the "ready" effect un-exhausts the target).
        assert not ally.exhausted, (
            f"Ally should have been readied by Chakram Dancer's on_play trigger. "
            f"Logs: {all_logs}"
        )
