"""Integration tests for the Accelerate keyword — full pipeline flow.

Rule 731: Accelerate — "As you play me, you may pay [1] energy and 1 Power as an
additional cost. If you do, I enter ready."

Uses card ogn-001-298 (Blazing Scorcher): 5E, fury domain, accelerate keyword.
Accelerate additional cost = 1E + 1 Fury Power.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engine.card_db import CardDB
from app.engine.card_types import CardDefinition, CardInstance, KeywordInstance
from app.engine.enums import ActionType, CardType, Domain, Keyword, Phase, ZoneType
from app.engine.action_executor import execute_action
from app.engine.action_validator import validate_action

from helpers import make_game, load_card_db, make_unit_def


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------


def _setup_game(energy: int = 10, fury_power: int = 5):
    """Create a game with Blazing Scorcher in P1's hand and given resources.

    Returns (GameState, CardInstance).
    """
    load_card_db()
    gs = make_game(phase=Phase.ACTION)

    defn = CardDB.get("ogn-001-298")  # Blazing Scorcher — 5E, fury, accelerate
    card = CardInstance.create(defn, "p1", ZoneType.HAND)
    gs.instances[card.instance_id] = card
    gs.players["p1"].hand.append(card.instance_id)

    gs.players["p1"].rune_pool.add_energy(energy)
    gs.players["p1"].rune_pool.add_power(Domain.FURY, fury_power)

    return gs, card


def _play_card(gs, player_id, card, *, pay_accelerate=False, destination=None):
    """Validate then execute PLAY_CARD.  Returns (validation_result, exec_logs)."""
    payload = {
        "instance_id": card.instance_id,
        "pay_accelerate": pay_accelerate,
    }
    if destination is not None:
        payload["destination"] = destination

    result = validate_action(gs, player_id, ActionType.PLAY_CARD, payload)
    if not result.ok:
        return result, []

    logs = execute_action(gs, player_id, ActionType.PLAY_CARD, payload)
    return result, logs


# ---------------------------------------------------------------------------
# 1. Paid accelerate — unit enters ready
# ---------------------------------------------------------------------------


class TestPlayAccelerateUnitPaidEntersReady:
    """Play an accelerate unit with pay_accelerate=True.

    Verify the full observable state: card leaves hand, card is tracked in
    base_units, zone is BASE, and exhausted is False.
    """

    def test_play_accelerate_unit_paid_enters_ready(self):
        gs, card = _setup_game(energy=10, fury_power=5)
        pool = gs.players["p1"].rune_pool

        result, logs = _play_card(gs, "p1", card, pay_accelerate=True)
        assert result.ok, result.error

        # Card should no longer be in hand
        assert card.instance_id not in gs.players["p1"].hand

        # Card should be on the board (base zone, tracked in base_units)
        assert card.zone == ZoneType.BASE
        assert card.instance_id in gs.base_units["p1"]

        # Accelerated: enters ready (not exhausted)
        assert card.exhausted is False, (
            f"Accelerated unit should enter ready, got exhausted={card.exhausted}"
        )

        # Resources: base 5E + accelerate 1E = 6E deducted; fury: 0 + 1 = 1 deducted
        assert pool.energy == 4  # 10 - 6
        assert pool.power.get(Domain.FURY, 0) == 4  # 5 - 1


# ---------------------------------------------------------------------------
# 2. Unpaid accelerate — unit enters exhausted
# ---------------------------------------------------------------------------


class TestPlayAccelerateUnitUnpaidEntersExhausted:
    """Play an accelerate unit without paying accelerate.

    Unit enters exhausted (normal).  Only base cost deducted.
    """

    def test_play_accelerate_unit_unpaid_enters_exhausted(self):
        gs, card = _setup_game(energy=10, fury_power=5)
        pool = gs.players["p1"].rune_pool

        result, logs = _play_card(gs, "p1", card, pay_accelerate=False)
        assert result.ok, result.error

        # Card is on the board
        assert card.instance_id not in gs.players["p1"].hand
        assert card.zone == ZoneType.BASE
        assert card.instance_id in gs.base_units["p1"]

        # Not accelerated: enters exhausted
        assert card.exhausted is True

        # Only base cost deducted (5E, 0 power)
        assert pool.energy == 5  # 10 - 5
        assert pool.power.get(Domain.FURY, 0) == 5  # unchanged


# ---------------------------------------------------------------------------
# 3. Exact cost — single-domain accelerate drains resources to zero
# ---------------------------------------------------------------------------


class TestAccelerateCostCorrectSingleDomain:
    """Blazing Scorcher: 5E base, fury domain.

    Accelerate adds +1E +1 Fury.  Total: 6E + 1 Fury.
    Give player exactly 6E and 1 Fury.  After play, both should be 0.
    """

    def test_accelerate_cost_correct_single_domain(self):
        gs, card = _setup_game(energy=6, fury_power=1)
        pool = gs.players["p1"].rune_pool

        result, logs = _play_card(gs, "p1", card, pay_accelerate=True)
        assert result.ok, result.error

        assert pool.energy == 0, f"Expected 0 energy, got {pool.energy}"
        assert pool.power.get(Domain.FURY, 0) == 0, (
            f"Expected 0 fury, got {pool.power.get(Domain.FURY, 0)}"
        )

        # Also confirm the unit entered ready
        assert card.exhausted is False


# ---------------------------------------------------------------------------
# 4. Multi-domain / no-domain: accelerate power cost is "any"
# ---------------------------------------------------------------------------


class TestAccelerateCostCorrectMultiDomain:
    """A unit with two domains should have accelerate cost = 1E + 1 power of any domain.

    The engine returns an empty power dict for multi-domain units, which means
    the accelerate energy cost (1E) is still charged but no specific power domain
    is required.  We verify via a synthetic multi-domain accelerate unit.
    """

    def test_accelerate_cost_multi_domain(self):
        load_card_db()
        gs = make_game(phase=Phase.ACTION)

        # Create a synthetic unit with two domains + accelerate
        defn = CardDefinition(
            card_id="test_multi_domain",
            name="Multi-Domain Accelerator",
            card_type=CardType.UNIT,
            domains=(Domain.FURY, Domain.CALM),
            base_might=3,
            keywords=(KeywordInstance(keyword=Keyword.ACCELERATE),),
            cost_energy=3,
        )
        card = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[card.instance_id] = card
        gs.players["p1"].hand.append(card.instance_id)

        # Give exactly enough: 3E base + 1E accelerate = 4E, and no power needed
        # (multi-domain accelerate returns empty power dict)
        gs.players["p1"].rune_pool.add_energy(4)
        pool = gs.players["p1"].rune_pool

        result, logs = _play_card(gs, "p1", card, pay_accelerate=True)
        assert result.ok, result.error

        # 4E - 4E = 0E, no power cost
        assert pool.energy == 0, f"Expected 0 energy, got {pool.energy}"

        # Unit should enter ready
        assert card.exhausted is False

    def test_accelerate_cost_no_domain(self):
        """A domainless unit has the same behavior — empty power dict."""
        load_card_db()
        gs = make_game(phase=Phase.ACTION)

        defn = CardDefinition(
            card_id="test_no_domain",
            name="Domainless Accelerator",
            card_type=CardType.UNIT,
            domains=(),
            base_might=2,
            keywords=(KeywordInstance(keyword=Keyword.ACCELERATE),),
            cost_energy=2,
        )
        card = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[card.instance_id] = card
        gs.players["p1"].hand.append(card.instance_id)

        # 2E base + 1E accelerate = 3E total, 0 power
        gs.players["p1"].rune_pool.add_energy(3)
        pool = gs.players["p1"].rune_pool

        result, logs = _play_card(gs, "p1", card, pay_accelerate=True)
        assert result.ok, result.error

        assert pool.energy == 0
        assert card.exhausted is False


# ---------------------------------------------------------------------------
# 5. Cannot afford accelerate — validation rejects
# ---------------------------------------------------------------------------


class TestCannotAffordAccelerateRejected:
    """Player has enough for base cost (5E) but NOT enough for accelerate.

    Three sub-cases: short on energy, short on power, short on both.
    """

    def test_short_on_energy_for_accelerate(self):
        """5E + 5 Fury: enough power, but not enough energy (needs 6E)."""
        gs, card = _setup_game(energy=5, fury_power=5)

        result = validate_action(gs, "p1", ActionType.PLAY_CARD, {
            "instance_id": card.instance_id,
            "pay_accelerate": True,
        })
        assert not result.ok, "Should reject: not enough energy for accelerate"

    def test_short_on_power_for_accelerate(self):
        """6E + 0 Fury: enough energy, but no fury power (needs 1)."""
        gs, card = _setup_game(energy=6, fury_power=0)

        result = validate_action(gs, "p1", ActionType.PLAY_CARD, {
            "instance_id": card.instance_id,
            "pay_accelerate": True,
        })
        assert not result.ok, "Should reject: not enough fury power for accelerate"

    def test_short_on_both(self):
        """5E + 0 Fury: short on both energy and power."""
        gs, card = _setup_game(energy=5, fury_power=0)

        result = validate_action(gs, "p1", ActionType.PLAY_CARD, {
            "instance_id": card.instance_id,
            "pay_accelerate": True,
        })
        assert not result.ok, "Should reject: can't afford accelerate at all"


# ---------------------------------------------------------------------------
# 6. Can play without accelerate even when can't afford it
# ---------------------------------------------------------------------------


class TestCanPlayWithoutAccelerateWhenCantAffordIt:
    """A player who can afford the base cost but NOT the accelerate cost
    should still be able to play the card with pay_accelerate=False.
    """

    def test_play_without_accelerate_when_broke(self):
        gs, card = _setup_game(energy=5, fury_power=0)

        result, logs = _play_card(gs, "p1", card, pay_accelerate=False)
        assert result.ok, f"Should succeed without accelerate: {result.error}"

        # Card is on the board, exhausted (no accelerate)
        assert card.zone == ZoneType.BASE
        assert card.exhausted is True
        assert gs.players["p1"].rune_pool.energy == 0  # 5 - 5

    def test_play_without_accelerate_when_short_on_energy(self):
        """5E + 5 Fury but need 6E for accelerate — play without is fine."""
        gs, card = _setup_game(energy=5, fury_power=5)

        result, logs = _play_card(gs, "p1", card, pay_accelerate=False)
        assert result.ok, f"Should succeed without accelerate: {result.error}"

        assert card.zone == ZoneType.BASE
        assert card.exhausted is True
        assert gs.players["p1"].rune_pool.energy == 0  # 5 - 5
        assert gs.players["p1"].rune_pool.power[Domain.FURY] == 5  # untouched


# ---------------------------------------------------------------------------
# 7. Accelerated unit can act immediately (move to battlefield)
# ---------------------------------------------------------------------------


class TestAccelerateUnitCanActImmediately:
    """An accelerated (ready) unit should be able to move to a battlefield
    in the same turn.  A non-accelerated (exhausted) unit should be blocked.
    """

    def test_accelerated_unit_can_move_to_battlefield(self):
        gs, card = _setup_game(energy=10, fury_power=5)

        result, logs = _play_card(gs, "p1", card, pay_accelerate=True)
        assert result.ok, result.error
        assert card.exhausted is False, "Unit should be ready after accelerate"

        # Pick a battlefield to move to
        bf_id = list(gs.battlefields.keys())[0]

        # Validate the move
        move_result = validate_action(gs, "p1", ActionType.MOVE_UNIT, {
            "instance_id": card.instance_id,
            "destination": {"zone": "battlefield", "id": bf_id},
        })
        assert move_result.ok, (
            f"Ready unit should be able to move: {move_result.error}"
        )

        # Execute the move
        move_logs = execute_action(gs, "p1", ActionType.MOVE_UNIT, {
            "instance_id": card.instance_id,
            "destination": {"zone": "battlefield", "id": bf_id},
        })

        # Unit should now be at the battlefield
        assert card.zone == ZoneType.BATTLEFIELD
        assert card.instance_id in gs.battlefields[bf_id].units

    def test_non_accelerated_unit_cannot_move(self):
        """An exhausted unit (no accelerate paid) should fail the move validation."""
        gs, card = _setup_game(energy=10, fury_power=5)

        result, logs = _play_card(gs, "p1", card, pay_accelerate=False)
        assert result.ok, result.error
        assert card.exhausted is True, "Unit should be exhausted without accelerate"

        bf_id = list(gs.battlefields.keys())[0]

        move_result = validate_action(gs, "p1", ActionType.MOVE_UNIT, {
            "instance_id": card.instance_id,
            "destination": {"zone": "battlefield", "id": bf_id},
        })
        assert not move_result.ok, "Exhausted unit should NOT be able to move"
        assert "exhausted" in move_result.error.lower()
