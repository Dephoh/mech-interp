"""Proof scenarios for RUNE mechanics: exhaust and recycle.

Every test follows the same structure:
  1. Set up a game state with a rune on the rune board
  2. Execute an action through the full validate->execute->cleanup pipeline
  3. Assert on BOTH the game state AND the serialized frontend view
  4. The contract schema is validated automatically on every action
"""

from __future__ import annotations

import pytest
from helpers import Scenario

from app.engine.card_types import CardDefinition, CardInstance
from app.engine.enums import ActionType, CardType, Domain, ZoneType


def _add_rune(
    s: Scenario,
    player_id: str,
    name: str = "Test Rune",
    domains: tuple[Domain, ...] = (Domain.FURY,),
) -> CardInstance:
    """Place a rune directly on a player's rune board."""
    rune_def = CardDefinition(
        card_id=f"test_{name.lower().replace(' ', '_')}",
        name=name,
        card_type=CardType.RUNE,
        domains=domains,
    )
    rune = CardInstance.create(rune_def, player_id, ZoneType.RUNE_BOARD)
    s.gs.instances[rune.instance_id] = rune
    s.gs.base_runes[player_id].append(rune.instance_id)
    return rune


class TestExhaustRuneProof:
    """Proof: exhausting a rune gains energy and marks it exhausted."""

    def test_exhaust_rune_gains_energy(self):
        """Proof: exhausting a rune adds +1 energy to the rune pool
        and marks the rune as exhausted, both visible in the frontend."""
        s = Scenario()
        rune = _add_rune(s, "p1", name="Fury Rune")
        s.give_resources("p1", energy=0)

        view = s.act("p1", ActionType.EXHAUST_RUNE, {
            "instance_id": rune.instance_id,
        })

        # Game state: rune is exhausted, energy increased
        assert rune.exhausted is True
        assert s.gs.players["p1"].rune_pool.energy == 1

        # Frontend view: energy reflects the gain
        assert view["you"]["rune_pool"]["energy"] == 1

        # Frontend view: rune shows exhausted
        rune_cards = view["you"]["rune_board"]
        rune_view = next(
            c for c in rune_cards if c["instance_id"] == rune.instance_id
        )
        assert rune_view["exhausted"] is True

    def test_exhausted_rune_shows_in_view(self):
        """Proof: after exhausting a rune, the exhausted field on the
        rune card in view['you']['rune_board'] is True."""
        s = Scenario()
        rune = _add_rune(s, "p1", name="Calm Rune", domains=(Domain.CALM,))
        s.give_resources("p1", energy=2)

        # Before exhaust: rune is not exhausted
        view_before = s.view("p1")
        rune_before = next(
            c for c in view_before["you"]["rune_board"]
            if c["instance_id"] == rune.instance_id
        )
        assert rune_before["exhausted"] is False

        # Exhaust the rune
        view_after = s.act("p1", ActionType.EXHAUST_RUNE, {
            "instance_id": rune.instance_id,
        })

        # Frontend view: exhausted field is now True
        rune_after = next(
            c for c in view_after["you"]["rune_board"]
            if c["instance_id"] == rune.instance_id
        )
        assert rune_after["exhausted"] is True

        # Energy went from 2 to 3
        assert view_after["you"]["rune_pool"]["energy"] == 3


class TestRecycleRuneProof:
    """Proof: recycling a rune gains power and moves it off the board."""

    def test_recycle_rune_gains_power(self):
        """Proof: recycling a Fury domain rune adds +1 fury power to
        the rune pool and removes the rune from the rune board."""
        s = Scenario()
        rune = _add_rune(s, "p1", name="Fury Rune", domains=(Domain.FURY,))
        s.give_resources("p1", energy=0, power={Domain.FURY: 0})

        view = s.act("p1", ActionType.RECYCLE_RUNE, {
            "instance_id": rune.instance_id,
        })

        # Game state: rune moved to rune deck, power increased
        assert rune.zone == ZoneType.RUNE_DECK
        assert s.gs.players["p1"].rune_pool.power.get(Domain.FURY, 0) == 1
        assert rune.instance_id not in s.gs.base_runes["p1"]

        # Frontend view: power reflects the gain
        assert view["you"]["rune_pool"]["power"]["fury"] == 1

        # Frontend view: rune no longer on rune_board
        rune_ids = [c["instance_id"] for c in view["you"]["rune_board"]]
        assert rune.instance_id not in rune_ids
