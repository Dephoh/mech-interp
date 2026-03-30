"""Proof scenarios for the ACCELERATE keyword.

Accelerate lets a unit enter play ready (not exhausted) by paying an
additional cost.  These tests prove the full pipeline
(validate -> execute -> cleanup -> serialize) handles Accelerate correctly
and that the frontend view reflects the outcome.
"""

from __future__ import annotations

import pytest
from helpers import Scenario

from app.engine.card_types import KeywordInstance
from app.engine.enums import ActionType, Domain, Keyword


class TestAccelerateProof:
    """Proof: Accelerate keyword lets a unit enter play ready for extra cost."""

    def _make_speedster(self, s: Scenario):
        """Create a unit with Accelerate in p1's hand and give plenty of resources.

        ``add_hand_unit`` creates a card with no domains, so
        ``get_accelerate_cost`` returns ``(1, {})`` -- 1 extra energy,
        no power.  Total cost when accelerated: base 2E + accel 1E = 3E.
        """
        kw = (KeywordInstance(Keyword.ACCELERATE, 0),)
        unit = s.add_hand_unit(
            "p1",
            might=3,
            cost_energy=2,
            name="Speedster",
            keywords=kw,
        )
        s.give_resources("p1", energy=10)
        return unit

    def test_accelerate_unit_enters_ready(self):
        """Proof: paying Accelerate makes the unit enter play NOT exhausted,
        visible in both game state and frontend view."""
        s = Scenario()
        unit = self._make_speedster(s)

        view = s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": unit.instance_id,
            "targets": [],
            "destination": {"zone": "base"},
            "pay_accelerate": True,
        })

        # Game state: unit is ready (not exhausted)
        assert unit.exhausted is False

        # Frontend view: unit shows exhausted=False
        card_view = Scenario.find_card_in_view(view, unit.instance_id)
        assert card_view is not None
        assert card_view["exhausted"] is False

    def test_normal_play_enters_exhausted(self):
        """Proof: playing the same unit WITHOUT Accelerate leaves it exhausted,
        visible in both game state and frontend view."""
        s = Scenario()
        unit = self._make_speedster(s)

        view = s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": unit.instance_id,
            "targets": [],
            "destination": {"zone": "base"},
        })

        # Game state: unit is exhausted (default for entering play)
        assert unit.exhausted is True

        # Frontend view: unit shows exhausted=True
        card_view = Scenario.find_card_in_view(view, unit.instance_id)
        assert card_view is not None
        assert card_view["exhausted"] is True

    def test_accelerate_costs_extra_energy(self):
        """Proof: Accelerate spends extra energy beyond the base cost.

        Base cost: 2 energy.
        Accelerate cost (no-domain card): 1 energy.
        Total when accelerated: 3 energy.
        Without accelerate the same card costs only 2 energy.
        """
        s = Scenario()
        unit = self._make_speedster(s)

        starting_energy = s.gs.players["p1"].rune_pool.energy  # 10

        view = s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": unit.instance_id,
            "targets": [],
            "destination": {"zone": "base"},
            "pay_accelerate": True,
        })

        # Game state: energy spent = base 2 + accelerate 1 = 3
        expected_energy = starting_energy - 3
        assert s.gs.players["p1"].rune_pool.energy == expected_energy

        # Frontend view: energy reflects total spend
        assert view["you"]["rune_pool"]["energy"] == expected_energy
