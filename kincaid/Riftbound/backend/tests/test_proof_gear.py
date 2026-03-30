"""Proof scenarios for GEAR card mechanics.

Each test proves that gear actions produce correct frontend-visible results.
The Scenario class validates every action through the full
validate -> execute -> cleanup pipeline, then schema-validates the serialized
output before any manual assertions run.
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.enums import ActionType, ZoneType


class TestGearProof:
    """Proof: gear cards enter the base, boost attached units, and are visible
    to both players."""

    def test_play_gear_to_base(self):
        """Proof: playing a gear card spends energy and the gear appears in
        the player's base_gear in the frontend view with correct might_bonus.
        Gear finalizes immediately on the chain (rule 356.2) -- no
        pass_priority needed."""
        s = Scenario()
        gear = s.add_hand_gear("p1", name="Sword", cost_energy=1, might_bonus=2)
        s.give_resources("p1", energy=5)

        view = s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": gear.instance_id,
            "targets": [],
            "destination": {"zone": "base"},
        })

        # Game state: gear in base, energy spent
        assert gear.zone == ZoneType.BASE
        assert s.gs.players["p1"].rune_pool.energy == 4  # 5 - 1

        # Game state: gear is in base_gear list
        assert gear.instance_id in s.gs.base_gear["p1"]

        # Frontend view: energy reflects the cost
        assert view["you"]["rune_pool"]["energy"] == 4

        # Frontend view: gear visible in base_gear
        base_gear_ids = [c["instance_id"] for c in view["you"]["base_gear"]]
        assert gear.instance_id in base_gear_ids

        # Frontend view: gear has correct might_bonus
        gear_view = next(
            c for c in view["you"]["base_gear"]
            if c["instance_id"] == gear.instance_id
        )
        assert gear_view["might_bonus"] == 2

        # Frontend view: gear no longer in hand
        hand_ids = [c["instance_id"] for c in view["you"]["hand"]]
        assert gear.instance_id not in hand_ids

    def test_gear_attached_to_unit_boosts_might(self):
        """Proof: attaching gear to a unit increases the unit's effective_might
        in the frontend view by the gear's might_bonus."""
        s = Scenario()
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Warrior")
        gear = s.add_hand_gear("p1", name="Shield", cost_energy=1, might_bonus=2)
        s.give_resources("p1", energy=5)

        # Play gear to base first
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": gear.instance_id,
            "targets": [],
            "destination": {"zone": "base"},
        })

        # Manually attach gear to unit
        gear.attached_to = unit.instance_id
        unit.attached_cards.append(gear.instance_id)
        s.gs.recalculate_modifiers()

        # Game state: gear_might_bonus applied
        assert unit.gear_might_bonus == 2
        assert unit.effective_might == 5  # 3 base + 2 gear

        # Frontend view: unit shows boosted effective_might
        view = s.view("p1")
        unit_view = Scenario.find_card_in_view(view, unit.instance_id)
        assert unit_view is not None
        assert unit_view["effective_might"] == 5
        assert unit_view["base_might"] == 3  # base unchanged

        # Frontend view: gear shows attachment
        gear_view = Scenario.find_card_in_view(view, gear.instance_id)
        assert gear_view is not None
        assert gear_view["attached_to"] == unit.instance_id

        # Frontend view: unit shows attached gear
        assert gear.instance_id in unit_view["attached_cards"]

    def test_gear_visible_to_opponent(self):
        """Proof: gear played to base is visible to the opponent in their
        view under opponent.base_gear."""
        s = Scenario()
        gear = s.add_hand_gear("p1", name="Helm", cost_energy=1, might_bonus=1)
        s.give_resources("p1", energy=5)

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": gear.instance_id,
            "targets": [],
            "destination": {"zone": "base"},
        })

        # Opponent's frontend view: gear appears in opponent.base_gear
        opp_view = s.view("p2")
        opp_gear_ids = [c["instance_id"] for c in opp_view["opponent"]["base_gear"]]
        assert gear.instance_id in opp_gear_ids

        # Opponent can see gear details (gear is public information)
        gear_view = next(
            c for c in opp_view["opponent"]["base_gear"]
            if c["instance_id"] == gear.instance_id
        )
        assert gear_view["name"] == "Helm"
        assert gear_view["might_bonus"] == 1
        assert gear_view["card_type"] == "gear"
