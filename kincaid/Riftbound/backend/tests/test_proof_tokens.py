"""Proof scenarios: token creation produces correct frontend-visible results.

Each test proves that playing a spell with a play_token effect creates a
token unit that is correctly visible in the serialized frontend view for
both the controller and the opponent.
"""

from __future__ import annotations

import pytest
from helpers import Scenario

from app.engine.effect_ir import play_token
from app.engine.enums import ActionType


class TestTokenCreationProof:
    """Proof: play_token effects create units visible in the frontend."""

    def test_token_created_by_spell(self):
        """Proof: a spell with play_token("Recruit", 1) creates a 1-might
        token unit that appears in the controller's base_units in the
        frontend view after the spell resolves."""
        s = Scenario()
        spell = s.add_hand_spell(
            "p1", name="Summon", cost_energy=2,
            effect_ir=play_token(name="Recruit", might=1),
        )
        s.give_resources("p1", energy=10)

        base_before = len(s.gs.base_units["p1"])

        # Play the spell
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        # Resolve: both players pass priority
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: one new unit in p1's base
        assert len(s.gs.base_units["p1"]) == base_before + 1
        token_id = s.gs.base_units["p1"][-1]
        token_inst = s.gs.instances[token_id]
        assert token_inst.definition.name == "Recruit"
        assert token_inst.definition.base_might == 1

        # Frontend view: token visible in controller's base_units
        view = s.view("p1")
        base_names = [c["name"] for c in view["you"]["base_units"]]
        assert "Recruit" in base_names

        token_view = next(
            c for c in view["you"]["base_units"] if c["instance_id"] == token_id
        )
        assert token_view["name"] == "Recruit"
        assert token_view["effective_might"] == 1

    def test_token_visible_to_opponent(self):
        """Proof: after a token is created, the opponent can see it in
        their view under opponent.base_units with full card details."""
        s = Scenario()
        spell = s.add_hand_spell(
            "p1", name="Conjure", cost_energy=2,
            effect_ir=play_token(name="Spirit", might=3),
        )
        s.give_resources("p1", energy=10)

        # Play and resolve
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        token_id = s.gs.base_units["p1"][-1]

        # Frontend: opponent sees the token with full details
        opp_view = s.view("p2")
        opp_base = opp_view["opponent"]["base_units"]
        opp_ids = [c["instance_id"] for c in opp_base]
        assert token_id in opp_ids

        token_view = next(c for c in opp_base if c["instance_id"] == token_id)
        assert token_view["name"] == "Spirit"
        assert token_view["effective_might"] == 3
        assert token_view["base_might"] == 3

    def test_token_with_custom_might(self):
        """Proof: a token created with might=4 shows effective_might=4
        in the frontend view."""
        s = Scenario()
        spell = s.add_hand_spell(
            "p1", name="Raise Guard", cost_energy=3,
            effect_ir=play_token(name="Guardian", might=4),
        )
        s.give_resources("p1", energy=10)

        # Play and resolve
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        token_id = s.gs.base_units["p1"][-1]
        token_inst = s.gs.instances[token_id]

        # Game state: base might is 4
        assert token_inst.definition.base_might == 4

        # Frontend view: effective_might matches
        view = s.view("p1")
        token_view = next(
            c for c in view["you"]["base_units"] if c["instance_id"] == token_id
        )
        assert token_view["effective_might"] == 4
        assert token_view["base_might"] == 4
        assert token_view["name"] == "Guardian"
