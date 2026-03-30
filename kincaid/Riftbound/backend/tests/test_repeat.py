"""Tests for the Repeat keyword (Rule 746).

Repeat lets a spell's controller pay an additional energy cost to execute
the spell's instructions a second time during resolution.
"""

from __future__ import annotations

import pytest
from helpers import add_unit, make_game

from app.engine.card_types import (
    AbilityDefinition,
    CardDefinition,
    CardInstance,
    KeywordInstance,
)
from app.engine.chain import push_card_to_chain, resolve_top_item
from app.engine.effect_ir import deal_damage, draw_cards
from app.engine.enums import CardType, Domain, Keyword, ZoneType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repeat_spell(
    name: str = "Repeat Spell",
    repeat_cost: int = 2,
    effect_ir: dict | None = None,
    cost_energy: int = 1,
    domains: tuple[Domain, ...] = (Domain.FURY,),
) -> CardDefinition:
    """Build a spell definition with the Repeat keyword."""
    if effect_ir is None:
        effect_ir = deal_damage(2)
    return CardDefinition(
        card_id="test_repeat_spell",
        name=name,
        card_type=CardType.SPELL,
        domains=domains,
        cost_energy=cost_energy,
        keywords=(
            KeywordInstance(keyword=Keyword.REPEAT, value=repeat_cost),
        ),
        abilities=(
            AbilityDefinition(
                ability_id="test_repeat_spell_ab",
                ability_type="activated",
                effect_ir=effect_ir,
            ),
        ),
    )


def _play_spell(gs, spell_def, controller="p1", targets=None):
    """Create a spell instance, place it on the chain, and return it."""
    inst = CardInstance.create(spell_def, controller, ZoneType.HAND)
    gs.instances[inst.instance_id] = inst
    push_card_to_chain(gs, inst, controller, targets=targets or [])
    return inst


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRepeatKeyword:
    """Rule 746: Repeat — pay additional cost to execute spell again."""

    def test_repeat_deals_damage_twice(self):
        """A Repeat spell with enough energy should deal damage twice."""
        gs = make_game()
        target = add_unit(gs, "p2", might=10, name="Tough Target")
        gs.players["p1"].rune_pool.energy = 5  # enough for repeat cost of 2

        spell_def = _make_repeat_spell(
            name="Blast",
            repeat_cost=2,
            effect_ir=deal_damage(3),
        )
        spell = _play_spell(gs, spell_def, targets=[target.instance_id])

        # Resolve the spell from the chain
        logs = resolve_top_item(gs)

        # First execution: 3 damage. Repeat: 3 more damage. Total: 6.
        assert target.damage == 6
        assert any("Repeat" in log for log in logs)
        # Energy should have been spent: 2 for repeat
        assert gs.players["p1"].rune_pool.energy == 3

    def test_repeat_not_affordable(self):
        """If the controller cannot pay the Repeat cost, spell fires only once."""
        gs = make_game()
        target = add_unit(gs, "p2", might=10, name="Target")
        gs.players["p1"].rune_pool.energy = 1  # not enough for repeat cost of 2

        spell_def = _make_repeat_spell(
            name="Blast",
            repeat_cost=2,
            effect_ir=deal_damage(3),
        )
        spell = _play_spell(gs, spell_def, targets=[target.instance_id])

        logs = resolve_top_item(gs)

        # Only one execution: 3 damage
        assert target.damage == 3
        assert not any("Repeat" in log for log in logs)
        # Energy untouched (couldn't afford repeat)
        assert gs.players["p1"].rune_pool.energy == 1

    def test_repeat_zero_energy_cost(self):
        """A spell with Repeat cost 0 should repeat for free."""
        gs = make_game()
        target = add_unit(gs, "p2", might=10, name="Target")
        gs.players["p1"].rune_pool.energy = 0

        spell_def = _make_repeat_spell(
            name="Free Repeat",
            repeat_cost=0,
            effect_ir=deal_damage(1),
        )
        spell = _play_spell(gs, spell_def, targets=[target.instance_id])

        logs = resolve_top_item(gs)

        # Two executions: 1 + 1 = 2 damage
        assert target.damage == 2
        assert any("Repeat" in log for log in logs)

    def test_repeat_draws_cards_twice(self):
        """Repeat works with draw effects, not just damage."""
        gs = make_game()
        ps = gs.players["p1"]
        ps.rune_pool.energy = 5

        # Put cards in deck to draw
        for i in range(4):
            deck_card_def = CardDefinition(
                card_id=f"deck_{i}", name=f"Deck Card {i}",
                card_type=CardType.UNIT, base_might=1,
            )
            deck_card = CardInstance.create(deck_card_def, "p1", ZoneType.MAIN_DECK)
            gs.instances[deck_card.instance_id] = deck_card
            ps.main_deck.append(deck_card.instance_id)

        spell_def = _make_repeat_spell(
            name="Double Draw",
            repeat_cost=2,
            effect_ir=draw_cards(1),
        )
        spell = _play_spell(gs, spell_def)

        logs = resolve_top_item(gs)

        # Drew 1 + 1 = 2 cards
        assert len(ps.hand) == 2
        assert ps.rune_pool.energy == 3

    def test_repeat_only_once(self):
        """Even with huge energy, Repeat should only fire once (no infinite loop)."""
        gs = make_game()
        target = add_unit(gs, "p2", might=100, name="Tank")
        gs.players["p1"].rune_pool.energy = 99  # way more than enough

        spell_def = _make_repeat_spell(
            name="Blast",
            repeat_cost=1,
            effect_ir=deal_damage(5),
        )
        spell = _play_spell(gs, spell_def, targets=[target.instance_id])

        logs = resolve_top_item(gs)

        # Exactly two executions: 5 + 5 = 10, not 5 * 99
        assert target.damage == 10
        assert gs.players["p1"].rune_pool.energy == 98

    def test_no_repeat_without_keyword(self):
        """A spell without Repeat should execute exactly once."""
        gs = make_game()
        target = add_unit(gs, "p2", might=10, name="Target")
        gs.players["p1"].rune_pool.energy = 99

        # Spell without Repeat keyword
        spell_def = CardDefinition(
            card_id="plain_spell",
            name="Plain Blast",
            card_type=CardType.SPELL,
            abilities=(
                AbilityDefinition(
                    ability_id="plain_ab",
                    ability_type="activated",
                    effect_ir=deal_damage(3),
                ),
            ),
        )
        spell = _play_spell(gs, spell_def, targets=[target.instance_id])

        logs = resolve_top_item(gs)

        assert target.damage == 3
        assert gs.players["p1"].rune_pool.energy == 99

    def test_spell_goes_to_trash_after_repeat(self):
        """After Repeat execution, the spell still goes to trash normally."""
        gs = make_game()
        target = add_unit(gs, "p2", might=10, name="Target")
        gs.players["p1"].rune_pool.energy = 5

        spell_def = _make_repeat_spell(
            name="Blast",
            repeat_cost=2,
            effect_ir=deal_damage(1),
        )
        spell = _play_spell(gs, spell_def, targets=[target.instance_id])

        logs = resolve_top_item(gs)

        assert spell.zone == ZoneType.TRASH
        assert spell.instance_id in gs.players["p1"].trash
        assert any("trash" in log for log in logs)

    def test_repeat_energy_exactly_matches_cost(self):
        """When energy exactly matches Repeat cost, it should still repeat."""
        gs = make_game()
        target = add_unit(gs, "p2", might=10, name="Target")
        gs.players["p1"].rune_pool.energy = 2  # exactly the repeat cost

        spell_def = _make_repeat_spell(
            name="Blast",
            repeat_cost=2,
            effect_ir=deal_damage(4),
        )
        spell = _play_spell(gs, spell_def, targets=[target.instance_id])

        logs = resolve_top_item(gs)

        assert target.damage == 8  # 4 + 4
        assert gs.players["p1"].rune_pool.energy == 0
