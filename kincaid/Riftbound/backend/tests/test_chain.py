"""Tests for chain mechanics: push, pass priority, resolve."""

import pytest
from helpers import make_game, add_unit, make_spell_def

from app.engine.card_types import CardInstance
from app.engine.chain import pass_priority, push_card_to_chain, resolve_top_item
from app.engine.enums import ZoneType


class TestChainPush:
    def test_push_spell_to_chain(self):
        gs = make_game()
        spell_def = make_spell_def(effect_ir={"type": "deal_damage", "amount": 3})
        spell = CardInstance.create(spell_def, "p1", ZoneType.HAND)
        gs.instances[spell.instance_id] = spell

        item = push_card_to_chain(gs, spell, "p1", [])
        assert not gs.chain.is_empty
        assert spell.zone == ZoneType.CHAIN
        assert item.controller_id == "p1"

    def test_push_clears_passes(self):
        gs = make_game()
        gs.chain.passed_players.add("p1")
        spell_def = make_spell_def()
        spell = CardInstance.create(spell_def, "p1", ZoneType.HAND)
        gs.instances[spell.instance_id] = spell

        push_card_to_chain(gs, spell, "p1", [])
        assert len(gs.chain.passed_players) == 0


class TestPassPriority:
    def test_single_pass_no_resolve(self):
        gs = make_game()
        spell_def = make_spell_def()
        spell = CardInstance.create(spell_def, "p1", ZoneType.HAND)
        gs.instances[spell.instance_id] = spell
        push_card_to_chain(gs, spell, "p1", [])

        should_resolve = pass_priority(gs, "p1")
        assert not should_resolve
        assert "p1" in gs.chain.passed_players

    def test_both_pass_triggers_resolve(self):
        gs = make_game()
        spell_def = make_spell_def()
        spell = CardInstance.create(spell_def, "p1", ZoneType.HAND)
        gs.instances[spell.instance_id] = spell
        push_card_to_chain(gs, spell, "p1", [])

        pass_priority(gs, "p1")
        should_resolve = pass_priority(gs, "p2")
        assert should_resolve


class TestResolveTop:
    def test_resolve_spell_goes_to_trash(self):
        gs = make_game()
        spell_def = make_spell_def(effect_ir={"type": "deal_damage", "amount": 3})
        spell = CardInstance.create(spell_def, "p1", ZoneType.HAND)
        gs.instances[spell.instance_id] = spell
        gs.players["p1"].hand.append(spell.instance_id)

        push_card_to_chain(gs, spell, "p1", [])
        logs = resolve_top_item(gs)

        assert gs.chain.is_empty
        assert spell.zone == ZoneType.TRASH

    def test_resolve_unit_enters_base(self):
        gs = make_game()
        from helpers import make_unit_def
        unit_def = make_unit_def()
        unit = CardInstance.create(unit_def, "p1", ZoneType.HAND)
        gs.instances[unit.instance_id] = unit

        push_card_to_chain(gs, unit, "p1", [])
        logs = resolve_top_item(gs)

        assert gs.chain.is_empty
        assert unit.zone == ZoneType.BASE
        assert unit.exhausted  # enters exhausted by default
        assert unit.instance_id in gs.base_units["p1"]

    def test_lifo_ordering(self):
        gs = make_game()
        spell1_def = make_spell_def(card_id="s1", name="Spell1")
        spell2_def = make_spell_def(card_id="s2", name="Spell2")

        s1 = CardInstance.create(spell1_def, "p1", ZoneType.HAND)
        s2 = CardInstance.create(spell2_def, "p2", ZoneType.HAND)
        gs.instances[s1.instance_id] = s1
        gs.instances[s2.instance_id] = s2

        push_card_to_chain(gs, s1, "p1", [])
        push_card_to_chain(gs, s2, "p2", [])

        # Resolve top — should be s2 (LIFO)
        logs = resolve_top_item(gs)
        assert s2.zone == ZoneType.TRASH
        assert s1.zone == ZoneType.CHAIN  # still on chain

        logs = resolve_top_item(gs)
        assert s1.zone == ZoneType.TRASH
        assert gs.chain.is_empty
