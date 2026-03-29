"""TS-02: Card Pipeline unit tests for effect text -> IR conversion."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.engine.card_pipeline import (
    convert_card,
    make_basic_rune,
    parse_activated_cost,
    parse_effect_text,
    _extract_keywords,
    _parse_abilities,
    _parse_single_effect,
    _parse_target,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cms_card(
    card_id: str = "test-001",
    name: str = "Test Card",
    card_type: str = "unit",
    text: str = "",
    accessibility_text: str = "",
    might: int = 0,
    energy: int = 0,
    domains: list[str] | None = None,
    might_bonus: int = 0,
) -> dict:
    """Build a minimal CMS-format card dict for pipeline testing."""
    cms: dict = {
        "id": card_id,
        "name": name,
        "cardType": {"type": [{"id": card_type}]},
        "cardImage": {
            "accessibilityText": accessibility_text or text,
            "url": "",
        },
        "text": {"richText": {"body": ""}},
        "domain": {
            "values": [{"id": d} for d in (domains or [])],
        },
        "energy": {"value": energy},
        "might": {"value": might},
        "power": {"value": 0},
    }
    if might_bonus:
        cms["mightBonus"] = {"value": might_bonus}
    return cms


# ---------------------------------------------------------------------------
# Core conversion tests
# ---------------------------------------------------------------------------


class TestConvertUnit:
    def test_basic_unit(self):
        cms = _make_cms_card(
            name="Warrior",
            card_type="unit",
            might=3,
            energy=2,
            domains=["fury"],
        )
        result = convert_card(cms)
        assert result["card_type"] == "unit"
        assert result["base_might"] == 3
        assert result["cost_energy"] == 2
        assert result["domains"] == ["fury"]

    def test_unit_with_keywords(self):
        cms = _make_cms_card(
            name="Shield Bearer",
            card_type="unit",
            might=4,
            text="[Tank] [Shield 2]",
        )
        result = convert_card(cms)
        kw_names = [k["keyword"] for k in result.get("keywords", [])]
        assert "tank" in kw_names
        assert "shield" in kw_names
        shield_kw = next(k for k in result["keywords"] if k["keyword"] == "shield")
        assert shield_kw["value"] == 2


class TestConvertSpell:
    def test_basic_spell(self):
        cms = _make_cms_card(
            name="Fireball",
            card_type="spell",
            energy=3,
            domains=["fury"],
            text="Deal 4 damage to a unit.",
        )
        result = convert_card(cms)
        assert result["card_type"] == "spell"
        assert result["cost_energy"] == 3
        abilities = result.get("abilities", [])
        assert len(abilities) >= 1
        # The first ability should produce deal_damage IR
        ab = abilities[0]
        ir = ab.get("effect_ir")
        assert ir is not None
        assert ir["type"] == "deal_damage"
        assert ir["amount"] == 4


class TestConvertGear:
    def test_gear_with_might_bonus(self):
        cms = _make_cms_card(
            name="Sword of Fury",
            card_type="gear",
            energy=2,
            domains=["fury"],
            might_bonus=2,
        )
        result = convert_card(cms)
        assert result["card_type"] == "gear"
        assert result.get("might_bonus") == 2


class TestConvertRune:
    def test_basic_rune_creation(self):
        result = make_basic_rune("rune_01", "Fury Rune", "fury")
        assert result["card_type"] == "rune"
        assert result["domains"] == ["fury"]
        assert len(result["abilities"]) == 2
        # First ability: exhaust for energy
        ab0 = result["abilities"][0]
        assert ab0["ability_type"] == "activated"
        assert ab0["cost"]["exhaust_source"] is True
        # Second ability: recycle for power
        ab1 = result["abilities"][1]
        assert ab1["ability_type"] == "activated"


# ---------------------------------------------------------------------------
# Pattern matching tests for parse_effect_text / _parse_single_effect
# ---------------------------------------------------------------------------


class TestEffectPatterns:
    """Test that common card text patterns parse to correct IR nodes."""

    def test_deal_damage_to_target_unit(self):
        ir = parse_effect_text("Deal 3 damage to a unit.", "TestCard")
        assert ir is not None
        assert ir["type"] == "deal_damage"
        assert ir["amount"] == 3

    def test_deal_damage_to_enemy(self):
        ir = parse_effect_text("Deal 5 to an enemy unit.", "TestCard")
        assert ir is not None
        assert ir["type"] == "deal_damage"
        assert ir["amount"] == 5
        assert ir["target"]["scope"] == "enemy"

    def test_draw_cards_numeric(self):
        ir = parse_effect_text("Draw 2", "TestCard")
        assert ir is not None
        assert ir["type"] == "draw_cards"
        assert ir["count"] == 2

    def test_draw_a_card(self):
        ir = parse_effect_text("Draw a card", "TestCard")
        assert ir is not None
        assert ir["type"] == "draw_cards"
        assert ir["count"] == 1

    def test_give_might(self):
        ir = parse_effect_text("Give me +2 [S] this turn", "TestCard")
        assert ir is not None
        assert ir["type"] == "give_might"
        assert ir["amount"] == 2

    def test_give_negative_might(self):
        ir = parse_effect_text("Give an enemy unit -2 [S] this turn", "TestCard")
        assert ir is not None
        assert ir["type"] == "give_might"
        assert ir["amount"] == -2
        assert ir["target"]["scope"] == "enemy"

    def test_kill_target_unit(self):
        ir = parse_effect_text("Kill a unit.", "TestCard")
        assert ir is not None
        assert ir["type"] == "kill"

    def test_return_to_hand(self):
        ir = parse_effect_text("Return a unit to its owner's hand", "TestCard")
        assert ir is not None
        assert ir["type"] == "return_to_hand"

    def test_heal(self):
        ir = parse_effect_text("Heal 3 from a friendly unit", "TestCard")
        assert ir is not None
        assert ir["type"] == "heal"
        assert ir["amount"] == 3

    def test_heal_all(self):
        ir = parse_effect_text("Heal me", "TestCard")
        assert ir is not None
        assert ir["type"] == "heal"
        assert ir["amount"] == "all"

    def test_stun_target(self):
        ir = parse_effect_text("Stun an enemy unit", "TestCard")
        assert ir is not None
        assert ir["type"] == "stun"
        assert ir["target"]["scope"] == "enemy"

    def test_buff_target(self):
        ir = parse_effect_text("Buff me", "TestCard")
        assert ir is not None
        assert ir["type"] == "buff"
        assert ir["target"]["scope"] == "self"

    def test_counter_spell(self):
        ir = parse_effect_text("Counter a spell", "TestCard")
        assert ir is not None
        assert ir["type"] == "counter"
        assert ir["target"]["obj_type"] == "spell"

    def test_play_token(self):
        ir = parse_effect_text("Play a 1 [S] Recruit unit token", "TestCard")
        assert ir is not None
        assert ir["type"] == "play_token"
        assert ir["might"] == 1
        assert ir["name"] == "Recruit"

    def test_play_temporary_token(self):
        ir = parse_effect_text("Play a ready 2 [S] Spirit unit token with [Temporary]", "TestCard")
        assert ir is not None
        assert ir["type"] == "play_token"
        assert ir["temporary"] is True
        assert ir.get("ready_on_enter") is True

    def test_gain_xp(self):
        ir = parse_effect_text("Gain 1 XP", "TestCard")
        assert ir is not None
        assert ir["type"] == "gain_xp"
        assert ir["amount"] == 1

    def test_score_points(self):
        ir = parse_effect_text("You score 1 point", "TestCard")
        assert ir is not None
        assert ir["type"] == "score_points"
        assert ir["amount"] == 1

    def test_discard(self):
        ir = parse_effect_text("Discard 1", "TestCard")
        assert ir is not None
        assert ir["type"] == "discard"
        assert ir["count"] == 1

    def test_banish(self):
        ir = parse_effect_text("Banish a unit", "TestCard")
        assert ir is not None
        assert ir["type"] == "banish"

    def test_recycle(self):
        ir = parse_effect_text("Recycle a rune", "TestCard")
        assert ir is not None
        assert ir["type"] == "recycle"

    def test_ready_target(self):
        ir = parse_effect_text("Ready a friendly unit", "TestCard")
        assert ir is not None
        assert ir["type"] == "ready"

    def test_exhaust_target(self):
        ir = parse_effect_text("Exhaust an enemy unit", "TestCard")
        assert ir is not None
        assert ir["type"] == "exhaust"
        assert ir["target"]["scope"] == "enemy"

    def test_channel_rune(self):
        ir = parse_effect_text("Channel 2 runes", "TestCard")
        assert ir is not None
        assert ir["type"] == "channel_rune"
        assert ir["count"] == 2


# ---------------------------------------------------------------------------
# Trigger detection tests
# ---------------------------------------------------------------------------


class TestTriggerParsing:
    """Test that trigger prefix patterns are recognized."""

    def test_on_play_trigger(self):
        abilities = _parse_abilities(
            "When you play me, draw 2", "", "test", "TestCard", "unit", []
        )
        assert len(abilities) >= 1
        ab = abilities[0]
        assert ab["ability_type"] == "triggered"
        assert ab["trigger_condition"] == "on_play"
        assert ab["effect_ir"]["type"] == "draw_cards"

    def test_on_death_trigger(self):
        abilities = _parse_abilities(
            "When I die, deal 2 to an enemy unit", "", "test", "TestCard", "unit", []
        )
        assert len(abilities) >= 1
        ab = abilities[0]
        assert ab["ability_type"] == "triggered"
        assert ab["trigger_condition"] == "on_death"

    def test_on_turn_start_trigger(self):
        abilities = _parse_abilities(
            "At the start of your turn, draw a card", "", "test", "TestCard", "unit", []
        )
        assert len(abilities) >= 1
        ab = abilities[0]
        assert ab["ability_type"] == "triggered"
        assert ab["trigger_condition"] == "on_turn_start"

    def test_on_attack_trigger(self):
        abilities = _parse_abilities(
            "When I attack, gain 1 XP", "", "test", "TestCard", "unit", []
        )
        assert len(abilities) >= 1
        ab = abilities[0]
        assert ab["ability_type"] == "triggered"
        assert ab["trigger_condition"] == "on_attack"

    def test_on_conquer_trigger(self):
        abilities = _parse_abilities(
            "When you conquer, you score 1 point", "", "test", "TestCard", "unit", []
        )
        assert len(abilities) >= 1
        ab = abilities[0]
        assert ab["trigger_condition"] == "on_conquer"


# ---------------------------------------------------------------------------
# Activated ability cost parsing
# ---------------------------------------------------------------------------


class TestActivatedCostParsing:
    def test_exhaust_cost(self):
        cost, remainder = parse_activated_cost("[T]: Draw a card")
        assert cost is not None
        assert cost["exhaust_source"] is True
        assert "Draw a card" in remainder

    def test_energy_cost(self):
        cost, remainder = parse_activated_cost("[4], [T]: Deal 3 to a unit")
        assert cost is not None
        assert cost.get("energy") == 4
        assert cost.get("exhaust_source") is True

    def test_recycle_cost(self):
        cost, remainder = parse_activated_cost("Recycle this: Draw 2")
        assert cost is not None
        assert cost.get("recycle_source") is True
        assert "Draw 2" in remainder

    def test_no_cost(self):
        cost, remainder = parse_activated_cost("Deal 3 to a unit")
        assert cost is None
        assert remainder == "Deal 3 to a unit"


# ---------------------------------------------------------------------------
# Conditional and compositional parsing
# ---------------------------------------------------------------------------


class TestConditionalEffects:
    def test_optional_effect(self):
        ir = parse_effect_text("You may draw a card", "TestCard")
        assert ir is not None
        assert ir["type"] == "optional"
        assert ir["effect"]["type"] == "draw_cards"

    def test_conditional_if(self):
        ir = parse_effect_text(
            "If you have played another card, draw 2", "TestCard"
        )
        assert ir is not None
        assert ir["type"] == "conditional"
        assert ir["condition"]["cond_type"] == "legion"
        assert ir["then"]["type"] == "draw_cards"


class TestSequenceEffects:
    def test_multiple_sentences(self):
        ir = parse_effect_text("Deal 3 to an enemy unit. Draw a card", "TestCard")
        assert ir is not None
        assert ir["type"] == "sequence"
        assert len(ir["steps"]) == 2
        assert ir["steps"][0]["type"] == "deal_damage"
        assert ir["steps"][1]["type"] == "draw_cards"


# ---------------------------------------------------------------------------
# Target parsing tests
# ---------------------------------------------------------------------------


class TestTargetParsing:
    def test_self_target(self):
        t = _parse_target("me")
        assert t["scope"] == "self"

    def test_enemy_unit(self):
        t = _parse_target("an enemy unit")
        assert t["scope"] == "enemy"
        assert t["obj_type"] == "unit"

    def test_friendly_unit(self):
        t = _parse_target("a friendly unit")
        assert t["scope"] == "friendly"
        assert t["obj_type"] == "unit"

    def test_all_units(self):
        t = _parse_target("all units")
        assert t["count"] == -1
        assert t["scope"] == "any"

    def test_all_enemy_units(self):
        t = _parse_target("all enemy units")
        assert t["count"] == -1
        assert t["scope"] == "enemy"

    def test_unit_here(self):
        t = _parse_target("a unit here")
        assert t["location"] == "here"

    def test_another_friendly_unit(self):
        t = _parse_target("another friendly unit")
        assert t["scope"] == "friendly"
        # Should have a not_self filter
        filters = t.get("filters", [])
        assert any(f.get("field") == "not_self" for f in filters)


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------


class TestKeywordExtraction:
    def test_simple_keywords(self):
        keywords, remaining = _extract_keywords("[Tank] [Hidden] Some text")
        kw_names = [k["keyword"] for k in keywords]
        assert "tank" in kw_names
        assert "hidden" in kw_names
        assert "Some text" in remaining

    def test_valued_keyword(self):
        keywords, remaining = _extract_keywords("[Assault 3] Attack text")
        assert len(keywords) >= 1
        assault = next(k for k in keywords if k["keyword"] == "assault")
        assert assault["value"] == 3

    def test_equip_keyword(self):
        keywords, remaining = _extract_keywords("[Equip [2]] Some effect text")
        kw_names = [k["keyword"] for k in keywords]
        assert "equip" in kw_names


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_ability_text(self):
        """Empty string should return None."""
        ir = parse_effect_text("", "TestCard")
        assert ir is None

    def test_whitespace_only(self):
        ir = parse_effect_text("   ", "TestCard")
        assert ir is None

    def test_unknown_text_returns_none(self):
        """Unrecognized text returns None gracefully without crashing."""
        ir = parse_effect_text("Xylophone the quantum flux", "TestCard")
        assert ir is None

    def test_missing_fields_graceful(self):
        """Card definition with missing optional fields does not crash."""
        cms = {
            "id": "edge-001",
            "name": "Minimal Card",
            "cardType": {"type": [{"id": "unit"}]},
            "cardImage": {"accessibilityText": "", "url": ""},
            "text": {"richText": {"body": ""}},
            "domain": {"values": []},
            "energy": {},
            "might": {},
            "power": {},
        }
        result = convert_card(cms)
        assert result["card_id"] == "edge-001"
        assert result["card_type"] == "unit"
        assert result["base_might"] == 0
        assert result["cost_energy"] == 0

    def test_missing_card_type_defaults(self):
        """Card without cardType list defaults to 'unit'."""
        cms = {
            "id": "edge-002",
            "name": "No Type",
            "cardType": {"type": []},
            "cardImage": {"accessibilityText": "", "url": ""},
            "text": {"richText": {"body": ""}},
            "domain": {"values": []},
            "energy": {"value": 0},
            "might": {"value": 0},
            "power": {"value": 0},
        }
        result = convert_card(cms)
        assert result["card_type"] == "unit"

    def test_multiple_abilities(self):
        """Card with two ability lines produces two abilities."""
        text = "When you play me, draw a card\nWhen I die, deal 2 to an enemy unit"
        abilities = _parse_abilities(text, "", "multi", "MultiCard", "unit", [])
        assert len(abilities) == 2
        assert abilities[0]["trigger_condition"] == "on_play"
        assert abilities[1]["trigger_condition"] == "on_death"

    def test_spell_text_becomes_activated_ability(self):
        """For spell cards, the text IS the effect."""
        text = "Deal 4 to a unit"
        abilities = _parse_abilities(text, "", "spell1", "Fireball", "spell", [])
        assert len(abilities) >= 1
        ab = abilities[0]
        assert ab["ability_type"] == "activated"
        assert ab.get("effect_ir", {}).get("type") == "deal_damage"
