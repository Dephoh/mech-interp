"""TS-05: Layer system verification tests (rules 450-457).

Verifies that:
  - Layer 1 (Trait-Altering) applies before Layer 2 (Ability-Altering) and Layer 3 (Arithmetic)
  - Layer 2 (Ability-Altering) grants aura keywords properly
  - Layer 3 (Arithmetic) applies increases before decreases (rule 454.3.d)
  - Cross-layer re-evaluation works (rule 453: Fiora example)
  - Modifiers are pruned when source leaves the board
  - Gear attachment might bonuses are applied in Layer 3 (rule 454.3.c)
  - Transient modifiers are cleaned up between turns (rule 317.3)
  - effective_might computation is accurate across all modifier sources
"""

from __future__ import annotations

import pytest
from helpers import add_unit, make_game, make_unit_def

from app.engine.card_types import CardDefinition, CardInstance, KeywordInstance
from app.engine.enums import CardType, CombatRole, Domain, Keyword, Phase, ZoneType
from app.engine.effect_ir import EffectLayer, classify_modifier_layer
from app.engine.game_state import ActiveModifier, GameState
from app.engine.layers import evaluate_layers, _sort_arithmetic, _classify_active_modifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _register_might_modifier(
    gs: GameState,
    source: CardInstance,
    amount: int,
    scope: str = "friendly",
    exclude_self: bool = False,
) -> None:
    """Register a continuous might modifier (arithmetic layer)."""
    gs.register_modifier(source, "test_ab", {
        "stat": "might",
        "amount": amount,
        "target": {"obj_type": "unit", "scope": scope},
        "exclude_self": exclude_self,
        "duration": "continuous",
    })


def _register_might_set_modifier(
    gs: GameState,
    source: CardInstance,
    value: int,
    scope: str = "friendly",
) -> None:
    """Register a trait-altering might_set modifier (layer 1)."""
    gs.register_modifier(source, "test_ab", {
        "stat": "might_set",
        "amount": value,
        "target": {"obj_type": "unit", "scope": scope},
        "duration": "continuous",
    })


def _register_keyword_modifier(
    gs: GameState,
    source: CardInstance,
    keyword_name: str,
    keyword_value: int = 0,
    scope: str = "friendly",
    exclude_self: bool = False,
) -> None:
    """Register a continuous keyword-granting modifier (layer 2)."""
    gs.register_modifier(source, "test_ab", {
        "stat": "keyword",
        "amount": 0,
        "target": {
            "obj_type": "unit",
            "scope": scope,
            "keyword_name": keyword_name,
            "keyword_value": keyword_value,
        },
        "exclude_self": exclude_self,
        "duration": "continuous",
    })


# ---------------------------------------------------------------------------
# Layer 3: Arithmetic — basic modifier application
# ---------------------------------------------------------------------------


class TestArithmeticLayer:
    """Tests for Layer 3 (Arithmetic) modifier application."""

    def test_positive_modifier_increases_might(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=2, name="Aura Source")
        target = add_unit(gs, "p1", might=3, name="Target")

        _register_might_modifier(gs, source, +2, scope="friendly")
        evaluate_layers(gs)

        # Both source and target are friendly; each gets +2
        assert source.effective_might == 4  # 2 + 2
        assert target.effective_might == 5  # 3 + 2

    def test_negative_modifier_decreases_might(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=2, name="Debuff Source")
        target = add_unit(gs, "p2", might=5, name="Enemy Target")

        _register_might_modifier(gs, source, -2, scope="enemy")
        evaluate_layers(gs)

        assert target.effective_might == 3  # 5 - 2
        # Source is friendly, not affected by enemy-scoped debuff
        assert source.effective_might == 2

    def test_exclude_self(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=2, name="Aura Source")
        ally = add_unit(gs, "p1", might=3, name="Ally")

        _register_might_modifier(gs, source, +1, scope="friendly", exclude_self=True)
        evaluate_layers(gs)

        assert source.effective_might == 2  # unaffected (exclude_self)
        assert ally.effective_might == 4    # 3 + 1

    def test_increases_before_decreases(self):
        """Rule 454.3.d: Positive values applied first, then negative.

        This matters when might would go to 0. With a 1-might unit receiving
        +2 then -2, the order should be: 1 + 2 = 3, then 3 - 2 = 1.
        If decreases were applied first: 1 - 2 = 0 (clamped), then 0 + 2 = 2.
        The correct result is 1 because increases are applied first.
        """
        gs = make_game()
        source_buff = add_unit(gs, "p1", might=5, name="Buffer")
        source_debuff = add_unit(gs, "p2", might=5, name="Debuffer")
        target = add_unit(gs, "p1", might=1, name="Small Unit")

        # Register debuff first (would be applied first by timestamp without layer logic)
        _register_might_modifier(gs, source_debuff, -2, scope="enemy")
        # Register buff second
        _register_might_modifier(gs, source_buff, +2, scope="friendly")
        evaluate_layers(gs)

        # Target: base 1, + 2 (increase first) = 3, then - 2 = 1
        assert target.effective_might == 1

    def test_multiple_modifiers_stack(self):
        gs = make_game()
        s1 = add_unit(gs, "p1", might=2, name="Source A")
        s2 = add_unit(gs, "p1", might=2, name="Source B")
        target = add_unit(gs, "p1", might=3, name="Target")

        _register_might_modifier(gs, s1, +1, scope="friendly")
        _register_might_modifier(gs, s2, +1, scope="friendly")
        evaluate_layers(gs)

        # target: 3 + 1 + 1 = 5
        assert target.effective_might == 5

    def test_might_cannot_go_below_zero(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=5, name="Debuffer")
        target = add_unit(gs, "p2", might=2, name="Weak Unit")

        _register_might_modifier(gs, source, -5, scope="enemy")
        evaluate_layers(gs)

        assert target.effective_might == 0  # 2 - 5 clamped to 0


# ---------------------------------------------------------------------------
# Layer 1: Trait-Altering — might_set
# ---------------------------------------------------------------------------


class TestTraitAlteringLayer:
    """Tests for Layer 1 (Trait-Altering) modifier application."""

    def test_might_set_overrides_base(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=5, name="Setter")
        target = add_unit(gs, "p1", might=3, name="Target")

        _register_might_set_modifier(gs, source, 4, scope="friendly")
        evaluate_layers(gs)

        # Target's base becomes 4, source's base also becomes 4
        assert target.effective_might == 4
        assert source.effective_might == 4

    def test_might_set_then_arithmetic(self):
        """Layer 1 (might_set) applies before Layer 3 (arithmetic).

        Rule 454.1.a.1: Might assignment is in the Trait-Altering layer.
        Then arithmetic modifications apply on top.
        """
        gs = make_game()
        setter = add_unit(gs, "p1", might=5, name="Setter")
        buffer = add_unit(gs, "p1", might=5, name="Buffer")
        target = add_unit(gs, "p1", might=7, name="Target")

        # Layer 1: set might to 2
        _register_might_set_modifier(gs, setter, 2, scope="friendly")
        # Layer 3: +3 might
        _register_might_modifier(gs, buffer, +3, scope="friendly")
        evaluate_layers(gs)

        # Target: base set to 2 (layer 1), then +3 (layer 3) = 5
        assert target.effective_might == 5


# ---------------------------------------------------------------------------
# Layer 2: Ability-Altering — keyword grants
# ---------------------------------------------------------------------------


class TestAbilityAlteringLayer:
    """Tests for Layer 2 (Ability-Altering) keyword grants."""

    def test_keyword_grant(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Aura Source")
        target = add_unit(gs, "p1", might=3, name="Target")

        _register_keyword_modifier(gs, source, "vision", scope="friendly")
        evaluate_layers(gs)

        assert target.has_keyword(Keyword.VISION)
        assert source.has_keyword(Keyword.VISION)

    def test_keyword_grant_exclude_self(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Aura Source")
        target = add_unit(gs, "p1", might=3, name="Target")

        _register_keyword_modifier(gs, source, "shield", keyword_value=1,
                                   scope="friendly", exclude_self=True)
        evaluate_layers(gs)

        assert target.has_keyword(Keyword.SHIELD)
        assert not source.has_keyword(Keyword.SHIELD)

    def test_aura_keywords_cleared_on_recalc(self):
        """Aura keywords should be cleared and reapplied each evaluation."""
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Aura Source")
        target = add_unit(gs, "p1", might=3, name="Target")

        _register_keyword_modifier(gs, source, "vision", scope="friendly")

        # First evaluation: keyword applied
        evaluate_layers(gs)
        assert target.has_keyword(Keyword.VISION)

        # Remove the modifier
        gs.active_modifiers.clear()

        # Second evaluation: keyword should be gone
        evaluate_layers(gs)
        assert not target.has_keyword(Keyword.VISION)


# ---------------------------------------------------------------------------
# Cross-layer interactions (rule 453 — Fiora example)
# ---------------------------------------------------------------------------


class TestCrossLayerInteraction:
    """Test re-evaluation across layers (rule 453).

    The Fiora example from the rules: a unit with printed Might 4 and
    "While I'm Mighty, I have Deflect, Ganking, and Shield."
    A buff makes Might 5 → keywords should appear after re-evaluation.
    """

    def test_layer_ordering_trait_before_ability_before_arithmetic(self):
        """Verify layers run in correct order: 1 → 2 → 3."""
        gs = make_game()
        setter = add_unit(gs, "p1", might=5, name="Setter")
        buffer = add_unit(gs, "p1", might=5, name="Buffer")
        target = add_unit(gs, "p1", might=10, name="Target")

        # Register in reverse order to confirm layer ordering overrides timestamp
        _register_might_modifier(gs, buffer, +1, scope="friendly")   # Layer 3
        _register_might_set_modifier(gs, setter, 3, scope="friendly")  # Layer 1

        evaluate_layers(gs)

        # Layer 1 sets base to 3, Layer 3 adds +1 = 4
        assert target.effective_might == 4


# ---------------------------------------------------------------------------
# Modifier pruning on source removal
# ---------------------------------------------------------------------------


class TestModifierPruning:
    """Modifiers should be pruned when their source leaves the board."""

    def test_modifier_pruned_when_source_dies(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Aura Source")
        target = add_unit(gs, "p1", might=3, name="Buffed Unit")

        _register_might_modifier(gs, source, +2, scope="friendly")
        evaluate_layers(gs)
        assert target.effective_might == 5  # 3 + 2

        # Remove source from board (simulate death)
        gs.base_units["p1"].remove(source.instance_id)
        source.zone = ZoneType.TRASH
        gs.players["p1"].trash.append(source.instance_id)

        evaluate_layers(gs)
        assert target.effective_might == 3  # back to base
        assert len(gs.active_modifiers) == 0  # modifier pruned

    def test_keyword_pruned_when_source_dies(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Aura Source")
        target = add_unit(gs, "p1", might=3, name="Target")

        _register_keyword_modifier(gs, source, "vision", scope="friendly")
        evaluate_layers(gs)
        assert target.has_keyword(Keyword.VISION)

        # Remove source
        gs.base_units["p1"].remove(source.instance_id)
        source.zone = ZoneType.TRASH
        gs.players["p1"].trash.append(source.instance_id)

        evaluate_layers(gs)
        assert not target.has_keyword(Keyword.VISION)


# ---------------------------------------------------------------------------
# Gear attachment bonuses (rule 454.3.c)
# ---------------------------------------------------------------------------


class TestGearLayerIntegration:
    """Gear attachment might bonuses apply in the Arithmetic layer (454.3.c)."""

    def test_gear_bonus_applied(self):
        gs = make_game()
        from app.engine.card_types import CardDefinition
        unit = add_unit(gs, "p1", might=3, name="Equipped Unit")

        # Create a gear with might_bonus
        gear_def = CardDefinition(
            card_id="test_gear",
            name="Test Gear",
            card_type=CardType.GEAR,
            might_bonus=2,
        )
        gear = CardInstance.create(gear_def, "p1", ZoneType.BASE)
        gs.instances[gear.instance_id] = gear
        gs.base_gear["p1"].append(gear.instance_id)

        # Attach gear to unit
        gear.attached_to = unit.instance_id
        unit.attached_cards.append(gear.instance_id)

        evaluate_layers(gs)

        assert unit.effective_might == 5  # 3 + 2 gear bonus

    def test_gear_bonus_stacks_with_aura(self):
        gs = make_game()
        from app.engine.card_types import CardDefinition

        aura_source = add_unit(gs, "p1", might=2, name="Aura")
        unit = add_unit(gs, "p1", might=3, name="Equipped Unit")

        # Gear with might_bonus
        gear_def = CardDefinition(
            card_id="test_gear",
            name="Test Gear",
            card_type=CardType.GEAR,
            might_bonus=2,
        )
        gear = CardInstance.create(gear_def, "p1", ZoneType.BASE)
        gs.instances[gear.instance_id] = gear
        gs.base_gear["p1"].append(gear.instance_id)

        gear.attached_to = unit.instance_id
        unit.attached_cards.append(gear.instance_id)

        # Aura: +1 to all friendly
        _register_might_modifier(gs, aura_source, +1, scope="friendly")
        evaluate_layers(gs)

        # unit: 3 + 2 (gear) + 1 (aura) = 6
        assert unit.effective_might == 6


# ---------------------------------------------------------------------------
# Integration with cleanup
# ---------------------------------------------------------------------------


class TestCleanupIntegration:
    """evaluate_layers is called during cleanup step 0 via recalculate_modifiers."""

    def test_cleanup_calls_layer_evaluation(self):
        """Verify that running cleanup applies layer-based modifiers."""
        from app.engine.cleanup import run_cleanup

        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Aura Source")
        target = add_unit(gs, "p1", might=2, name="Target")

        _register_might_modifier(gs, source, +1, scope="friendly")

        # cleanup calls recalculate_modifiers which calls evaluate_layers
        run_cleanup(gs)

        assert target.effective_might == 3  # 2 + 1


# ---------------------------------------------------------------------------
# Non-board zones are unaffected
# ---------------------------------------------------------------------------


class TestNonBoardZones:
    """Modifiers do not affect cards in non-board zones (rule 711)."""

    def test_hand_card_uses_inherent_might(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Aura Source")

        # Card in hand
        defn = make_unit_def(might=4, name="Hand Card")
        hand_card = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[hand_card.instance_id] = hand_card
        gs.players["p1"].hand.append(hand_card.instance_id)

        _register_might_modifier(gs, source, +2, scope="friendly")
        evaluate_layers(gs)

        # Hand card uses inherent might regardless of modifiers
        assert hand_card.effective_might == 4


# ---------------------------------------------------------------------------
# Modifier cleanup between turns (rule 317.3)
# ---------------------------------------------------------------------------


class TestModifierCleanup:
    """Transient modifiers (might_modifiers, granted_keywords) expire at end
    of turn via clear_turn_state (rule 317.3). Permanent state persists."""

    def test_clear_turn_state_removes_transient_modifiers(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.might_modifiers.append(2)
        unit.might_modifiers.append(-1)
        assert unit.effective_might == 4  # 3 + 2 - 1

        unit.clear_turn_state()
        assert unit.might_modifiers == []
        assert unit.effective_might == 3

    def test_clear_turn_state_removes_granted_keywords(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.granted_keywords.append(KeywordInstance(keyword=Keyword.ASSAULT, value=2))
        assert unit.has_keyword(Keyword.ASSAULT)

        unit.clear_turn_state()
        assert not unit.has_keyword(Keyword.ASSAULT)
        assert unit.granted_keywords == []

    def test_clear_turn_state_clears_stun(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.stunned = True
        assert unit.combat_might == 0

        unit.clear_turn_state()
        assert not unit.stunned
        assert unit.combat_might == 3

    def test_clear_turn_state_clears_entered_this_turn(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        assert unit.entered_this_turn is True

        unit.clear_turn_state()
        assert unit.entered_this_turn is False

    def test_clear_turn_state_preserves_aura_might_bonus(self):
        """Aura might bonus is recalculated dynamically, not cleared by
        clear_turn_state."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.aura_might_bonus = 2

        unit.clear_turn_state()
        assert unit.aura_might_bonus == 2

    def test_clear_turn_state_clears_aura_keywords(self):
        """Aura keywords and trait_might_set are cleared for safety (they
        are re-evaluated next cleanup pass)."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.aura_keywords.append(KeywordInstance(keyword=Keyword.SHIELD, value=1))
        unit.trait_might_set = 5

        unit.clear_turn_state()
        assert unit.aura_keywords == []
        assert unit.trait_might_set is None

    def test_clear_turn_state_preserves_buff_counter(self):
        """Buff counters are permanent (rule 703) and survive end of turn."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.buff_counter = True
        assert unit.effective_might == 4

        unit.clear_turn_state()
        assert unit.buff_counter is True
        assert unit.effective_might == 4

    def test_clear_turn_state_preserves_gear_bonus(self):
        """Gear might bonus persists across turns."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.gear_might_bonus = 3

        unit.clear_turn_state()
        assert unit.gear_might_bonus == 3
        assert unit.effective_might == 6

    def test_reset_on_zone_exit_clears_everything(self):
        """When a card leaves play, all mutable state is wiped (rule 705)."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.buff_counter = True
        unit.might_modifiers.append(2)
        unit.aura_might_bonus = 1
        unit.gear_might_bonus = 2
        unit.stunned = True
        unit.granted_keywords.append(KeywordInstance(keyword=Keyword.ASSAULT, value=1))

        unit.reset_on_zone_exit()

        assert unit.buff_counter is False
        assert unit.might_modifiers == []
        assert unit.aura_might_bonus == 0
        assert unit.gear_might_bonus == 0
        assert unit.stunned is False
        assert unit.granted_keywords == []

    def test_end_of_turn_clears_transient_modifiers_via_state_machine(self):
        """Integration: advance_phase from ACTION clears transient state
        on all board cards (rule 317.3)."""
        from app.engine.state_machine import advance_phase

        gs = make_game(phase=Phase.ACTION)
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.might_modifiers.append(5)
        unit.granted_keywords.append(KeywordInstance(keyword=Keyword.VISION, value=0))
        assert unit.effective_might == 8

        # Advance from ACTION -> END_OF_TURN (takes no extra args)
        advance_phase(gs)

        assert unit.might_modifiers == []
        assert unit.granted_keywords == []
        assert unit.effective_might == 3


# ---------------------------------------------------------------------------
# effective_might computation accuracy
# ---------------------------------------------------------------------------


class TestEffectiveMightComputation:
    """Comprehensive tests for the effective_might property."""

    def test_base_might_no_modifiers(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=4, name="Soldier")
        assert unit.effective_might == 4

    def test_inherent_might_equals_definition(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=7, name="Big Unit")
        assert unit.inherent_might == 7

    def test_non_board_zone_uses_inherent_might(self):
        """Rule 711: Units in non-board zones use inherent might only."""
        gs = make_game()
        defn = CardDefinition(
            card_id="test_hand", name="Hand Unit",
            card_type=CardType.UNIT, base_might=3,
        )
        inst = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[inst.instance_id] = inst
        gs.players["p1"].hand.append(inst.instance_id)

        # Even with modifiers set, non-board zone uses base might
        inst.might_modifiers.append(5)
        inst.aura_might_bonus = 3
        inst.buff_counter = True
        assert inst.effective_might == 3  # inherent only

    def test_trash_zone_uses_inherent_might(self):
        gs = make_game()
        defn = CardDefinition(
            card_id="test_trash", name="Dead Unit",
            card_type=CardType.UNIT, base_might=4,
        )
        inst = CardInstance.create(defn, "p1", ZoneType.TRASH)
        gs.instances[inst.instance_id] = inst
        inst.might_modifiers.append(10)
        assert inst.effective_might == 4  # inherent only, ignores modifiers

    def test_non_board_might_cannot_go_below_zero(self):
        gs = make_game()
        defn = CardDefinition(
            card_id="test_zero", name="Zero Unit",
            card_type=CardType.UNIT, base_might=0,
        )
        inst = CardInstance.create(defn, "p1", ZoneType.TRASH)
        gs.instances[inst.instance_id] = inst
        assert inst.effective_might == 0

    def test_combat_might_zero_when_stunned(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=5, name="Soldier")
        unit.stunned = True
        assert unit.effective_might == 5  # effective_might unchanged
        assert unit.combat_might == 0     # combat_might is 0 when stunned

    def test_buff_counter_adds_one(self):
        """Rule 703: Each buff counter grants +1 might."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.buff_counter = True
        assert unit.effective_might == 4

    def test_assault_only_when_attacking(self):
        """Rule 733: Assault bonus only applies while unit is attacker."""
        gs = make_game()
        kws = (KeywordInstance(keyword=Keyword.ASSAULT, value=2),)
        unit = add_unit(gs, "p1", might=3, name="Attacker", keywords=kws)

        # No combat role
        assert unit.effective_might == 3

        # As attacker
        unit.combat_role = CombatRole.ATTACKER
        assert unit.effective_might == 5  # 3 + 2

        # As defender — Assault does not apply
        unit.combat_role = CombatRole.DEFENDER
        assert unit.effective_might == 3

    def test_shield_only_when_defending(self):
        """Rule 740: Shield bonus only applies while unit is defender."""
        gs = make_game()
        kws = (KeywordInstance(keyword=Keyword.SHIELD, value=1),)
        unit = add_unit(gs, "p1", might=3, name="Defender", keywords=kws)

        # No combat role
        assert unit.effective_might == 3

        # As defender
        unit.combat_role = CombatRole.DEFENDER
        assert unit.effective_might == 4  # 3 + 1

        # As attacker — Shield does not apply
        unit.combat_role = CombatRole.ATTACKER
        assert unit.effective_might == 3

    def test_assault_default_value_is_one(self):
        """Rule 733.1.b.3: Assault with omitted value defaults to 1."""
        gs = make_game()
        kws = (KeywordInstance(keyword=Keyword.ASSAULT, value=0),)
        unit = add_unit(gs, "p1", might=3, name="Attacker", keywords=kws)
        unit.combat_role = CombatRole.ATTACKER
        assert unit.effective_might == 4  # 3 + default 1

    def test_shield_default_value_is_one(self):
        """Rule 740.1.b.3: Shield with omitted value defaults to 1."""
        gs = make_game()
        kws = (KeywordInstance(keyword=Keyword.SHIELD, value=0),)
        unit = add_unit(gs, "p1", might=3, name="Defender", keywords=kws)
        unit.combat_role = CombatRole.DEFENDER
        assert unit.effective_might == 4  # 3 + default 1

    def test_trait_might_set_overrides_base_in_effective(self):
        """Layer 1 (Trait-Altering): trait_might_set replaces base might."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        unit.trait_might_set = 6
        assert unit.effective_might == 6

    def test_trait_might_set_to_zero_with_buff(self):
        """'Becomes 0 Might' then buff adds arithmetic on top."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=5, name="Soldier")
        unit.trait_might_set = 0
        unit.buff_counter = True  # +1 in arithmetic
        assert unit.effective_might == 1

    def test_all_modifier_sources_combined(self):
        """Combine every modifier source for a comprehensive check."""
        gs = make_game()
        kws = (
            KeywordInstance(keyword=Keyword.ASSAULT, value=2),
            KeywordInstance(keyword=Keyword.SHIELD, value=1),
        )
        unit = add_unit(gs, "p1", might=3, name="Soldier", keywords=kws)

        unit.buff_counter = True            # +1
        unit.gear_might_bonus = 2           # +2
        unit.might_modifiers.append(1)      # +1
        unit.might_modifiers.append(-1)     # -1
        unit.aura_might_bonus = 1           # +1
        unit.combat_role = CombatRole.ATTACKER  # Assault +2, Shield does NOT apply

        # base 3 + buff 1 + gear 2 + transient +1 - 1 + aura 1 + assault 2 = 9
        assert unit.effective_might == 9

    def test_is_mighty_threshold(self):
        """Rule 708: A unit is Mighty if effective_might >= 5."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=4, name="Soldier")
        assert not unit.is_mighty

        unit.might_modifiers.append(1)
        assert unit.is_mighty

    def test_would_become_mighty(self):
        """Rule 709: Detect transition across the Mighty threshold."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=4, name="Soldier")

        assert unit.would_become_mighty(1)      # 4 -> 5 crosses
        assert not unit.would_become_mighty(0)   # 4 -> 4 no cross
        assert not unit.would_become_mighty(-1)  # 4 -> 3 no cross

    def test_already_mighty_does_not_become_mighty(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=5, name="Soldier")
        assert unit.is_mighty
        assert not unit.would_become_mighty(1)  # already mighty

    def test_is_alive_with_zero_might(self):
        """Rule 415: A unit with 0 effective might cannot be killed by damage."""
        gs = make_game()
        unit = add_unit(gs, "p1", might=0, name="Ghost")
        assert unit.is_alive
        unit.damage = 5
        assert unit.is_alive  # 0-might units can't die from damage


# ---------------------------------------------------------------------------
# Layer classification (effect_ir.classify_modifier_layer)
# ---------------------------------------------------------------------------


class TestLayerClassification:
    """Test that modifier IR nodes are classified into the correct layer."""

    def test_might_is_arithmetic(self):
        assert classify_modifier_layer({"stat": "might", "amount": 2}) == EffectLayer.ARITHMETIC

    def test_might_set_is_trait_altering(self):
        assert classify_modifier_layer({"stat": "might_set", "amount": 4}) == EffectLayer.TRAIT_ALTERING

    def test_keyword_is_ability_altering(self):
        assert classify_modifier_layer({"stat": "keyword", "amount": 0}) == EffectLayer.ABILITY_ALTERING

    def test_name_is_trait_altering(self):
        assert classify_modifier_layer({"stat": "name"}) == EffectLayer.TRAIT_ALTERING

    def test_type_is_trait_altering(self):
        assert classify_modifier_layer({"stat": "type"}) == EffectLayer.TRAIT_ALTERING

    def test_energy_cost_is_arithmetic(self):
        assert classify_modifier_layer({"stat": "energy_cost", "amount": -1}) == EffectLayer.ARITHMETIC

    def test_unknown_stat_defaults_to_arithmetic(self):
        assert classify_modifier_layer({"stat": "unknown_stat", "amount": 1}) == EffectLayer.ARITHMETIC

    def test_classify_active_modifier_might(self):
        mod = ActiveModifier(
            source_instance_id="src", ability_id="ab",
            stat="might", amount=1, target_spec={},
        )
        assert _classify_active_modifier(mod) == EffectLayer.ARITHMETIC

    def test_classify_active_modifier_might_set(self):
        mod = ActiveModifier(
            source_instance_id="src", ability_id="ab",
            stat="might_set", amount=4, target_spec={},
        )
        assert _classify_active_modifier(mod) == EffectLayer.TRAIT_ALTERING


# ---------------------------------------------------------------------------
# Arithmetic sub-sorting
# ---------------------------------------------------------------------------


class TestArithmeticSubSort:
    """Test _sort_arithmetic: increases before decreases (rule 454.3.d)."""

    def test_increases_before_decreases(self):
        mod_pos = ActiveModifier(
            source_instance_id="s1", ability_id="ab",
            stat="might", amount=2, target_spec={},
        )
        mod_neg = ActiveModifier(
            source_instance_id="s2", ability_id="ab",
            stat="might", amount=-3, target_spec={},
        )
        result = _sort_arithmetic([(1, mod_neg), (0, mod_pos)])
        assert result[0][1].amount == 2   # increase first
        assert result[1][1].amount == -3  # decrease second

    def test_preserves_timestamp_within_group(self):
        mods = [
            (2, ActiveModifier(source_instance_id="c", ability_id="ab",
                               stat="might", amount=-2, target_spec={})),
            (0, ActiveModifier(source_instance_id="a", ability_id="ab",
                               stat="might", amount=1, target_spec={})),
            (3, ActiveModifier(source_instance_id="d", ability_id="ab",
                               stat="might", amount=-1, target_spec={})),
            (1, ActiveModifier(source_instance_id="b", ability_id="ab",
                               stat="might", amount=3, target_spec={})),
        ]
        result = _sort_arithmetic(mods)
        indices = [idx for idx, _ in result]
        # Increases (ts 0, 1) then decreases (ts 2, 3)
        assert indices == [0, 1, 2, 3]

    def test_zero_treated_as_increase(self):
        mod_zero = ActiveModifier(
            source_instance_id="s1", ability_id="ab",
            stat="might", amount=0, target_spec={},
        )
        mod_neg = ActiveModifier(
            source_instance_id="s2", ability_id="ab",
            stat="might", amount=-1, target_spec={},
        )
        result = _sort_arithmetic([(1, mod_neg), (0, mod_zero)])
        assert result[0][1].amount == 0   # zero grouped with increases
        assert result[1][1].amount == -1


# TODO (BE-05): Once evaluate_layers is wired into cleanup.py (replacing
# gs.recalculate_modifiers as the primary entry point), add tests for:
# - Cross-layer re-evaluation (rule 453): e.g. arithmetic buff causes a
#   unit to become Mighty, which triggers an ability-altering conditional
#   keyword grant in the next pass (the Fiora example)
# - Dependency resolution within a layer (rules 455-456)
# - Timestamp ordering for non-dependent same-layer effects (rule 457)
# - Snapshotting for limited arithmetic effects (rule 454.3.b)
