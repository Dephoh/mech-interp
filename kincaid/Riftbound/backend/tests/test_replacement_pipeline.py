"""Integration tests for the full replacement effects pipeline (BE-04).

Verifies the complete lifecycle:
  1. Card with replacement ability enters play
  2. Resolver registers ActiveReplacement on GameState
  3. When the watched event fires, check_replacements / check_primitive_replacement
     intercepts it
  4. Replacement effect resolves instead of the original

Also tests:
  - Battlefield cards with passive replacement abilities register at game start
  - Replacements are unregistered when the source leaves the board
  - Only the first matching replacement applies (rule 366)
  - Replacement target scoping (friendly-only, enemy-only)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.engine.card_types import (
    AbilityDefinition,
    CardDefinition,
    CardInstance,
)
from app.engine.chain import (
    _resolve_passive_abilities,
    push_card_to_chain,
    register_board_passives,
)
from app.engine.effect_ir import (
    REPLACEMENT,
    TargetSpec,
    deal_damage,
    heal,
    replacement,
    sequence,
)
from app.engine.effect_resolver import resolve_effect_ir
from app.engine.enums import (
    AbilityType,
    CardType,
    Phase,
    ZoneType,
)
from app.engine.game_state import (
    ActiveReplacement,
    BattlefieldState,
    GameState,
)
from app.engine.trigger_system import (
    GameEvent,
    check_primitive_replacement,
    check_replacements,
)

from helpers import add_unit, make_game, make_unit_def


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit_with_replacement(
    name: str = "Shield Bearer",
    might: int = 3,
    watch_event: str = "deal_damage",
    alternative_ir: dict | None = None,
    target_scope: str = "friendly",
    condition: dict | None = None,
) -> CardDefinition:
    """Create a unit definition that carries a passive replacement ability."""
    if alternative_ir is None:
        alternative_ir = heal(amount="all")
    rep_ir = replacement(
        watch_event=watch_event,
        condition=condition or {"cond_type": "always"},
        alternative_ir=alternative_ir,
        target=TargetSpec(scope=target_scope).to_dict(),
    )
    ability = AbilityDefinition(
        ability_id=f"{name.lower().replace(' ', '_')}_rep",
        ability_type=AbilityType.PASSIVE,
        effect_ir=rep_ir,
        text=f"If a {target_scope} unit would take damage, heal it instead.",
    )
    return CardDefinition(
        card_id=f"test_{name.lower().replace(' ', '_')}",
        name=name,
        card_type=CardType.UNIT,
        base_might=might,
        cost_energy=2,
        abilities=(ability,),
    )


def _make_battlefield_with_replacement(
    name: str = "Sacred Ground",
    watch_event: str = "kill",
    alternative_ir: dict | None = None,
) -> CardDefinition:
    """Create a battlefield definition with a passive replacement ability."""
    if alternative_ir is None:
        alternative_ir = {"type": "return_to_hand"}
    rep_ir = replacement(
        watch_event=watch_event,
        condition={"cond_type": "always"},
        alternative_ir=alternative_ir,
        target=TargetSpec(scope="any").to_dict(),
    )
    ability = AbilityDefinition(
        ability_id=f"{name.lower().replace(' ', '_')}_rep",
        ability_type=AbilityType.PASSIVE,
        effect_ir=rep_ir,
        text=f"Units here cannot be killed; return them to hand instead.",
    )
    return CardDefinition(
        card_id=f"test_{name.lower().replace(' ', '_')}",
        name=name,
        card_type=CardType.BATTLEFIELD,
        abilities=(ability,),
    )


# ---------------------------------------------------------------------------
# Unit tests: _resolve_passive_abilities
# ---------------------------------------------------------------------------


class TestResolvePassiveAbilities:
    """Verify _resolve_passive_abilities registers replacement effects."""

    def test_passive_replacement_registered_on_resolve(self):
        gs = make_game()
        defn = _make_unit_with_replacement(name="Guardian")
        card = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[card.instance_id] = card
        gs.base_units["p1"].append(card.instance_id)

        assert len(gs.active_replacements) == 0
        logs = _resolve_passive_abilities(gs, card)
        assert len(gs.active_replacements) == 1
        assert gs.active_replacements[0].watch_event == "deal_damage"
        assert gs.active_replacements[0].source_instance_id == card.instance_id
        assert any("replacement effect active" in l for l in logs)

    def test_no_double_registration(self):
        gs = make_game()
        defn = _make_unit_with_replacement(name="Guardian")
        card = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[card.instance_id] = card
        gs.base_units["p1"].append(card.instance_id)

        _resolve_passive_abilities(gs, card)
        _resolve_passive_abilities(gs, card)
        # Should still be 1 due to dedup in _resolve_replacement
        assert len(gs.active_replacements) == 1

    def test_non_passive_abilities_ignored(self):
        """Only passive abilities should be processed."""
        gs = make_game()
        # A unit with a triggered ability (not passive)
        triggered_ab = AbilityDefinition(
            ability_id="trigger_ab",
            ability_type=AbilityType.TRIGGERED,
            trigger_condition="on_play",
            effect_ir=deal_damage(2),
        )
        defn = CardDefinition(
            card_id="test_triggered_only",
            name="Triggered Only",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(triggered_ab,),
        )
        card = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[card.instance_id] = card
        gs.base_units["p1"].append(card.instance_id)

        logs = _resolve_passive_abilities(gs, card)
        assert len(gs.active_replacements) == 0
        assert len(logs) == 0


# ---------------------------------------------------------------------------
# Integration test: permanent enters play -> replacement registers
# ---------------------------------------------------------------------------


class TestReplacementRegistrationOnEntry:
    """Card enters board via push_card_to_chain -> passive replacement registers."""

    def test_unit_entering_board_registers_replacement(self):
        gs = make_game()
        gs.players["p1"].rune_pool.energy = 10

        defn = _make_unit_with_replacement(name="Shield Bearer")
        card = CardInstance.create(defn, "p1", ZoneType.HAND)
        card.location_id = "p1"
        gs.instances[card.instance_id] = card
        gs.players["p1"].hand.append(card.instance_id)

        # Push to chain (simulates playing the card)
        push_card_to_chain(gs, card, "p1", destination={"zone": "base"})

        assert len(gs.active_replacements) == 1
        rep = gs.active_replacements[0]
        assert rep.watch_event == "deal_damage"
        assert rep.source_instance_id == card.instance_id

    def test_unit_leaving_board_unregisters_replacement(self):
        gs = make_game()
        defn = _make_unit_with_replacement(name="Guardian")
        card = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[card.instance_id] = card
        gs.players["p1"].hand.append(card.instance_id)

        push_card_to_chain(gs, card, "p1", destination={"zone": "base"})
        assert len(gs.active_replacements) == 1

        # Simulate leaving the board
        gs.unregister_card(card.instance_id)
        assert len(gs.active_replacements) == 0


# ---------------------------------------------------------------------------
# Integration test: battlefield passive replacement registered at game start
# ---------------------------------------------------------------------------


class TestBattlefieldPassiveRegistration:
    """Battlefield cards with replacement passives register via register_board_passives."""

    def test_register_board_passives_registers_battlefield_replacement(self):
        gs = make_game(num_battlefields=0)

        # Manually add a battlefield with a replacement ability
        bf_def = _make_battlefield_with_replacement(name="Sacred Ground")
        bf_inst = CardInstance.create(bf_def, "neutral", ZoneType.BATTLEFIELD)
        gs.instances[bf_inst.instance_id] = bf_inst
        gs.battlefields[bf_inst.instance_id] = BattlefieldState(
            battlefield_id=bf_inst.instance_id,
            card_instance_id=bf_inst.instance_id,
        )

        assert len(gs.active_replacements) == 0
        logs = register_board_passives(gs)
        assert len(gs.active_replacements) == 1
        assert gs.active_replacements[0].watch_event == "kill"
        assert any("replacement effect active" in l for l in logs)

    def test_register_board_passives_skips_already_registered(self):
        gs = make_game(num_battlefields=0)

        bf_def = _make_battlefield_with_replacement(name="Sacred Ground")
        bf_inst = CardInstance.create(bf_def, "neutral", ZoneType.BATTLEFIELD)
        gs.instances[bf_inst.instance_id] = bf_inst
        gs.battlefields[bf_inst.instance_id] = BattlefieldState(
            battlefield_id=bf_inst.instance_id,
            card_instance_id=bf_inst.instance_id,
        )

        register_board_passives(gs)
        assert len(gs.active_replacements) == 1

        # Call again — should skip since already registered
        register_board_passives(gs)
        assert len(gs.active_replacements) == 1


# ---------------------------------------------------------------------------
# End-to-end: replacement intercepts a primitive
# ---------------------------------------------------------------------------


class TestReplacementInterceptsPrimitive:
    """Full pipeline: card enters -> replacement registers -> kill intercepted."""

    def test_kill_intercepted_by_replacement(self):
        """Unit with 'if friendly would die, heal instead' replacement.

        Steps:
          1. Shield Bearer enters play (has passive replacement watching 'kill')
          2. Protectee is on the board
          3. Attacker tries to kill Protectee
          4. Replacement intercepts: heals Protectee instead of killing
        """
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]

        # Create Shield Bearer with replacement: intercept kill -> heal
        defn = _make_unit_with_replacement(
            name="Shield Bearer",
            watch_event="kill",
            alternative_ir=heal(amount="all"),
            target_scope="friendly",
        )
        shield_bearer = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[shield_bearer.instance_id] = shield_bearer
        gs.players["p1"].hand.append(shield_bearer.instance_id)

        # Place Shield Bearer on the board (this should register the replacement)
        push_card_to_chain(gs, shield_bearer, "p1", destination={"zone": "base"})
        assert len(gs.active_replacements) == 1

        # Add a protectee that we'll try to kill
        protectee = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=3,
                             name="Protectee", bf_id=bf_id)
        protectee.damage = 1  # give some damage so heal has visible effect

        # Add attacker
        attacker = add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=5,
                            name="Attacker", bf_id=bf_id)

        # Try to kill protectee — replacement should intercept
        kill_ir = {"type": "kill"}
        logs = resolve_effect_ir(kill_ir, attacker, gs, [protectee.instance_id])

        # Protectee should NOT be in trash — replacement healed instead of killing
        assert protectee.damage == 0  # healed
        # The replacement should have fired (check logs)
        assert any("heal" in l.lower() for l in logs)

    def test_deal_damage_intercepted_by_replacement(self):
        """Unit with 'if friendly would take damage, heal instead' replacement.

        Steps:
          1. Guardian enters play (has passive replacement watching 'deal_damage')
          2. Ally is on the board
          3. Enemy tries to deal damage to Ally
          4. Replacement intercepts: heals instead of dealing damage
        """
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]

        # Guardian with replacement: intercept deal_damage -> heal
        defn = _make_unit_with_replacement(
            name="Guardian",
            watch_event="deal_damage",
            alternative_ir=heal(amount="all"),
            target_scope="friendly",
        )
        guardian = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[guardian.instance_id] = guardian
        gs.players["p1"].hand.append(guardian.instance_id)

        push_card_to_chain(gs, guardian, "p1", destination={"zone": "base"})
        assert len(gs.active_replacements) == 1

        # Ally to protect
        ally = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=5,
                        name="Ally", bf_id=bf_id)

        # Enemy source
        enemy = add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=4,
                         name="Enemy", bf_id=bf_id)

        # Deal damage to ally — replacement should intercept
        damage_ir = deal_damage(3)
        logs = resolve_effect_ir(damage_ir, enemy, gs, [ally.instance_id])

        # Ally should not have taken damage
        assert ally.damage == 0

    def test_replacement_does_not_intercept_enemy_units(self):
        """Replacement with scope='friendly' should not intercept enemy damage."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]

        defn = _make_unit_with_replacement(
            name="Guardian",
            watch_event="deal_damage",
            alternative_ir=heal(amount="all"),
            target_scope="friendly",
        )
        guardian = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[guardian.instance_id] = guardian
        gs.players["p1"].hand.append(guardian.instance_id)

        push_card_to_chain(gs, guardian, "p1", destination={"zone": "base"})

        # Enemy unit — should NOT be protected
        enemy_unit = add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=5,
                              name="Enemy Target", bf_id=bf_id)

        # Source dealing damage to the enemy
        source = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=4,
                          name="Source", bf_id=bf_id)

        damage_ir = deal_damage(3)
        resolve_effect_ir(damage_ir, source, gs, [enemy_unit.instance_id])

        # Enemy SHOULD have taken damage (replacement only protects friendlies)
        assert enemy_unit.damage == 3


# ---------------------------------------------------------------------------
# Edge case: multiple replacements (rule 366 — only first applies)
# ---------------------------------------------------------------------------


class TestMultipleReplacements:
    """When multiple replacements match, only the first eligible one fires."""

    def test_only_first_replacement_applies(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]

        # Two guardians, each with a replacement watching deal_damage
        for i in range(2):
            defn = _make_unit_with_replacement(
                name=f"Guardian {i}",
                watch_event="deal_damage",
                alternative_ir=heal(amount="all"),
                target_scope="friendly",
            )
            card = CardInstance.create(defn, "p1", ZoneType.BASE)
            gs.instances[card.instance_id] = card
            gs.base_units["p1"].append(card.instance_id)
            _resolve_passive_abilities(gs, card)

        assert len(gs.active_replacements) == 2

        # Create a target and deal damage
        target = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=5,
                          name="Target", bf_id=bf_id)
        attacker = add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=4,
                            name="Attacker", bf_id=bf_id)

        damage_ir = deal_damage(3)
        resolve_effect_ir(damage_ir, attacker, gs, [target.instance_id])

        # Target should not have taken damage (first replacement intercepted)
        assert target.damage == 0


# ---------------------------------------------------------------------------
# check_replacements (event-level interception)
# ---------------------------------------------------------------------------


class TestCheckReplacements:
    """Test the event-level check_replacements function."""

    def test_check_replacements_returns_true_on_match(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]

        defn = _make_unit_with_replacement(
            name="Interceptor",
            watch_event="unit_entered",
            alternative_ir={"type": "sequence", "steps": []},
            target_scope="any",
        )
        card = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[card.instance_id] = card
        gs.base_units["p1"].append(card.instance_id)
        _resolve_passive_abilities(gs, card)

        # Create a target unit for the context
        target = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Target")

        result = check_replacements(
            gs,
            GameEvent.UNIT_ENTERED,
            {"card_id": target.instance_id},
        )
        assert result is True

    def test_check_replacements_returns_false_on_no_match(self):
        gs = make_game()

        defn = _make_unit_with_replacement(
            name="Interceptor",
            watch_event="unit_entered",
            target_scope="any",
        )
        card = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[card.instance_id] = card
        gs.base_units["p1"].append(card.instance_id)
        _resolve_passive_abilities(gs, card)

        # Fire a DIFFERENT event — should not match
        result = check_replacements(
            gs,
            GameEvent.TURN_START,
            {},
        )
        assert result is False


# ---------------------------------------------------------------------------
# check_primitive_replacement (primitive-level interception)
# ---------------------------------------------------------------------------


class TestCheckPrimitiveReplacement:
    """Test the primitive-level check_primitive_replacement function."""

    def test_primitive_replacement_returns_alt_ir(self):
        gs = make_game()

        alt_ir = heal(amount="all")
        defn = _make_unit_with_replacement(
            name="Guardian",
            watch_event="deal_damage",
            alternative_ir=alt_ir,
            target_scope="friendly",
        )
        guardian = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[guardian.instance_id] = guardian
        gs.base_units["p1"].append(guardian.instance_id)
        _resolve_passive_abilities(gs, guardian)

        target = add_unit(gs, "p1", ZoneType.BASE, might=5, name="Protectee")

        result = check_primitive_replacement(
            gs, "deal_damage", guardian, [target.instance_id]
        )
        assert result is not None
        assert result["type"] == "heal"

    def test_primitive_replacement_returns_none_when_no_match(self):
        gs = make_game()

        defn = _make_unit_with_replacement(
            name="Guardian",
            watch_event="kill",  # watches kill, not deal_damage
            target_scope="friendly",
        )
        guardian = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[guardian.instance_id] = guardian
        gs.base_units["p1"].append(guardian.instance_id)
        _resolve_passive_abilities(gs, guardian)

        target = add_unit(gs, "p1", ZoneType.BASE, might=5, name="Target")

        # Ask about deal_damage — guardian watches kill, so no match
        result = check_primitive_replacement(
            gs, "deal_damage", guardian, [target.instance_id]
        )
        assert result is None


# ---------------------------------------------------------------------------
# Conditional replacement (replacement wrapped in a conditional)
# ---------------------------------------------------------------------------


class TestConditionalReplacement:
    """Replacement wrapped in a 'conditional' node should still register."""

    def test_conditional_wrapper_registers_inner_replacement(self):
        gs = make_game()

        # A conditional node that wraps a replacement
        inner_rep = replacement(
            watch_event="deal_damage",
            condition={"cond_type": "always"},
            alternative_ir={"type": "sequence", "steps": []},
            target=TargetSpec(scope="friendly").to_dict(),
        )
        conditional_ir = {
            "type": "conditional",
            "condition": {"cond_type": "always"},
            "then": inner_rep,
        }

        ability = AbilityDefinition(
            ability_id="cond_rep_ab",
            ability_type=AbilityType.PASSIVE,
            effect_ir=conditional_ir,
        )
        defn = CardDefinition(
            card_id="test_cond_rep",
            name="Conditional Replacer",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(ability,),
        )
        card = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[card.instance_id] = card
        gs.base_units["p1"].append(card.instance_id)

        logs = _resolve_passive_abilities(gs, card)
        # The conditional evaluates to true -> the inner replacement registers
        assert len(gs.active_replacements) == 1
        assert gs.active_replacements[0].watch_event == "deal_damage"
