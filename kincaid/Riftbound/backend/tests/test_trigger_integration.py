"""Integration tests for the trigger system wired into the engine.

Verifies that fire_event() calls in chain, cleanup, combat, scoring,
state_machine, and action_executor correctly trigger card abilities.
"""

from __future__ import annotations

import pytest
from helpers import add_unit, make_game, make_unit_def

from app.engine.card_types import (
    AbilityDefinition,
    CardDefinition,
    CardInstance,
    KeywordInstance,
)
from app.engine.chain import push_card_to_chain, resolve_top_item
from app.engine.cleanup import run_cleanup
from app.engine.combat import start_combat
from app.engine.effect_ir import deal_damage, draw_cards, give_might, sequence
from app.engine.enums import (
    CardType,
    CombatRole,
    ControlStatus,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.game_state import BattlefieldState, _draw_cards
from app.engine.scoring import score_conquer, score_hold
from app.engine.state_machine import _execute_awaken, _execute_end_of_turn
from app.engine.trigger_system import GameEvent, fire_event


# ---------------------------------------------------------------------------
# Play triggers (chain.py → fire_event UNIT_PLAYED)
# ---------------------------------------------------------------------------


class TestPlayTriggers:
    def test_on_play_triggers_via_fire_event(self):
        """Unit with on_play draw effect triggers when played through chain."""
        gs = make_game()

        # Put a card in p1's deck for the draw effect to find
        deck_def = make_unit_def(name="Deck Card")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        # Unit with "When you play me, draw 1"
        unit_def = CardDefinition(
            card_id="play_draw",
            name="Play Drawer",
            card_type=CardType.UNIT,
            base_might=2,
            abilities=(
                AbilityDefinition(
                    ability_id="draw_on_play",
                    ability_type="triggered",
                    trigger_condition="on_play",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        unit = CardInstance.create(unit_def, "p1", ZoneType.HAND)
        gs.instances[unit.instance_id] = unit

        # Play the unit through the chain
        push_card_to_chain(gs, unit, "p1")

        # Unit should be on board
        assert unit.zone == ZoneType.BASE
        # Play trigger should be on chain
        assert not gs.chain.is_empty

        # Resolve the play effect
        logs = resolve_top_item(gs)
        assert deck_card.instance_id in gs.players["p1"].hand

    def test_observer_when_you_play_a_unit(self):
        """Board observer triggers 'When you play a unit' for ally plays."""
        gs = make_game()

        # Observer unit already on board with "When you play a unit, draw 1"
        observer_def = CardDefinition(
            card_id="observer",
            name="Observer",
            card_type=CardType.UNIT,
            base_might=2,
            abilities=(
                AbilityDefinition(
                    ability_id="obs_on_unit_played",
                    ability_type="triggered",
                    trigger_condition="on_unit_played",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        observer = CardInstance.create(observer_def, "p1", ZoneType.BASE)
        gs.instances[observer.instance_id] = observer
        gs.base_units["p1"].append(observer.instance_id)

        # Deck card for the draw
        deck_def = make_unit_def(name="Draw Me")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        # Play a different unit
        new_unit_def = CardDefinition(
            card_id="new_unit", name="New Unit", card_type=CardType.UNIT, base_might=3,
        )
        new_unit = CardInstance.create(new_unit_def, "p1", ZoneType.HAND)
        gs.instances[new_unit.instance_id] = new_unit

        push_card_to_chain(gs, new_unit, "p1")

        # Observer's trigger should be on chain
        assert not gs.chain.is_empty

        # Resolve the trigger
        logs = resolve_top_item(gs)
        assert deck_card.instance_id in gs.players["p1"].hand

    def test_opponent_unit_play_doesnt_trigger_observer(self):
        """Observer's 'When you play a unit' should NOT fire for opponent plays."""
        gs = make_game()

        observer_def = CardDefinition(
            card_id="observer",
            name="Observer",
            card_type=CardType.UNIT,
            base_might=2,
            abilities=(
                AbilityDefinition(
                    ability_id="obs_on_unit_played",
                    ability_type="triggered",
                    trigger_condition="on_unit_played",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        observer = CardInstance.create(observer_def, "p1", ZoneType.BASE)
        gs.instances[observer.instance_id] = observer
        gs.base_units["p1"].append(observer.instance_id)

        # Opponent plays a unit
        opp_unit_def = CardDefinition(
            card_id="opp_unit", name="Opp Unit", card_type=CardType.UNIT, base_might=2,
        )
        opp_unit = CardInstance.create(opp_unit_def, "p2", ZoneType.HAND)
        gs.instances[opp_unit.instance_id] = opp_unit

        push_card_to_chain(gs, opp_unit, "p2")

        # Chain should be empty — observer's trigger didn't fire for opponent
        assert gs.chain.is_empty


# ---------------------------------------------------------------------------
# Spell triggers (chain.py → fire_event SPELL_PLAYED)
# ---------------------------------------------------------------------------


class TestSpellTriggers:
    def test_when_you_play_a_spell(self):
        """Board object with 'When you play a spell' triggers when spell is played."""
        gs = make_game()

        # Observer with "When you play a spell, draw 1"
        observer_def = CardDefinition(
            card_id="spell_observer",
            name="Spell Observer",
            card_type=CardType.UNIT,
            base_might=2,
            abilities=(
                AbilityDefinition(
                    ability_id="on_spell",
                    ability_type="triggered",
                    trigger_condition="on_spell_played",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        observer = CardInstance.create(observer_def, "p1", ZoneType.BASE)
        gs.instances[observer.instance_id] = observer
        gs.base_units["p1"].append(observer.instance_id)

        # Deck card
        deck_def = make_unit_def(name="Drawn")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        # Play a spell
        spell_def = CardDefinition(
            card_id="test_spell", name="Bolt", card_type=CardType.SPELL,
            abilities=(
                AbilityDefinition(
                    ability_id="bolt_effect",
                    ability_type="activated",
                    effect_ir=deal_damage(3),
                ),
            ),
        )
        spell = CardInstance.create(spell_def, "p1", ZoneType.HAND)
        gs.instances[spell.instance_id] = spell

        push_card_to_chain(gs, spell, "p1")

        # Chain should have: [spell (bottom), observer trigger (top)]
        assert len(gs.chain.stack) == 2

        # Resolve top = observer trigger (LIFO)
        logs = resolve_top_item(gs)
        assert deck_card.instance_id in gs.players["p1"].hand

        # Remaining = spell itself
        assert len(gs.chain.stack) == 1


# ---------------------------------------------------------------------------
# Death triggers (cleanup.py → fire_event UNIT_DIED)
# ---------------------------------------------------------------------------


class TestDeathTriggers:
    def test_deathknell_fires_on_death(self):
        """Unit with on_death trigger fires when killed."""
        gs = make_game()

        # Deck card for the deathknell draw
        deck_def = make_unit_def(name="Drawn From Death")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        # Unit with on_death draw
        dk_def = CardDefinition(
            card_id="deathknell_unit",
            name="Dying Drawer",
            card_type=CardType.UNIT,
            base_might=2,
            abilities=(
                AbilityDefinition(
                    ability_id="dk_draw",
                    ability_type="triggered",
                    trigger_condition="on_death",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        dk_unit = CardInstance.create(dk_def, "p1", ZoneType.BASE)
        dk_unit.damage = 3  # lethal (might=2)
        gs.instances[dk_unit.instance_id] = dk_unit
        gs.base_units["p1"].append(dk_unit.instance_id)

        # Run cleanup — should kill the unit and fire death trigger
        logs = run_cleanup(gs)

        # Unit should be in trash
        assert dk_unit.zone == ZoneType.TRASH
        # Death trigger should be on chain
        assert not gs.chain.is_empty

    def test_friendly_death_trigger(self):
        """Observer fires 'When a friendly unit dies' when ally dies."""
        gs = make_game()

        # Observer with "When a friendly unit dies, draw 1"
        observer_def = CardDefinition(
            card_id="death_observer",
            name="Death Observer",
            card_type=CardType.UNIT,
            base_might=5,
            abilities=(
                AbilityDefinition(
                    ability_id="on_friendly_death",
                    ability_type="triggered",
                    trigger_condition="on_friendly_death",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        observer = CardInstance.create(observer_def, "p1", ZoneType.BASE)
        gs.instances[observer.instance_id] = observer
        gs.base_units["p1"].append(observer.instance_id)

        # Dying unit (same controller)
        dying = add_unit(gs, "p1", might=2, name="Doomed")
        dying.damage = 3  # lethal

        # Deck card
        deck_def = make_unit_def(name="Consolation Draw")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        logs = run_cleanup(gs)

        # Observer's friendly death trigger should be on chain
        assert not gs.chain.is_empty

    def test_enemy_death_doesnt_trigger_friendly_death(self):
        """Enemy dying does NOT trigger 'When a friendly unit dies' on your units."""
        gs = make_game()

        observer_def = CardDefinition(
            card_id="death_observer",
            name="Death Observer",
            card_type=CardType.UNIT,
            base_might=5,
            abilities=(
                AbilityDefinition(
                    ability_id="on_friendly_death",
                    ability_type="triggered",
                    trigger_condition="on_friendly_death",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        observer = CardInstance.create(observer_def, "p1", ZoneType.BASE)
        gs.instances[observer.instance_id] = observer
        gs.base_units["p1"].append(observer.instance_id)

        # Enemy dies (different controller)
        enemy = add_unit(gs, "p2", might=2, name="Enemy Doomed")
        enemy.damage = 3

        logs = run_cleanup(gs)

        # Chain should be empty — observer doesn't care about enemy deaths
        assert gs.chain.is_empty


# ---------------------------------------------------------------------------
# Combat triggers (combat.py → fire_event ATTACK_STARTED / DEFEND_STARTED)
# ---------------------------------------------------------------------------


class TestCombatTriggers:
    def _setup_combat(self, gs):
        """Set up a contested battlefield with units from both players."""
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        attacker = add_unit(gs, "p1", zone=ZoneType.BATTLEFIELD, might=3,
                           name="Attacker", bf_id=bf_id)
        defender = add_unit(gs, "p2", zone=ZoneType.BATTLEFIELD, might=3,
                           name="Defender", bf_id=bf_id)
        return bf_id, attacker, defender

    def test_attack_trigger_fires(self):
        """Unit with on_attack triggers when combat starts."""
        gs = make_game()

        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        # Attacker with "When I attack, draw 1"
        atk_def = CardDefinition(
            card_id="atk_trigger",
            name="Battle Drawer",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(
                AbilityDefinition(
                    ability_id="on_attack_draw",
                    ability_type="triggered",
                    trigger_condition="on_attack",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        atk = CardInstance.create(atk_def, "p1", ZoneType.BATTLEFIELD)
        atk.location_id = bf_id
        gs.instances[atk.instance_id] = atk
        bf.units.append(atk.instance_id)

        defender = add_unit(gs, "p2", zone=ZoneType.BATTLEFIELD, might=3,
                           name="Defender", bf_id=bf_id)

        # Deck card
        deck_def = make_unit_def(name="Combat Draw")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        logs = start_combat(gs, bf_id)

        # Attack trigger should be on chain
        assert not gs.chain.is_empty

    def test_defend_trigger_fires(self):
        """Unit with on_defend triggers when combat starts."""
        gs = make_game()

        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.contested_by = "p1"

        attacker = add_unit(gs, "p1", zone=ZoneType.BATTLEFIELD, might=3,
                           name="Attacker", bf_id=bf_id)

        # Defender with "When I defend, give me +2 might"
        def_unit_def = CardDefinition(
            card_id="def_trigger",
            name="Battle Buffer",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(
                AbilityDefinition(
                    ability_id="on_defend_buff",
                    ability_type="triggered",
                    trigger_condition="on_defend",
                    effect_ir=give_might(2),
                ),
            ),
        )
        def_unit = CardInstance.create(def_unit_def, "p2", ZoneType.BATTLEFIELD)
        def_unit.location_id = bf_id
        gs.instances[def_unit.instance_id] = def_unit
        bf.units.append(def_unit.instance_id)

        logs = start_combat(gs, bf_id)

        # Defend trigger should be on chain
        assert not gs.chain.is_empty


# ---------------------------------------------------------------------------
# Scoring triggers (scoring.py → fire_event HOLD / CONQUER)
# ---------------------------------------------------------------------------


class TestScoringTriggers:
    def test_hold_trigger_fires(self):
        """Unit with on_hold triggers during Beginning Phase scoring."""
        gs = make_game()

        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        # Unit at the controlled battlefield with "When I hold, draw 1"
        hold_def = CardDefinition(
            card_id="hold_trigger",
            name="Hold Drawer",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(
                AbilityDefinition(
                    ability_id="on_hold_draw",
                    ability_type="triggered",
                    trigger_condition="on_hold",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        hold_unit = CardInstance.create(hold_def, "p1", ZoneType.BATTLEFIELD)
        hold_unit.location_id = bf_id
        gs.instances[hold_unit.instance_id] = hold_unit
        bf.units.append(hold_unit.instance_id)

        # Deck card
        deck_def = make_unit_def(name="Hold Draw")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        logs = score_hold(gs, "p1")

        # Hold trigger should be on chain
        assert not gs.chain.is_empty
        assert gs.players["p1"].score == 1

    def test_conquer_trigger_fires(self):
        """Unit with on_conquer triggers when player conquers."""
        gs = make_game()

        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        # Unit at the conquered battlefield with "When I conquer, draw 1"
        conq_def = CardDefinition(
            card_id="conq_trigger",
            name="Conquer Drawer",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(
                AbilityDefinition(
                    ability_id="on_conquer_draw",
                    ability_type="triggered",
                    trigger_condition="on_conquer",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        conq_unit = CardInstance.create(conq_def, "p1", ZoneType.BATTLEFIELD)
        conq_unit.location_id = bf_id
        gs.instances[conq_unit.instance_id] = conq_unit
        bf.units.append(conq_unit.instance_id)

        deck_def = make_unit_def(name="Conquer Draw")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        logs = score_conquer(gs, "p1", bf_id)

        # Conquer trigger should be on chain
        assert not gs.chain.is_empty
        assert gs.players["p1"].score == 1


# ---------------------------------------------------------------------------
# Turn lifecycle triggers (state_machine → fire_event TURN_START / TURN_END)
# ---------------------------------------------------------------------------


class TestTurnTriggers:
    def test_turn_start_trigger(self):
        """Unit with on_turn_start triggers at start of turn."""
        gs = make_game()

        start_def = CardDefinition(
            card_id="turn_start",
            name="Turn Starter",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(
                AbilityDefinition(
                    ability_id="on_turn_start",
                    ability_type="triggered",
                    trigger_condition="on_turn_start",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        start_unit = CardInstance.create(start_def, "p1", ZoneType.BASE)
        gs.instances[start_unit.instance_id] = start_unit
        gs.base_units["p1"].append(start_unit.instance_id)

        deck_def = make_unit_def(name="Turn Draw")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        logs = _execute_awaken(gs)

        # Turn start trigger should be on chain
        assert not gs.chain.is_empty

    def test_turn_end_trigger(self):
        """Unit with on_turn_end triggers at end of turn."""
        gs = make_game()

        end_def = CardDefinition(
            card_id="turn_end",
            name="Turn Ender",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(
                AbilityDefinition(
                    ability_id="on_turn_end",
                    ability_type="triggered",
                    trigger_condition="on_turn_end",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        end_unit = CardInstance.create(end_def, "p1", ZoneType.BASE)
        gs.instances[end_unit.instance_id] = end_unit
        gs.base_units["p1"].append(end_unit.instance_id)

        deck_def = make_unit_def(name="End Draw")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        logs = _execute_end_of_turn(gs)

        # Turn end trigger should be on chain
        assert not gs.chain.is_empty


# ---------------------------------------------------------------------------
# Move triggers (action_executor → fire_event UNIT_MOVED_TO_BF)
# ---------------------------------------------------------------------------


class TestMoveTriggers:
    def test_move_to_bf_trigger(self):
        """Unit with on_move_to_bf triggers when moved to a battlefield."""
        gs = make_game()

        bf_id = list(gs.battlefields.keys())[0]

        move_def = CardDefinition(
            card_id="move_trigger",
            name="Move Reactor",
            card_type=CardType.UNIT,
            base_might=3,
            abilities=(
                AbilityDefinition(
                    ability_id="on_move_to_bf",
                    ability_type="triggered",
                    trigger_condition="on_move_to_bf",
                    effect_ir=draw_cards(1),
                ),
            ),
        )
        move_unit = CardInstance.create(move_def, "p1", ZoneType.BASE)
        move_unit.location_id = "p1"
        gs.instances[move_unit.instance_id] = move_unit
        gs.base_units["p1"].append(move_unit.instance_id)

        deck_def = make_unit_def(name="Move Draw")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        # Simulate the move (without full action_executor to avoid phase checks)
        fire_event(gs, GameEvent.UNIT_MOVED_TO_BF, {
            "card_id": move_unit.instance_id,
            "player_id": "p1",
            "battlefield_id": bf_id,
        })

        # Move trigger should be on chain
        assert not gs.chain.is_empty


# ---------------------------------------------------------------------------
# Target system integration with action_validator
# ---------------------------------------------------------------------------


class TestTargetValidation:
    def test_count_targets_via_target_system(self):
        """action_validator uses target_system for counting available targets."""
        from app.engine.action_validator import _count_available_targets

        gs = make_game()
        add_unit(gs, "p1", might=3, name="Friendly1")
        add_unit(gs, "p1", might=3, name="Friendly2")
        add_unit(gs, "p2", might=3, name="Enemy1")

        assert _count_available_targets(gs, "friendly_unit", "p1") == 2
        assert _count_available_targets(gs, "enemy_unit", "p1") == 1
        assert _count_available_targets(gs, "unit", "p1") == 3

    def test_validate_target_via_target_system(self):
        """action_validator uses target_system for individual target validation."""
        from app.engine.action_validator import _is_valid_target

        gs = make_game()
        friendly = add_unit(gs, "p1", might=3, name="Friendly")
        enemy = add_unit(gs, "p2", might=3, name="Enemy")

        assert _is_valid_target(gs, friendly.instance_id, "friendly_unit", "p1") is True
        assert _is_valid_target(gs, enemy.instance_id, "friendly_unit", "p1") is False
        assert _is_valid_target(gs, enemy.instance_id, "enemy_unit", "p1") is True
        assert _is_valid_target(gs, friendly.instance_id, "unit", "p1") is True

    def test_spell_target_validation_with_ir(self):
        """Spell with effect_ir target spec validates through target_system."""
        from app.engine.action_validator import _validate_spell_targets, ValidationResult

        gs = make_game()
        target = add_unit(gs, "p2", might=5, name="Target")

        spell_def = CardDefinition(
            card_id="ir_spell",
            name="IR Bolt",
            card_type=CardType.SPELL,
            abilities=(
                AbilityDefinition(
                    ability_id="bolt",
                    ability_type="activated",
                    effect_ir=deal_damage(3, {"obj_type": "unit", "scope": "enemy", "count": 1}),
                    targets_required=1,
                    target_type="enemy_unit",
                ),
            ),
        )
        spell = CardInstance.create(spell_def, "p1", ZoneType.HAND)
        gs.instances[spell.instance_id] = spell

        # Valid target
        result = _validate_spell_targets(gs, spell, "p1", [target.instance_id])
        assert result.ok

        # No targets provided
        result = _validate_spell_targets(gs, spell, "p1", [])
        assert not result.ok
