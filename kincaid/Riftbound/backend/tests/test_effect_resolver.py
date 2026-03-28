"""Tests for the composable effect IR system (effect_ir, effect_primitives, effect_resolver)."""

from __future__ import annotations

import pytest
from helpers import add_unit, make_game, make_unit_def

from app.engine.card_types import AbilityDefinition, CardDefinition, CardInstance, KeywordInstance
from app.engine.effect_ir import (
    DEAL_DAMAGE,
    DRAW_CARDS,
    SEQUENCE,
    CONDITIONAL,
    FOR_EACH,
    OPTIONAL,
    TargetSpec,
    FilterSpec,
    ConditionSpec,
    deal_damage,
    draw_cards,
    give_might,
    buff,
    stun,
    heal,
    kill,
    play_token,
    counter,
    return_to_hand,
    sequence,
    conditional,
    for_each,
    optional,
    look_at_top,
    add_energy,
    ENEMY_UNIT,
    FRIENDLY_UNIT,
    ANY_UNIT,
    SELF_TARGET,
    ALL_UNITS,
    has_keyword,
    might_lte,
)
from app.engine.effect_resolver import resolve_effect_ir, evaluate_condition
from app.engine.enums import (
    CardType,
    CombatRole,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.game_state import _draw_cards


# ---------------------------------------------------------------------------
# Primitive tests
# ---------------------------------------------------------------------------


class TestDealDamage:
    def test_deal_damage_to_unit(self):
        gs = make_game()
        target = add_unit(gs, "p2", might=5, name="Victim")
        source = add_unit(gs, "p1", name="Attacker")

        ir = deal_damage(3)
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.damage == 3
        assert "deals 3 damage" in logs[0]

    def test_deal_damage_multiple_targets(self):
        gs = make_game()
        t1 = add_unit(gs, "p2", might=5, name="V1")
        t2 = add_unit(gs, "p2", might=5, name="V2")
        source = add_unit(gs, "p1", name="Attacker")

        ir = deal_damage(2)
        logs = resolve_effect_ir(ir, source, gs, [t1.instance_id, t2.instance_id])
        assert t1.damage == 2
        assert t2.damage == 2
        assert len(logs) == 2

    def test_deal_damage_zero(self):
        gs = make_game()
        target = add_unit(gs, "p2", might=3, name="Victim")
        source = add_unit(gs, "p1", name="Source")

        ir = deal_damage(0)
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.damage == 0


class TestDrawCards:
    def test_draw_one(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Drawer")

        # Put a card in the deck
        deck_card_def = make_unit_def(name="Deck Card")
        deck_card = CardInstance.create(deck_card_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        ir = draw_cards(1)
        logs = resolve_effect_ir(ir, source, gs, [])
        assert deck_card.instance_id in gs.players["p1"].hand
        assert deck_card.instance_id not in gs.players["p1"].main_deck

    def test_draw_for_opponent(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Source")

        deck_card_def = make_unit_def(name="Opp Card")
        deck_card = CardInstance.create(deck_card_def, "p2", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p2"].main_deck.append(deck_card.instance_id)

        ir = draw_cards(1, player="opponent")
        logs = resolve_effect_ir(ir, source, gs, [])
        assert deck_card.instance_id in gs.players["p2"].hand


class TestGiveMight:
    def test_give_might_this_turn(self):
        gs = make_game()
        target = add_unit(gs, "p1", might=3, name="Buffee")
        source = add_unit(gs, "p1", name="Buffer")

        ir = give_might(2)
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.effective_might == 5
        assert "+2 Might" in logs[0]


class TestBuff:
    def test_buff_unit(self):
        gs = make_game()
        target = add_unit(gs, "p1", might=3, name="Buffee")
        source = add_unit(gs, "p1", name="Buffer")

        ir = buff()
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.buff_counter is True
        assert target.effective_might == 4

    def test_buff_already_buffed(self):
        gs = make_game()
        target = add_unit(gs, "p1", might=3, name="Buffee")
        target.buff_counter = True
        source = add_unit(gs, "p1", name="Buffer")

        ir = buff()
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert "already has a buff" in logs[0]


class TestStun:
    def test_stun_unit(self):
        gs = make_game()
        target = add_unit(gs, "p2", might=3, name="Stunned")
        source = add_unit(gs, "p1", name="Stunner")

        ir = stun()
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.stunned is True
        assert target.combat_might == 0
        assert target.effective_might == 3  # not reduced, just combat_might


class TestHeal:
    def test_heal_partial(self):
        gs = make_game()
        target = add_unit(gs, "p1", might=5, name="Wounded")
        target.damage = 4
        source = add_unit(gs, "p1", name="Healer")

        ir = heal(2)
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.damage == 2

    def test_heal_all(self):
        gs = make_game()
        target = add_unit(gs, "p1", might=5, name="Wounded")
        target.damage = 4
        source = add_unit(gs, "p1", name="Healer")

        ir = heal("all")
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.damage == 0


class TestKill:
    def test_kill_unit(self):
        gs = make_game()
        target = add_unit(gs, "p2", might=5, name="Doomed")
        source = add_unit(gs, "p1", name="Killer")

        ir = kill()
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.damage >= target.effective_might
        assert not target.is_alive


class TestReturnToHand:
    def test_bounce_unit(self):
        gs = make_game()
        target = add_unit(gs, "p2", might=3, name="Bounced")
        target.damage = 2
        source = add_unit(gs, "p1", name="Bouncer")

        ir = return_to_hand()
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.zone == ZoneType.HAND
        assert target.damage == 0
        assert target.instance_id in gs.players["p2"].hand
        assert target.instance_id not in gs.base_units["p2"]


class TestPlayToken:
    def test_create_recruit_token(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Summoner")

        ir = play_token(name="Recruit", might=1, temporary=False)
        logs = resolve_effect_ir(ir, source, gs, [])

        # Find the token in p1's base
        token_ids = [
            uid for uid in gs.base_units["p1"]
            if gs.get_instance(uid) and gs.get_instance(uid).name == "Recruit"
        ]
        assert len(token_ids) == 1
        token = gs.get_instance(token_ids[0])
        assert token.effective_might == 1
        assert token.definition.card_type == CardType.UNIT

    def test_create_temporary_token(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Summoner")

        ir = play_token(name="Sprite", might=3, temporary=True, ready_on_enter=True)
        logs = resolve_effect_ir(ir, source, gs, [])

        token_ids = [
            uid for uid in gs.base_units["p1"]
            if gs.get_instance(uid) and gs.get_instance(uid).name == "Sprite"
        ]
        assert len(token_ids) == 1
        token = gs.get_instance(token_ids[0])
        assert token.effective_might == 3
        assert token.has_keyword(Keyword.TEMPORARY)
        assert not token.exhausted  # ready_on_enter


class TestAddEnergy:
    def test_add_energy(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Source")

        ir = add_energy(2)
        logs = resolve_effect_ir(ir, source, gs, [])
        assert gs.players["p1"].rune_pool.energy == 2


class TestLookAtTop:
    def test_vision(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Seer")

        deck_def = make_unit_def(name="Hidden Card")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.insert(0, deck_card.instance_id)

        ir = look_at_top(1, may_recycle=True)
        logs = resolve_effect_ir(ir, source, gs, [])
        assert "Hidden Card" in logs[0]


# ---------------------------------------------------------------------------
# Composition tests
# ---------------------------------------------------------------------------


class TestSequence:
    def test_deal_then_draw(self):
        gs = make_game()
        target = add_unit(gs, "p2", might=5, name="Victim")
        source = add_unit(gs, "p1", name="Caster")

        deck_def = make_unit_def(name="Draw Me")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        ir = sequence([
            deal_damage(3),
            draw_cards(1),
        ])
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.damage == 3
        assert deck_card.instance_id in gs.players["p1"].hand


class TestConditional:
    def test_condition_true(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=6, name="Big Unit")

        ir = conditional(
            ConditionSpec(cond_type="mighty"),
            then=draw_cards(1),
        )

        deck_def = make_unit_def(name="Reward")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        logs = resolve_effect_ir(ir, source, gs, [])
        assert deck_card.instance_id in gs.players["p1"].hand

    def test_condition_false(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Small Unit")

        ir = conditional(
            ConditionSpec(cond_type="mighty"),
            then=draw_cards(1),
        )

        deck_def = make_unit_def(name="Reward")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        logs = resolve_effect_ir(ir, source, gs, [])
        assert deck_card.instance_id not in gs.players["p1"].hand

    def test_condition_with_else(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Small")
        target = add_unit(gs, "p2", might=5, name="Victim")

        ir = conditional(
            ConditionSpec(cond_type="mighty"),
            then=draw_cards(1),
            else_=deal_damage(2),
        )
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.damage == 2  # else branch executed


class TestForEach:
    def test_damage_all_enemy_units(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Caster")
        e1 = add_unit(gs, "p2", might=5, name="E1")
        e2 = add_unit(gs, "p2", might=5, name="E2")
        friendly = add_unit(gs, "p1", might=5, name="Friendly")

        ir = for_each(
            ENEMY_UNIT.to_dict(),
            deal_damage(1),
        )
        logs = resolve_effect_ir(ir, source, gs, [])
        assert e1.damage == 1
        assert e2.damage == 1
        assert friendly.damage == 0


class TestOptional:
    def test_optional_executes(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Source")

        deck_def = make_unit_def(name="Optional Draw")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        ir = optional(draw_cards(1))
        logs = resolve_effect_ir(ir, source, gs, [])
        # Auto-play always executes optional effects
        assert deck_card.instance_id in gs.players["p1"].hand


# ---------------------------------------------------------------------------
# Condition evaluation tests
# ---------------------------------------------------------------------------


class TestConditions:
    def test_legion_true(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="LegionUnit")
        gs.players["p1"].played_main_deck_this_turn = True
        assert evaluate_condition({"cond_type": "legion"}, source, gs) is True

    def test_legion_false(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="LegionUnit")
        gs.players["p1"].played_main_deck_this_turn = False
        assert evaluate_condition({"cond_type": "legion"}, source, gs) is False

    def test_mighty_true(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=5, name="Big")
        assert evaluate_condition({"cond_type": "mighty"}, source, gs) is True

    def test_mighty_false(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=4, name="Small")
        assert evaluate_condition({"cond_type": "mighty"}, source, gs) is False

    def test_is_attacker(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Fighter")
        source.combat_role = CombatRole.ATTACKER
        assert evaluate_condition({"cond_type": "is_attacker"}, source, gs) is True
        source.combat_role = CombatRole.DEFENDER
        assert evaluate_condition({"cond_type": "is_attacker"}, source, gs) is False


# ---------------------------------------------------------------------------
# Auto-target resolution tests
# ---------------------------------------------------------------------------


class TestAutoResolve:
    def test_auto_resolve_enemy_units(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Caster")
        e1 = add_unit(gs, "p2", might=3, name="Enemy1")
        f1 = add_unit(gs, "p1", might=3, name="Friend1")

        ir = deal_damage(2, ENEMY_UNIT)
        logs = resolve_effect_ir(ir, source, gs, [])
        assert e1.damage == 2
        assert f1.damage == 0

    def test_auto_resolve_with_might_filter(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Caster")
        small = add_unit(gs, "p2", might=2, name="Small")
        big = add_unit(gs, "p2", might=5, name="Big")

        target_spec = TargetSpec(
            obj_type="unit", scope="enemy", count=-1,
            filters=(might_lte(2),),
        )
        ir = return_to_hand(target_spec)
        logs = resolve_effect_ir(ir, source, gs, [])
        assert small.zone == ZoneType.HAND
        assert big.zone == ZoneType.BASE  # unaffected


# ---------------------------------------------------------------------------
# Integration: effect_ir on AbilityDefinition through chain resolution
# ---------------------------------------------------------------------------


class TestChainIntegration:
    def test_spell_with_effect_ir_resolves(self):
        """A spell card with effect_ir resolves through the chain."""
        from app.engine.chain import push_card_to_chain, resolve_top_item

        gs = make_game()
        target = add_unit(gs, "p2", might=5, name="Victim")

        # Create a spell with an IR-based ability
        spell_def = CardDefinition(
            card_id="test_ir_spell",
            name="IR Bolt",
            card_type=CardType.SPELL,
            abilities=(
                AbilityDefinition(
                    ability_id="ir_bolt_effect",
                    ability_type="activated",
                    effect_ir=deal_damage(3),
                ),
            ),
        )
        spell = CardInstance.create(spell_def, "p1", ZoneType.HAND)
        gs.instances[spell.instance_id] = spell
        gs.players["p1"].hand.append(spell.instance_id)

        # Push spell to chain
        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])

        # Resolve it
        logs = resolve_top_item(gs)
        assert target.damage == 3
        assert spell.zone == ZoneType.TRASH

    def test_unit_with_play_effect_ir(self):
        """A unit with an on_play effect_ir triggers correctly."""
        from app.engine.chain import push_card_to_chain, resolve_top_item

        gs = make_game()

        deck_def = make_unit_def(name="Drawn Card")
        deck_card = CardInstance.create(deck_def, "p1", ZoneType.MAIN_DECK)
        gs.instances[deck_card.instance_id] = deck_card
        gs.players["p1"].main_deck.append(deck_card.instance_id)

        # Unit with on_play draw effect using IR
        unit_def = CardDefinition(
            card_id="test_ir_unit",
            name="Draw Unit",
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

        push_card_to_chain(gs, unit, "p1")

        # Unit should be on board, play effect on chain
        assert unit.zone == ZoneType.BASE
        assert not gs.chain.is_empty

        # Resolve the play effect
        logs = resolve_top_item(gs)
        assert deck_card.instance_id in gs.players["p1"].hand


# ---------------------------------------------------------------------------
# New primitive tests (Phase 3)
# ---------------------------------------------------------------------------


class TestGiveKeyword:
    def test_give_keyword_to_self(self):
        gs = make_game()
        unit = add_unit(gs, "p1", might=3)
        ir = {"type": "give_keyword", "keyword": "tank", "target": SELF_TARGET, "duration": "turn"}
        logs = resolve_effect_ir(ir, unit, gs, [])
        assert "tank" in unit.granted_keywords

    def test_give_keyword_to_target(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3)
        target = add_unit(gs, "p1", might=2)
        ir = {"type": "give_keyword", "keyword": "assault", "target": FRIENDLY_UNIT, "duration": "turn"}
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert "assault" in target.granted_keywords


class TestPlayTokenCount:
    def test_play_multiple_tokens(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3)
        ir = {"type": "play_token", "name": "Recruit", "might": 1, "count": 3}
        logs = resolve_effect_ir(ir, source, gs, [])
        # Should have 3 new tokens + the original unit in base
        units = gs.base_units["p1"]
        tokens = [u for u in units if gs.instances[u].name == "Recruit"]
        assert len(tokens) == 3

    def test_play_gear_token(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3)
        ir = {"type": "play_token", "name": "Gold", "token_type": "gear", "might": 0, "exhausted": True}
        logs = resolve_effect_ir(ir, source, gs, [])
        gear_ids = gs.base_gear.get("p1", [])
        assert len(gear_ids) == 1
        gear = gs.instances[gear_ids[0]]
        assert gear.name == "Gold"
        assert gear.exhausted


class TestChannelExhausted:
    def test_channel_rune_exhausted(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3)
        # Add a rune to the rune deck
        rune_def = CardDefinition(
            card_id="test_rune",
            name="Test Rune",
            card_type=CardType.RUNE,
        )
        rune = CardInstance.create(rune_def, "p1", ZoneType.RUNE_DECK)
        gs.instances[rune.instance_id] = rune
        gs.players["p1"].rune_deck.append(rune.instance_id)

        ir = {"type": "channel_rune", "count": 1, "exhausted": True}
        logs = resolve_effect_ir(ir, source, gs, [])
        assert rune.exhausted is True
        assert rune.instance_id in gs.base_runes.get("p1", [])


class TestRestrict:
    def test_restrict_logged(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3)
        ir = {"type": "restrict", "restriction": "cant_play", "scope": "opponents", "duration": "turn"}
        logs = resolve_effect_ir(ir, source, gs, [])
        assert any("cant_play" in l for l in logs)
