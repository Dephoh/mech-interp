"""Headless two-player scenario tests proving core card mechanics work.

Each test creates a minimal game state and plays out specific card interactions
to verify the engine handles them correctly. Tests are grouped by mechanic.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from helpers import add_unit, make_game, make_unit_def

from app.engine.card_types import AbilityDefinition, CardDefinition, CardInstance
from app.engine.chain import push_card_to_chain, resolve_top_item
from app.engine.cleanup import run_cleanup
from app.engine.effect_ir import (
    TargetSpec,
    ConditionSpec,
    FilterSpec,
    deal_damage,
    draw_cards,
    give_might,
    buff,
    stun,
    heal,
    kill,
    ready,
    play_token,
    counter,
    return_to_hand,
    recycle,
    banish,
    discard,
    add_energy,
    score_points,
    sequence,
    conditional,
    for_each,
    optional,
    might_lte,
    SELF_TARGET,
    ENEMY_UNIT,
    FRIENDLY_UNIT,
)
from app.engine.effect_resolver import resolve_effect_ir
from app.engine.enums import CardType, Domain, Phase, ZoneType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spell(gs, player_id, name, effect_ir, cost_energy=2, targets_required=0):
    """Create a spell card with an effect_ir ability and put it in hand."""
    spell_def = CardDefinition(
        card_id=f"test_{name.lower().replace(' ', '_')}",
        name=name,
        card_type=CardType.SPELL,
        cost_energy=cost_energy,
        abilities=(
            AbilityDefinition(
                ability_id=f"test_{name.lower()}_ab",
                ability_type="activated",
                effect_ir=effect_ir,
                targets_required=targets_required,
            ),
        ),
    )
    inst = CardInstance.create(spell_def, player_id, ZoneType.HAND)
    gs.instances[inst.instance_id] = inst
    gs.players[player_id].hand.append(inst.instance_id)
    return inst


def _make_unit_with_trigger(gs, player_id, name, trigger, effect_ir, might=3):
    """Create a unit with a triggered ability and put it in hand."""
    unit_def = CardDefinition(
        card_id=f"test_{name.lower().replace(' ', '_')}",
        name=name,
        card_type=CardType.UNIT,
        base_might=might,
        abilities=(
            AbilityDefinition(
                ability_id=f"test_{name.lower()}_ab",
                ability_type="triggered",
                trigger_condition=trigger,
                effect_ir=effect_ir,
            ),
        ),
    )
    inst = CardInstance.create(unit_def, player_id, ZoneType.HAND)
    gs.instances[inst.instance_id] = inst
    gs.players[player_id].hand.append(inst.instance_id)
    return inst


def _add_gear(gs, player_id, name="Sword", might_bonus=3):
    """Add a gear card to a player's base."""
    defn = CardDefinition(
        card_id=f"gear_{name.lower().replace(' ', '_')}",
        name=name,
        card_type=CardType.GEAR,
        might_bonus=might_bonus,
    )
    inst = CardInstance.create(defn, player_id, ZoneType.BASE)
    gs.instances[inst.instance_id] = inst
    gs.base_gear[player_id].append(inst.instance_id)
    return inst


def _add_deck_cards(gs, player_id, count=5):
    """Add N filler cards to a player's deck so they can draw."""
    cards = []
    for i in range(count):
        defn = make_unit_def(name=f"Deck Filler {i}")
        inst = CardInstance.create(defn, player_id, ZoneType.MAIN_DECK)
        gs.instances[inst.instance_id] = inst
        gs.players[player_id].main_deck.append(inst.instance_id)
        cards.append(inst)
    return cards


def _add_hand_cards(gs, player_id, count=3):
    """Add N filler cards to a player's hand."""
    cards = []
    for i in range(count):
        defn = make_unit_def(name=f"Hand Filler {i}")
        inst = CardInstance.create(defn, player_id, ZoneType.HAND)
        gs.instances[inst.instance_id] = inst
        gs.players[player_id].hand.append(inst.instance_id)
        cards.append(inst)
    return cards


def _add_rune(gs, player_id, domain=Domain.FURY, to_board=True):
    """Add a rune to board or deck."""
    defn = CardDefinition(
        card_id=f"rune_{domain.value}",
        name=f"{domain.value.title()} Rune",
        card_type=CardType.RUNE,
        domains=(domain,),
    )
    zone = ZoneType.RUNE_BOARD if to_board else ZoneType.RUNE_DECK
    inst = CardInstance.create(defn, player_id, zone)
    gs.instances[inst.instance_id] = inst
    if to_board:
        gs.base_runes[player_id].append(inst.instance_id)
    else:
        gs.players[player_id].rune_deck.append(inst.instance_id)
    return inst


# ===========================================================================
# Scenario 1: Spell — Draw cards
# Models: Angle Shot, Confront, Find Your Center
# ===========================================================================


class TestSpellDraw:
    def test_spell_draws_one_card(self):
        """Angle Shot (sfd-011-221): Draw 1."""
        gs = make_game()
        deck_cards = _add_deck_cards(gs, "p1", 3)
        spell = _make_spell(gs, "p1", "Angle Shot", draw_cards(1))

        push_card_to_chain(gs, spell, "p1")
        logs = resolve_top_item(gs)

        assert deck_cards[0].instance_id in gs.players["p1"].hand
        assert deck_cards[0].zone == ZoneType.HAND
        assert spell.zone == ZoneType.TRASH

    def test_spell_draws_multiple(self):
        """Progress Day (ogn-114-298): Draw 4."""
        gs = make_game()
        deck_cards = _add_deck_cards(gs, "p1", 5)
        spell = _make_spell(gs, "p1", "Progress Day", draw_cards(4), cost_energy=6)

        push_card_to_chain(gs, spell, "p1")
        logs = resolve_top_item(gs)

        hand = gs.players["p1"].hand
        for c in deck_cards[:4]:
            assert c.instance_id in hand
        assert deck_cards[4].instance_id not in hand  # 5th stays in deck


# ===========================================================================
# Scenario 2: Spell — Give Might
# Models: Against the Odds, En Garde, Call to Glory
# ===========================================================================


class TestSpellGiveMight:
    def test_give_plus2_might(self):
        """Against the Odds (sfd-001-221): Give +2 Might to a friendly unit."""
        gs = make_game()
        target = add_unit(gs, "p1", might=3, name="Soldier")
        ir = give_might(2, TargetSpec(obj_type="unit", scope="friendly", count=1))
        spell = _make_spell(gs, "p1", "Against the Odds", ir, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        logs = resolve_top_item(gs)

        assert target.effective_might == 5  # 3 base + 2

    def test_negative_might(self):
        """Eclipse (unl-063-219): Give -4 Might."""
        gs = make_game()
        target = add_unit(gs, "p1", might=6, name="Big Unit")
        ir = give_might(-4, TargetSpec(obj_type="unit", scope="friendly", count=1))
        spell = _make_spell(gs, "p1", "Eclipse", ir, cost_energy=3)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        logs = resolve_top_item(gs)

        assert target.effective_might == 2  # 6 - 4


# ===========================================================================
# Scenario 3: Spell — Deal Damage
# Models: Hextech Ray, Crescent Strike, Final Spark
# ===========================================================================


class TestSpellDealDamage:
    def test_deal_3_damage(self):
        """Hextech Ray (ogn-009-298): Deal 3 to a unit."""
        gs = make_game()
        target = add_unit(gs, "p2", might=5, name="Target")
        ir = deal_damage(3, TargetSpec(obj_type="unit", scope="friendly", count=1))
        spell = _make_spell(gs, "p1", "Hextech Ray", ir, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        logs = resolve_top_item(gs)

        assert target.damage == 3
        assert target.is_alive  # 5 might - 3 damage = alive

    def test_deal_lethal_damage_kills_on_cleanup(self):
        """Deal damage >= might → unit dies after cleanup."""
        gs = make_game()
        target = add_unit(gs, "p2", might=3, name="Fragile")
        ir = deal_damage(5)
        spell = _make_spell(gs, "p1", "Overkill", ir, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        resolve_top_item(gs)

        assert target.damage == 5
        assert not target.is_alive
        # Run cleanup to move to trash
        run_cleanup(gs)
        assert target.zone == ZoneType.TRASH

    def test_aoe_damage_all_enemies(self):
        """Firestorm (ogs-002-024): Deal 3 to all enemy units."""
        gs = make_game()
        e1 = add_unit(gs, "p2", might=5, name="Enemy1")
        e2 = add_unit(gs, "p2", might=5, name="Enemy2")
        friendly = add_unit(gs, "p1", might=5, name="Ally")
        ir = deal_damage(3, TargetSpec(obj_type="unit", scope="enemy", count=-1))
        spell = _make_spell(gs, "p1", "Firestorm", ir, cost_energy=6)

        push_card_to_chain(gs, spell, "p1")
        resolve_top_item(gs)

        assert e1.damage == 3
        assert e2.damage == 3
        assert friendly.damage == 0  # friendly untouched


# ===========================================================================
# Scenario 4: Spell — Kill a Gear
# Models: Detonate (sfd-005-221)
# ===========================================================================


class TestKillGear:
    def test_detonate_kills_gear(self):
        """Detonate (sfd-005-221): Kill a gear."""
        gs = make_game()
        gear = _add_gear(gs, "p2", "Enemy Artifact", might_bonus=2)
        ir = kill(TargetSpec(obj_type="gear", scope="any", count=1))
        spell = _make_spell(gs, "p1", "Detonate", ir, cost_energy=1, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[gear.instance_id])
        resolve_top_item(gs)

        assert gear.zone == ZoneType.TRASH
        assert gear.instance_id not in gs.base_gear["p2"]

    def test_kill_attached_gear_detaches(self):
        """Kill gear that's attached to a unit — gear goes to trash, unit keeps its might."""
        from app.engine.effect_primitives import prim_attach

        gs = make_game()
        unit = add_unit(gs, "p2", might=3, name="Equipped Soldier")
        gear = _add_gear(gs, "p2", "B.F. Sword", might_bonus=3)
        prim_attach(unit, gs, {}, [gear.instance_id, unit.instance_id])
        gs.recalculate_modifiers()
        assert unit.effective_might == 6

        ir = kill(TargetSpec(obj_type="gear", scope="any", count=1))
        spell = _make_spell(gs, "p1", "Detonate", ir, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[gear.instance_id])
        resolve_top_item(gs)

        assert gear.zone == ZoneType.TRASH
        gs.recalculate_modifiers()
        assert unit.effective_might == 3  # back to base might


# ===========================================================================
# Scenario 5: Spell — Kill a Unit
# Models: Thermo Beam, King's Edict, The Ruination
# ===========================================================================


class TestKillUnit:
    def test_kill_single_unit(self):
        """Thermo Beam (ogn-022-298): Kill a unit."""
        gs = make_game()
        target = add_unit(gs, "p2", might=8, name="Big Target")
        ir = kill(TargetSpec(obj_type="unit", scope="any", count=1))
        spell = _make_spell(gs, "p1", "Thermo Beam", ir, cost_energy=5, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        resolve_top_item(gs)

        assert target.damage >= target.effective_might
        assert not target.is_alive

    def test_ruination_kills_all(self):
        """The Ruination (unl-180-219): Kill all units (scope=any, count=-1)."""
        gs = make_game()
        u1 = add_unit(gs, "p1", might=3, name="Own Unit")
        u2 = add_unit(gs, "p2", might=5, name="Enemy Unit")
        u3 = add_unit(gs, "p2", might=7, name="Big Enemy")
        ir = kill(TargetSpec(obj_type="unit", scope="any", count=-1))
        spell = _make_spell(gs, "p1", "The Ruination", ir, cost_energy=9)

        push_card_to_chain(gs, spell, "p1")
        resolve_top_item(gs)

        for u in [u1, u2, u3]:
            assert not u.is_alive


# ===========================================================================
# Scenario 6: Spell — Return to Hand (Bounce)
# Models: Retreat, Gust, Morbid Return
# ===========================================================================


class TestBounce:
    def test_bounce_clears_damage(self):
        """Retreat (ogn-104-298): Return friendly unit to hand. Damage resets."""
        gs = make_game()
        target = add_unit(gs, "p1", might=5, name="Wounded Ally")
        target.damage = 3
        ir = return_to_hand(TargetSpec(obj_type="unit", scope="friendly", count=1))
        spell = _make_spell(gs, "p1", "Retreat", ir, cost_energy=1, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        resolve_top_item(gs)

        assert target.zone == ZoneType.HAND
        assert target.damage == 0
        assert target.instance_id in gs.players["p1"].hand
        assert target.instance_id not in gs.base_units["p1"]


# ===========================================================================
# Scenario 7: Spell — Stun
# Models: Rune Prison, Facebreaker
# ===========================================================================


class TestStun:
    def test_stun_unit(self):
        """Rune Prison (ogn-050-298): Stun a unit."""
        gs = make_game()
        target = add_unit(gs, "p2", might=4, name="Enemy")
        ir = stun(TargetSpec(obj_type="unit", scope="friendly", count=1))
        spell = _make_spell(gs, "p1", "Rune Prison", ir, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        resolve_top_item(gs)

        assert target.stunned is True
        assert target.combat_might == 0  # stunned units have 0 combat might


# ===========================================================================
# Scenario 8: Spell — Buff
# Models: Showstopper, Stand United
# ===========================================================================


class TestBuff:
    def test_buff_increases_might_by_one(self):
        """Showstopper (ogn-270-298): Buff a friendly unit (+1 Might)."""
        gs = make_game()
        target = add_unit(gs, "p1", might=3, name="Soldier")
        ir = buff(TargetSpec(obj_type="unit", scope="friendly", count=1))
        spell = _make_spell(gs, "p1", "Showstopper", ir, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        resolve_top_item(gs)

        assert target.buff_counter is True
        assert target.effective_might == 4

    def test_buff_doesnt_stack(self):
        """Rule: Buff is +1 max — applying twice doesn't stack."""
        gs = make_game()
        target = add_unit(gs, "p1", might=3, name="Soldier")
        target.buff_counter = True

        source = add_unit(gs, "p1", name="Buffer")
        ir = buff()
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])

        assert target.effective_might == 4  # still just +1
        assert "already has a buff" in logs[0]


# ===========================================================================
# Scenario 9: Heal
# Models: Janna Savior (sfd-053-221), heal primitives
# ===========================================================================


class TestHeal:
    def test_heal_removes_damage(self):
        """Heal 2: remove 2 damage from a wounded unit."""
        gs = make_game()
        target = add_unit(gs, "p1", might=5, name="Wounded")
        target.damage = 4
        source = add_unit(gs, "p1", name="Healer")

        logs = resolve_effect_ir(heal(2), source, gs, [target.instance_id])
        assert target.damage == 2

    def test_heal_all_full_recovery(self):
        """Janna, Savior: Heal all damage from a unit."""
        gs = make_game()
        target = add_unit(gs, "p1", might=8, name="Badly Wounded")
        target.damage = 7
        source = add_unit(gs, "p1", name="Janna")

        logs = resolve_effect_ir(heal("all"), source, gs, [target.instance_id])
        assert target.damage == 0
        assert target.is_alive


# ===========================================================================
# Scenario 10: Counter a Spell
# Models: Defy, Wind Wall, Lilting Lullaby
# ===========================================================================


class TestCounter:
    def test_counter_removes_spell_from_chain(self):
        """Defy (ogn-045-298): Counter a spell."""
        gs = make_game()
        target_unit = add_unit(gs, "p2", might=5, name="Would-Be Victim")

        # P1 plays a damage spell
        damage_spell = _make_spell(gs, "p1", "Bolt", deal_damage(5), targets_required=1)
        push_card_to_chain(gs, damage_spell, "p1", targets=[target_unit.instance_id])
        assert not gs.chain.is_empty

        # P2 counters with Defy
        counter_ir = counter(TargetSpec(obj_type="spell", zone="chain"))
        counter_spell = _make_spell(gs, "p2", "Defy", counter_ir, cost_energy=1, targets_required=1)
        push_card_to_chain(gs, counter_spell, "p2", targets=[damage_spell.instance_id])

        # Resolve the counter first (LIFO)
        resolve_top_item(gs)

        # Original spell should be gone from chain and in trash
        assert damage_spell.zone == ZoneType.TRASH
        assert target_unit.damage == 0  # never resolved


# ===========================================================================
# Scenario 11: Play Tokens
# Models: Arise!, Guards!, Recruit the Vanguard
# ===========================================================================


class TestPlayToken:
    def test_create_sand_soldier(self):
        """Arise! (sfd-198-221): Create a Sand Soldier token (2 Might)."""
        gs = make_game()
        ir = play_token(name="Sand Soldier", might=2, temporary=False)
        spell = _make_spell(gs, "p1", "Arise", ir, cost_energy=6)

        push_card_to_chain(gs, spell, "p1")
        resolve_top_item(gs)

        tokens = [
            uid for uid in gs.base_units["p1"]
            if gs.instances[uid].name == "Sand Soldier"
        ]
        assert len(tokens) == 1
        token = gs.instances[tokens[0]]
        assert token.effective_might == 2
        assert token.zone == ZoneType.BASE

    def test_create_multiple_tokens(self):
        """Recruit the Vanguard (ogs-015-024): Create 4 Recruit tokens."""
        gs = make_game()
        ir = {"type": "play_token", "name": "Recruit", "might": 1, "temporary": False, "count": 4}
        spell = _make_spell(gs, "p1", "Recruit the Vanguard", ir, cost_energy=6)

        push_card_to_chain(gs, spell, "p1")
        resolve_top_item(gs)

        tokens = [
            uid for uid in gs.base_units["p1"]
            if gs.instances[uid].name == "Recruit"
        ]
        assert len(tokens) == 4

    def test_create_gold_gear_token(self):
        """Bushwhack (sfd-004-221): Create a Gold gear token (exhausted)."""
        gs = make_game()
        ir = {"type": "play_token", "name": "Gold", "token_type": "gear",
              "might": 0, "count": 1, "exhausted": True}
        spell = _make_spell(gs, "p1", "Bushwhack", ir)

        push_card_to_chain(gs, spell, "p1")
        resolve_top_item(gs)

        gear_ids = gs.base_gear.get("p1", [])
        assert len(gear_ids) == 1
        gold = gs.instances[gear_ids[0]]
        assert gold.name == "Gold"
        assert gold.exhausted is True


# ===========================================================================
# Scenario 12: Discard
# Models: Get Excited!, Bewitching Spirit
# ===========================================================================


class TestDiscard:
    def test_self_discard(self):
        """Get Excited! (ogn-008-298): Controller discards 1."""
        gs = make_game()
        hand_cards = _add_hand_cards(gs, "p1", 3)
        source = add_unit(gs, "p1", name="Source")

        logs = resolve_effect_ir(discard(1), source, gs, [])
        assert len(gs.players["p1"].hand) == 2  # lost one
        # The discarded card should be in trash
        in_trash = [c for c in hand_cards if c.zone == ZoneType.TRASH]
        assert len(in_trash) == 1

    def test_opponent_discard(self):
        """Bewitching Spirit (unl-121-219): on_play → opponent discards 1."""
        gs = make_game()
        opp_hand = _add_hand_cards(gs, "p2", 3)
        source = add_unit(gs, "p1", name="Bewitching Spirit")

        logs = resolve_effect_ir(discard(1, player="opponent"), source, gs, [])
        assert len(gs.players["p2"].hand) == 2


# ===========================================================================
# Scenario 13: Sequence — Deal Damage + Draw
# Models: Void Seeker, Falling Star, Monster Harpoon
# ===========================================================================


class TestSequence:
    def test_void_seeker_deal_then_draw(self):
        """Void Seeker (ogn-024-298): Deal 4 damage, then draw 1."""
        gs = make_game()
        target = add_unit(gs, "p2", might=5, name="Target")
        deck_cards = _add_deck_cards(gs, "p1", 3)
        ir = sequence([
            deal_damage(4, TargetSpec(obj_type="unit", scope="friendly", count=1)),
            draw_cards(1),
        ])
        spell = _make_spell(gs, "p1", "Void Seeker", ir, cost_energy=3, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        resolve_top_item(gs)

        assert target.damage == 4
        assert deck_cards[0].instance_id in gs.players["p1"].hand

    def test_blood_money_kill_then_draw(self):
        """Blood Money (sfd-162-221): Kill friendly, if succeeded → draw 1."""
        gs = make_game()
        sacrifice = add_unit(gs, "p1", might=2, name="Sacrifice")
        deck_cards = _add_deck_cards(gs, "p1", 3)
        ir = sequence([
            kill(TargetSpec(obj_type="unit", scope="friendly", count=1)),
            conditional(
                {"cond_type": "previous_effect_succeeded"},
                then=draw_cards(1),
            ),
        ])
        spell = _make_spell(gs, "p1", "Blood Money", ir, targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[sacrifice.instance_id])
        resolve_top_item(gs)

        assert not sacrifice.is_alive
        assert deck_cards[0].instance_id in gs.players["p1"].hand


# ===========================================================================
# Scenario 14: Conditional — Kill Gear, If Succeeded Buff
# Models: Adaptatron, Pickpocket, Disarming Rake
# ===========================================================================


class TestConditionalKillGear:
    def test_adaptatron_kills_gear_and_buffs(self):
        """Adaptatron (ogn-056-298): Kill a gear → if succeeded, buff self."""
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Adaptatron")
        gear = _add_gear(gs, "p2", "Target Gear", might_bonus=1)
        ir = sequence([
            optional(kill(TargetSpec(obj_type="gear", scope="any", count=1))),
            conditional(
                {"cond_type": "previous_effect_succeeded"},
                then=buff(TargetSpec(scope="self")),
            ),
        ])

        logs = resolve_effect_ir(ir, source, gs, [])
        assert gear.zone == ZoneType.TRASH
        assert source.buff_counter is True
        assert source.effective_might == 4  # 3 + buff

    def test_adaptatron_no_gear_no_buff(self):
        """Adaptatron with no gear on board → kill fails → no buff."""
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Adaptatron")
        ir = sequence([
            optional(kill(TargetSpec(obj_type="gear", scope="any", count=1))),
            conditional(
                {"cond_type": "previous_effect_succeeded"},
                then=buff(TargetSpec(scope="self")),
            ),
        ])

        logs = resolve_effect_ir(ir, source, gs, [])
        assert source.buff_counter is False
        assert source.effective_might == 3


# ===========================================================================
# Scenario 15: Move
# Models: Relentless Pursuit, Ride the Wind, Fight or Flight
# ===========================================================================


class TestMove:
    def test_move_unit_to_base(self):
        """Emperor's Divide (sfd-043-221): Move a unit back to owner's base."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        target = add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=3, name="Displaced", bf_id=bf_id)
        source = add_unit(gs, "p1", name="Caster")

        ir = {"type": "move", "target": {"obj_type": "unit", "scope": "any", "count": 1},
              "destination": {"zone": "base", "location": "owner"}}

        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])
        assert target.zone == ZoneType.BASE
        assert target.instance_id not in gs.battlefields[bf_id].units


# ===========================================================================
# Scenario 16: Ready
# Models: On the Hunt, Wallop, Last Breath
# ===========================================================================


class TestReady:
    def test_ready_exhausted_unit(self):
        """Wallop (ogn-146-298): Ready a friendly unit."""
        gs = make_game()
        target = add_unit(gs, "p1", might=3, name="Tired Soldier")
        target.exhausted = True
        source = add_unit(gs, "p1", name="Commander")

        ir = ready(TargetSpec(obj_type="unit", scope="friendly", count=1))
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])

        assert target.exhausted is False


# ===========================================================================
# Scenario 17: Banish
# Models: Portal Rescue, Thrill of the Hunt, Arcane Shift
# ===========================================================================


class TestBanish:
    def test_banish_removes_from_board(self):
        """Portal Rescue (ogn-102-298): Banish a friendly unit."""
        gs = make_game()
        target = add_unit(gs, "p1", might=4, name="Banished")
        source = add_unit(gs, "p1", name="Mage")

        ir = banish(TargetSpec(obj_type="unit", scope="friendly", count=1))
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])

        assert target.zone == ZoneType.BANISHMENT
        assert target.instance_id not in gs.base_units["p1"]


# ===========================================================================
# Scenario 18: Recycle
# Models: Divine Judgment, Double Trouble, Fate Weaver
# ===========================================================================


class TestRecycle:
    def test_recycle_returns_to_deck_bottom(self):
        """Divine Judgment (ogn-244-298): Recycle a unit to bottom of deck."""
        gs = make_game()
        target = add_unit(gs, "p2", might=5, name="Recycled")
        source = add_unit(gs, "p1", name="Judge")

        ir = recycle(TargetSpec(obj_type="unit", scope="any", count=1))
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])

        assert target.zone == ZoneType.MAIN_DECK
        assert target.instance_id in gs.players["p2"].main_deck
        assert target.instance_id not in gs.base_units["p2"]


# ===========================================================================
# Scenario 19: ForEach — AoE Buff
# Models: Enthusiastic Promoter (on_hold → buff all units here)
# ===========================================================================


class TestForEach:
    def test_for_each_enemy_damage(self):
        """Cannon Barrage (ogn-127-298): Deal 2 to ALL enemy units."""
        gs = make_game()
        e1 = add_unit(gs, "p2", might=5, name="E1")
        e2 = add_unit(gs, "p2", might=5, name="E2")
        e3 = add_unit(gs, "p2", might=5, name="E3")
        friendly = add_unit(gs, "p1", might=3, name="Ally")
        source = add_unit(gs, "p1", name="Caster")

        ir = for_each(
            {"obj_type": "unit", "scope": "enemy", "count": -1},
            deal_damage(2),
        )
        logs = resolve_effect_ir(ir, source, gs, [])

        assert e1.damage == 2
        assert e2.damage == 2
        assert e3.damage == 2
        assert friendly.damage == 0


# ===========================================================================
# Scenario 20: Unit on_play Triggers
# Models: Buhru Captain, Blastcone Fae, Kinkou Initiate
# ===========================================================================


class TestUnitOnPlay:
    def test_buhru_captain_on_play_draws(self):
        """Buhru Captain (sfd-091-221): On play, optionally draw 1."""
        gs = make_game()
        deck_cards = _add_deck_cards(gs, "p1", 3)
        unit = _make_unit_with_trigger(
            gs, "p1", "Buhru Captain", "on_play",
            optional(draw_cards(1)), might=3,
        )

        push_card_to_chain(gs, unit, "p1")
        # Unit goes to base immediately (permanent), trigger goes on chain
        assert unit.zone == ZoneType.BASE
        assert not gs.chain.is_empty

        # Resolve the on_play trigger
        logs = resolve_top_item(gs)
        assert deck_cards[0].instance_id in gs.players["p1"].hand

    def test_blastcone_fae_on_play_gives_might(self):
        """Blastcone Fae (ogn-097-298): On play, +1 Might to a friendly unit."""
        gs = make_game()
        ally = add_unit(gs, "p1", might=3, name="Ally")
        ir = give_might(1, TargetSpec(obj_type="unit", scope="friendly", count=1), "turn")
        unit = _make_unit_with_trigger(gs, "p1", "Blastcone Fae", "on_play", ir, might=2)

        push_card_to_chain(gs, unit, "p1")
        logs = resolve_top_item(gs)  # resolve the trigger

        assert ally.effective_might == 4  # 3 + 1 from trigger

    def test_angler_beast_bounces_small_units(self):
        """Angler Beast (unl-132-219): On play, return all units with Might ≤ 2."""
        gs = make_game()
        small = add_unit(gs, "p2", might=2, name="Tiny")
        big = add_unit(gs, "p2", might=5, name="Big")
        ir = return_to_hand(TargetSpec(
            obj_type="unit", scope="any", count=-1,
            filters=(might_lte(2),),
        ))
        unit = _make_unit_with_trigger(gs, "p1", "Angler Beast", "on_play", ir, might=5)

        push_card_to_chain(gs, unit, "p1")
        logs = resolve_top_item(gs)

        assert small.zone == ZoneType.HAND
        assert big.zone == ZoneType.BASE  # unaffected


# ===========================================================================
# Scenario 21: Gear Attachment Might Bonus
# ===========================================================================


class TestGearMightBonus:
    def test_gear_adds_might(self):
        """Attach gear with might_bonus → unit might increases."""
        from app.engine.effect_primitives import prim_attach

        gs = make_game()
        unit = add_unit(gs, "p1", might=3, name="Soldier")
        gear = _add_gear(gs, "p1", "B.F. Sword", might_bonus=3)

        prim_attach(unit, gs, {}, [gear.instance_id, unit.instance_id])
        gs.recalculate_modifiers()

        assert unit.effective_might == 6
        assert gear.attached_to == unit.instance_id
        assert gear.instance_id in unit.attached_cards


# ===========================================================================
# Scenario 22: Gear Detaches on Unit Death
# ===========================================================================


class TestGearDetach:
    def test_gear_detaches_when_unit_killed(self):
        """Kill a unit with gear attached → gear goes to owner's base."""
        from app.engine.effect_primitives import prim_attach

        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        unit = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=3, name="Doomed Soldier", bf_id=bf_id)
        gear = _add_gear(gs, "p1", "Shield", might_bonus=2)
        prim_attach(unit, gs, {}, [gear.instance_id, unit.instance_id])
        gs.recalculate_modifiers()  # gear bonus now reflected in effective_might

        source = add_unit(gs, "p2", name="Killer")
        logs = resolve_effect_ir(kill(), source, gs, [unit.instance_id])

        assert not unit.is_alive
        run_cleanup(gs)
        assert unit.zone == ZoneType.TRASH
        assert gear.attached_to is None
        assert gear.zone == ZoneType.BASE
        assert gear.instance_id in gs.base_gear["p1"]


# ===========================================================================
# Scenario 23: Rune Exhaust for Energy
# ===========================================================================


class TestRuneExhaust:
    def test_exhaust_rune_adds_energy(self):
        """Exhaust a rune → +1 energy to pool (Rule 160.2.a)."""
        from app.engine.action_executor import execute_action
        from app.engine.enums import ActionType

        gs = make_game()
        rune = _add_rune(gs, "p1", domain=Domain.FURY, to_board=True)

        execute_action(gs, "p1", ActionType.EXHAUST_RUNE, {"instance_id": rune.instance_id})

        assert gs.players["p1"].rune_pool.energy == 1
        assert rune.exhausted is True


# ===========================================================================
# Scenario 24: Rune Recycle for Power
# ===========================================================================


class TestRuneRecycle:
    def test_recycle_rune_adds_power(self):
        """Recycle a rune → +1 domain power to pool (Rule 160.2.b)."""
        from app.engine.action_executor import execute_action
        from app.engine.enums import ActionType

        gs = make_game()
        rune = _add_rune(gs, "p1", domain=Domain.MIND, to_board=True)

        execute_action(gs, "p1", ActionType.RECYCLE_RUNE, {"instance_id": rune.instance_id})

        assert gs.players["p1"].rune_pool.power.get(Domain.MIND, 0) == 1
        assert rune.zone == ZoneType.RUNE_DECK
        assert rune.instance_id not in gs.base_runes["p1"]


# ===========================================================================
# Scenario 25: Channel Rune from Deck
# ===========================================================================


class TestChannelRune:
    def test_channel_moves_rune_to_board(self):
        """Channel rune: move from rune_deck to rune_board."""
        gs = make_game()
        rune = _add_rune(gs, "p1", domain=Domain.CALM, to_board=False)
        source = add_unit(gs, "p1", name="Channeler")

        ir = {"type": "channel_rune", "count": 1, "exhausted": False}
        logs = resolve_effect_ir(ir, source, gs, [])

        assert rune.instance_id in gs.base_runes["p1"]
        assert rune.exhausted is False


# ===========================================================================
# Scenario 26: Add Energy via Effect
# ===========================================================================


class TestAddEnergy:
    def test_add_energy_to_pool(self):
        """Effect adds energy to controller's pool."""
        gs = make_game()
        source = add_unit(gs, "p1", name="Generator")

        logs = resolve_effect_ir(add_energy(3), source, gs, [])
        assert gs.players["p1"].rune_pool.energy == 3


# ===========================================================================
# Scenario 27: Score Points
# Models: Ahri Alluring (on_hold → score 1 point)
# ===========================================================================


class TestScorePoints:
    def test_score_points_increases_score(self):
        """Ahri, Alluring (ogn-066a-298): Score 1 point on hold."""
        gs = make_game()
        source = add_unit(gs, "p1", name="Ahri")

        logs = resolve_effect_ir(score_points(1), source, gs, [])
        assert gs.players["p1"].score == 1

    def test_score_multiple_points(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Scorer")

        resolve_effect_ir(score_points(2), source, gs, [])
        resolve_effect_ir(score_points(1), source, gs, [])
        assert gs.players["p1"].score == 3


# ===========================================================================
# Scenario 28: Full Spell Chain — Play, Resolve, Trash
# Proves the entire lifecycle: hand → chain → resolve effects → trash
# ===========================================================================


class TestFullSpellLifecycle:
    def test_spell_lifecycle_draw(self):
        """Play a draw spell: hand → chain → resolve draw → spell goes to trash."""
        gs = make_game()
        deck_cards = _add_deck_cards(gs, "p1", 2)
        spell = _make_spell(gs, "p1", "Quick Draw", draw_cards(1))

        # Starts in hand
        assert spell.instance_id in gs.players["p1"].hand

        # Play to chain
        push_card_to_chain(gs, spell, "p1")
        assert spell.zone == ZoneType.CHAIN

        # Resolve
        logs = resolve_top_item(gs)
        assert spell.zone == ZoneType.TRASH
        assert deck_cards[0].instance_id in gs.players["p1"].hand

    def test_spell_lifecycle_targeted_damage(self):
        """Play targeted damage spell → resolves → target takes damage → spell trashed."""
        gs = make_game()
        target = add_unit(gs, "p2", might=5, name="Target")
        spell = _make_spell(gs, "p1", "Bolt", deal_damage(3), targets_required=1)

        push_card_to_chain(gs, spell, "p1", targets=[target.instance_id])
        resolve_top_item(gs)

        assert target.damage == 3
        assert spell.zone == ZoneType.TRASH


# ===========================================================================
# Scenario 29: Full Unit Play — Permanent Enters Board + Trigger
# ===========================================================================


class TestFullUnitPlay:
    def test_unit_enters_base_trigger_resolves(self):
        """Play unit with on_play draw: unit → base, trigger → chain → resolve."""
        gs = make_game()
        deck_cards = _add_deck_cards(gs, "p1", 2)
        unit = _make_unit_with_trigger(
            gs, "p1", "Kinkou Initiate", "on_play", draw_cards(1), might=3,
        )

        # Play the unit
        push_card_to_chain(gs, unit, "p1")

        # Unit should be on base immediately (permanent)
        assert unit.zone == ZoneType.BASE
        assert unit.instance_id in gs.base_units["p1"]

        # Trigger should be on chain
        assert not gs.chain.is_empty
        logs = resolve_top_item(gs)

        # Card drawn
        assert deck_cards[0].instance_id in gs.players["p1"].hand


# ===========================================================================
# Scenario 30: Multi-step Combo — Kill + Conditional Draw + Token
# Combines kill, conditional, draw, and token creation
# ===========================================================================


class TestMultiStepCombo:
    def test_salvage_combo(self):
        """Salvage (ogn-224-298): Optional kill a unit, then draw 1."""
        gs = make_game()
        sacrifice = add_unit(gs, "p1", might=2, name="Fodder")
        deck_cards = _add_deck_cards(gs, "p1", 3)
        ir = sequence([
            optional(kill(TargetSpec(obj_type="unit", scope="any", count=1))),
            draw_cards(1),
        ])
        spell = _make_spell(gs, "p1", "Salvage", ir)

        push_card_to_chain(gs, spell, "p1")
        resolve_top_item(gs)

        assert not sacrifice.is_alive
        assert deck_cards[0].instance_id in gs.players["p1"].hand

    def test_production_surge(self):
        """Production Surge (sfd-076-221): Create Mech token + draw 1."""
        gs = make_game()
        deck_cards = _add_deck_cards(gs, "p1", 2)
        ir = sequence([
            play_token(name="Mech", might=3, temporary=False),
            draw_cards(1),
        ])
        spell = _make_spell(gs, "p1", "Production Surge", ir, cost_energy=4)

        push_card_to_chain(gs, spell, "p1")
        resolve_top_item(gs)

        mechs = [uid for uid in gs.base_units["p1"] if gs.instances[uid].name == "Mech"]
        assert len(mechs) == 1
        assert gs.instances[mechs[0]].effective_might == 3
        assert deck_cards[0].instance_id in gs.players["p1"].hand
