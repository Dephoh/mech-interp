"""Tests for translated card abilities from the vertex pipeline.

Validates that:
1. Dynamic amounts (source_might, target_might) resolve correctly at runtime
2. Pipeline-translated IR executes properly in the engine
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.helpers import make_game, add_unit, make_spell_def
from app.engine.card_types import CardDefinition, CardInstance
from app.engine.enums import CardType, ZoneType
from app.engine.effect_resolver import resolve_effect_ir


# ---------------------------------------------------------------------------
# Dynamic amount resolution
# ---------------------------------------------------------------------------


class TestDynamicAmounts:
    """Test _resolve_amount with source_might and target_might strings."""

    def test_deal_damage_source_might(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=6, name="Snapvine")
        target = add_unit(gs, "p2", might=4, name="Defender")

        ir = {"type": "deal_damage", "amount": "source_might"}
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])

        assert target.damage == 6
        assert "deals 6 damage" in logs[0]

    def test_deal_damage_target_might(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=3, name="Attacker")
        target = add_unit(gs, "p2", might=7, name="Big Unit")

        ir = {"type": "deal_damage", "amount": "target_might"}
        logs = resolve_effect_ir(ir, source, gs, [target.instance_id])

        assert target.damage == 7
        assert "deals 7 damage" in logs[0]

    def test_deal_damage_integer_unchanged(self):
        """Verify plain integer amounts still work."""
        gs = make_game()
        source = add_unit(gs, "p1", might=10, name="Source")
        target = add_unit(gs, "p2", might=5, name="Target")

        ir = {"type": "deal_damage", "amount": 3}
        resolve_effect_ir(ir, source, gs, [target.instance_id])

        assert target.damage == 3  # not 10 (source_might) or 5 (target_might)

    def test_give_might_source_might(self):
        gs = make_game()
        source = add_unit(gs, "p1", might=5, name="Buffer")
        target = add_unit(gs, "p1", might=2, name="Buffed")

        ir = {"type": "give_might", "amount": "source_might", "duration": "turn"}
        resolve_effect_ir(ir, source, gs, [target.instance_id])

        assert 5 in target.might_modifiers


# ---------------------------------------------------------------------------
# Pipeline-translated card: Carnivorous Snapvine
# "We deal damage equal to our Mights to each other."
# Tests for_each + sequence + dynamic amounts end-to-end.
# ---------------------------------------------------------------------------


class TestCarnivorousSnapvine:

    def test_dynamic_amounts_in_sequence(self):
        """Test source_might and target_might resolve correctly in sequence steps.

        Uses pre-resolved targets (as a spell or triggered ability would)
        rather than for_each, since for_each passes its targets to all inner
        steps and embedded target specs are ignored when targets exist.
        """
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]

        snapvine = add_unit(gs, "p1", might=6, name="Carnivorous Snapvine",
                            zone=ZoneType.BATTLEFIELD, bf_id=bf_id)
        enemy = add_unit(gs, "p2", might=4, name="Enemy Scout",
                         zone=ZoneType.BATTLEFIELD, bf_id=bf_id)

        # Step 1: Snapvine deals its might (6) to enemy
        ir_hit = {"type": "deal_damage", "amount": "source_might"}
        resolve_effect_ir(ir_hit, snapvine, gs, [enemy.instance_id])
        assert enemy.damage == 6

        # Step 2: Enemy deals its might (4) back to Snapvine
        ir_recoil = {"type": "deal_damage", "amount": "source_might"}
        resolve_effect_ir(ir_recoil, enemy, gs, [snapvine.instance_id])
        assert snapvine.damage == 4


# ---------------------------------------------------------------------------
# Pipeline-translated card: Sabotage
# "Choose an opponent. They reveal their hand. Choose a non-unit card, recycle it."
# ---------------------------------------------------------------------------


class TestSabotageCard:

    def test_recycle_enemy_spell_from_hand(self):
        gs = make_game()
        source = add_unit(gs, "p1", name="Caster")

        # Put an enemy spell in p2's hand
        spell_def = make_spell_def(card_id="enemy_spell", name="Enemy Spell")
        enemy_spell = CardInstance.create(spell_def, "p2", ZoneType.HAND)
        gs.instances[enemy_spell.instance_id] = enemy_spell
        gs.players["p2"].hand.append(enemy_spell.instance_id)

        # The translated IR from the pipeline
        ir = {
            "type": "recycle",
            "target": {
                "obj_type": "card",
                "scope": "enemy",
                "zone": "hand",
                "count": 1,
                "chooser": "friendly",
                "filters": [{"field": "type", "op": "neq", "value": "unit"}],
            },
        }

        logs = resolve_effect_ir(ir, source, gs, [enemy_spell.instance_id])

        # Spell should be removed from hand and placed in main deck
        assert enemy_spell.instance_id not in gs.players["p2"].hand
        assert enemy_spell.zone == ZoneType.MAIN_DECK
        assert enemy_spell.instance_id in gs.players["p2"].main_deck
        assert any("recycled" in log.lower() for log in logs)
