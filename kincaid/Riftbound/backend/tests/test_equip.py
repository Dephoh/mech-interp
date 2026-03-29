"""Tests for the Equip mechanic: gear attachment via activated ability.

Rule 744: Equip is an activated ability on gear.
  744.1.b: Pay cost, attach gear to a friendly unit (target).
  744.1.c: Formatted as "Equip [Cost]", short for "[Cost]: Attach this gear to a unit you control."
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engine.card_types import (
    AbilityDefinition,
    CardDefinition,
    CardInstance,
    CostDefinition,
    KeywordInstance,
)
from app.engine.enums import (
    AbilityType,
    ActionType,
    CardType,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.action_executor import execute_action
from app.engine.action_validator import validate_action
from app.engine.cleanup import run_cleanup

from helpers import add_unit, make_game


def _make_equip_gear(
    gs,
    player_id: str,
    name: str = "Sword",
    might_bonus: int = 3,
    equip_energy: int = 0,
    equip_power: tuple[tuple[Domain, int], ...] = ((Domain.FURY, 1),),
    exhaust_source: bool = False,
) -> CardInstance:
    """Create a gear with an Equip activated ability and add it to a player's base."""
    equip_ability = AbilityDefinition(
        ability_id=f"gear_{name.lower()}_equip",
        ability_type=AbilityType.ACTIVATED,
        cost=CostDefinition(
            energy=equip_energy,
            power=equip_power,
            exhaust_source=exhaust_source,
        ),
        timing="default",
        targets_required=1,
        target_type="friendly_unit",
        effect_ir={"type": "attach"},
    )
    defn = CardDefinition(
        card_id=f"gear_{name.lower()}",
        name=name,
        card_type=CardType.GEAR,
        domains=(Domain.FURY,),
        might_bonus=might_bonus,
        abilities=(equip_ability,),
        keywords=(KeywordInstance(keyword=Keyword.EQUIP),),
    )
    inst = CardInstance.create(defn, player_id, ZoneType.BASE)
    gs.instances[inst.instance_id] = inst
    gs.base_gear.setdefault(player_id, []).append(inst.instance_id)
    return inst


class TestEquipBasic:
    """Basic equip: activate ability, gear attaches to unit, might bonus applies."""

    def test_equip_attaches_gear_to_unit(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _make_equip_gear(gs, "p1", "B.F. Sword", might_bonus=3, equip_power=((Domain.FURY, 1),))

        # Give player resources to pay equip cost
        gs.players["p1"].rune_pool.add_power(Domain.FURY, 1)

        ability_id = gear.definition.abilities[0].ability_id
        result = validate_action(gs, "p1", ActionType.ACTIVATE_ABILITY, {
            "source_id": gear.instance_id,
            "ability_id": ability_id,
            "targets": [unit.instance_id],
        })
        assert result.ok, result.error

        logs = execute_action(gs, "p1", ActionType.ACTIVATE_ABILITY, {
            "source_id": gear.instance_id,
            "ability_id": ability_id,
            "targets": [unit.instance_id],
        })

        # Ability goes on chain; resolve it
        from app.engine.chain import pass_priority, resolve_top_item
        pass_priority(gs, "p1")
        should_resolve = pass_priority(gs, "p2")
        assert should_resolve
        resolve_logs = resolve_top_item(gs)

        # Verify attachment
        assert gear.attached_to == unit.instance_id
        assert gear.instance_id in unit.attached_cards

        # Verify might bonus after recalculation
        gs.recalculate_modifiers()
        assert unit.gear_might_bonus == 3
        assert unit.effective_might == 6  # base 3 + gear 3

    def test_equip_deducts_power_cost(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _make_equip_gear(gs, "p1", "Sword", equip_power=((Domain.FURY, 1),))

        gs.players["p1"].rune_pool.add_power(Domain.FURY, 2)

        ability_id = gear.definition.abilities[0].ability_id
        execute_action(gs, "p1", ActionType.ACTIVATE_ABILITY, {
            "source_id": gear.instance_id,
            "ability_id": ability_id,
            "targets": [unit.instance_id],
        })

        # Cost was deducted
        assert gs.players["p1"].rune_pool.power[Domain.FURY] == 1  # had 2, spent 1

    def test_equip_deducts_energy_cost(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _make_equip_gear(gs, "p1", "Axe", equip_energy=1, equip_power=((Domain.FURY, 1),))

        gs.players["p1"].rune_pool.add_energy(3)
        gs.players["p1"].rune_pool.add_power(Domain.FURY, 1)

        ability_id = gear.definition.abilities[0].ability_id
        execute_action(gs, "p1", ActionType.ACTIVATE_ABILITY, {
            "source_id": gear.instance_id,
            "ability_id": ability_id,
            "targets": [unit.instance_id],
        })

        assert gs.players["p1"].rune_pool.energy == 2  # had 3, spent 1
        assert gs.players["p1"].rune_pool.power[Domain.FURY] == 0


class TestEquipValidation:
    """Equip cost/state validation prevents illegal equips."""

    def test_cannot_equip_without_resources(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _make_equip_gear(gs, "p1", "Sword", equip_power=((Domain.FURY, 1),))

        # No resources added — pool is empty
        ability_id = gear.definition.abilities[0].ability_id
        result = validate_action(gs, "p1", ActionType.ACTIVATE_ABILITY, {
            "source_id": gear.instance_id,
            "ability_id": ability_id,
            "targets": [unit.instance_id],
        })
        assert not result.ok
        assert "afford" in result.error.lower()

    def test_cannot_equip_exhausted_gear_with_exhaust_cost(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _make_equip_gear(gs, "p1", "Sword",
                                equip_power=(), exhaust_source=True)
        gear.exhausted = True

        ability_id = gear.definition.abilities[0].ability_id
        result = validate_action(gs, "p1", ActionType.ACTIVATE_ABILITY, {
            "source_id": gear.instance_id,
            "ability_id": ability_id,
            "targets": [unit.instance_id],
        })
        assert not result.ok
        assert "exhausted" in result.error.lower()

    def test_cannot_equip_opponents_gear(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _make_equip_gear(gs, "p2", "Enemy Sword", equip_power=())

        ability_id = gear.definition.abilities[0].ability_id
        result = validate_action(gs, "p1", ActionType.ACTIVATE_ABILITY, {
            "source_id": gear.instance_id,
            "ability_id": ability_id,
            "targets": [unit.instance_id],
        })
        assert not result.ok
        assert "not your" in result.error.lower()


class TestEquipReattach:
    """Gear can be re-equipped to a different unit."""

    def test_equip_moves_gear_to_new_unit(self):
        gs = make_game()
        unit1 = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        unit2 = add_unit(gs, "p1", ZoneType.BASE, might=4, name="Knight")
        gear = _make_equip_gear(gs, "p1", "Sword", might_bonus=2, equip_power=((Domain.FURY, 1),))

        # Give enough resources for two equips
        gs.players["p1"].rune_pool.add_power(Domain.FURY, 2)

        ability_id = gear.definition.abilities[0].ability_id

        # Equip to unit1
        execute_action(gs, "p1", ActionType.ACTIVATE_ABILITY, {
            "source_id": gear.instance_id,
            "ability_id": ability_id,
            "targets": [unit1.instance_id],
        })
        from app.engine.chain import pass_priority, resolve_top_item
        pass_priority(gs, "p1")
        pass_priority(gs, "p2")
        resolve_top_item(gs)

        assert gear.attached_to == unit1.instance_id
        gs.recalculate_modifiers()
        assert unit1.gear_might_bonus == 2

        # Equip to unit2 (re-equip)
        execute_action(gs, "p1", ActionType.ACTIVATE_ABILITY, {
            "source_id": gear.instance_id,
            "ability_id": ability_id,
            "targets": [unit2.instance_id],
        })
        pass_priority(gs, "p1")
        pass_priority(gs, "p2")
        resolve_top_item(gs)

        assert gear.attached_to == unit2.instance_id
        assert gear.instance_id not in unit1.attached_cards
        assert gear.instance_id in unit2.attached_cards

        gs.recalculate_modifiers()
        assert unit1.gear_might_bonus == 0
        assert unit2.gear_might_bonus == 2


class TestEquipAndDeath:
    """Gear detaches when equipped unit dies."""

    def test_gear_detaches_to_base_when_equipped_unit_dies(self):
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")
        gear = _make_equip_gear(gs, "p1", "Sword", might_bonus=2, equip_power=())

        # Manually attach (skip ability resolution)
        gear.attached_to = unit.instance_id
        unit.attached_cards.append(gear.instance_id)
        gs.base_gear["p1"].remove(gear.instance_id)

        # Deal lethal damage
        unit.damage = 5

        logs = run_cleanup(gs)

        # Gear should be detached and back in base
        assert gear.attached_to is None
        assert gear.zone == ZoneType.BASE
        assert gear.instance_id in gs.base_gear["p1"]
        assert unit.zone == ZoneType.TRASH


class TestEquipPipeline:
    """Card pipeline generates correct Equip abilities from card text."""

    def test_pipeline_parses_equip_c(self):
        from app.engine.card_pipeline import _parse_abilities, _extract_keywords

        text = "[Equip] [C] ([C]: Attach this to a unit you control.)"
        kws, clean = _extract_keywords(text)
        abilities = _parse_abilities(clean or text, text, "test-001", "B.F. Sword", "gear", kws, ["fury"])

        assert len(abilities) >= 1
        equip_ab = abilities[0]
        assert equip_ab["ability_type"] == "activated"
        assert equip_ab["targets_required"] == 1
        assert equip_ab["target_type"] == "friendly_unit"
        assert equip_ab["effect_ir"] == {"type": "attach"}
        assert equip_ab["cost"]["power"]["fury"] == 1

    def test_pipeline_parses_equip_1c(self):
        from app.engine.card_pipeline import _parse_abilities, _extract_keywords

        text = "[Equip] [1][C] ([1][C]: Attach this to a unit you control.)"
        kws, clean = _extract_keywords(text)
        abilities = _parse_abilities(clean or text, text, "test-002", "Battleaxe", "gear", kws, ["mind"])

        equip_ab = abilities[0]
        assert equip_ab["cost"]["energy"] == 1
        assert equip_ab["cost"]["power"]["mind"] == 1

    def test_pipeline_does_not_create_equip_for_non_gear(self):
        from app.engine.card_pipeline import _parse_abilities, _extract_keywords

        text = "[Equip] [C]"
        kws, clean = _extract_keywords(text)
        abilities = _parse_abilities(clean or text, text, "test-003", "Not Gear", "unit", kws, ["fury"])

        # Should NOT create an equip ability for a unit
        for ab in abilities:
            assert ab.get("effect_ir") != {"type": "attach"} or ab.get("target_type") != "friendly_unit"


class TestEquipModeAttach:
    """prim_attach works in 1-target equip mode (source=gear)."""

    def test_attach_with_one_target_uses_source_as_gear(self):
        from app.engine.effect_primitives import prim_attach

        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")

        gear_def = CardDefinition(
            card_id="gear_sword",
            name="Sword",
            card_type=CardType.GEAR,
            might_bonus=3,
        )
        gear = CardInstance.create(gear_def, "p1", ZoneType.BASE)
        gs.instances[gear.instance_id] = gear

        # 1-target mode: source IS the gear
        logs = prim_attach(gear, gs, {}, [unit.instance_id])

        assert gear.attached_to == unit.instance_id
        assert gear.instance_id in unit.attached_cards
        assert any("attached" in l.lower() for l in logs)

    def test_attach_with_two_targets_still_works(self):
        from app.engine.effect_primitives import prim_attach

        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, might=3, name="Soldier")

        gear_def = CardDefinition(
            card_id="gear_sword",
            name="Sword",
            card_type=CardType.GEAR,
            might_bonus=3,
        )
        gear = CardInstance.create(gear_def, "p1", ZoneType.BASE)
        gs.instances[gear.instance_id] = gear
        dummy_source = add_unit(gs, "p1", ZoneType.BASE, might=1, name="Dummy")

        # 2-target mode: backwards compatible
        logs = prim_attach(dummy_source, gs, {}, [gear.instance_id, unit.instance_id])

        assert gear.attached_to == unit.instance_id
        assert gear.instance_id in unit.attached_cards
