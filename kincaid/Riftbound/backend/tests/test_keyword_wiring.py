"""Tests for keyword wiring: verifies each keyword's runtime logic is
connected to the correct engine hook point.

Tests cover:
  - Vision:       fires on unit/gear entry, reveals top card
  - Quick-Draw:   auto-attaches gear to a unit on play
  - Weaponmaster: auto-attaches equipment on unit entry (with cost discount)
  - Hidden:       mark_hidden_ready at start of turn
  - Backline:     cannot be designated as attacker in combat
  - Ganking:      unit can move battlefield-to-battlefield
  - Hunt:         must-attack detection
  - Deflect:      extra cost for targeting
  - Unique:       prevents duplicate play
  - Ambush:       grants Reaction timing (play in Closed State)
  - Predict:      reveals top N cards of deck
  - Level:        XP tracking (add_xp, get_xp, check_level_up)
  - Assault/Shield: might bonuses during combat
  - Mighty:       threshold check (might >= 5)
  - Repeat:       spell executes again after paying repeat cost
  - Temporary:    unit dies at start of Beginning Phase
"""

from __future__ import annotations

import pytest
from helpers import add_unit, make_game, make_spell_def, make_unit_def

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
    CombatRole,
    ControlStatus,
    Domain,
    Keyword,
    Phase,
    TurnState,
    ZoneType,
)
from app.engine.game_state import BattlefieldState, GameState
from app.engine.keywords import (
    add_xp,
    apply_quick_draw_auto_attach,
    apply_vision,
    apply_vision_recycle,
    can_play_in_state,
    check_backline_protection,
    check_level_up,
    check_mighty,
    check_unique_violation,
    get_deflect_cost,
    get_hunt_units_that_must_attack,
    get_predict_count,
    get_shield_bonus,
    get_assault_bonus,
    get_xp,
    mark_hidden_ready,
    must_attack_hunt,
    can_gank_move,
    can_hide_card,
    process_predict_on_play,
    process_weaponmaster_on_enter,
    should_die_temporary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gear_def(
    name: str = "Test Gear",
    might_bonus: int = 2,
    keywords: tuple[KeywordInstance, ...] = (),
    tags: tuple[str, ...] = (),
    abilities: tuple = (),
) -> CardDefinition:
    return CardDefinition(
        card_id=f"test_{name.lower().replace(' ', '_')}",
        name=name,
        card_type=CardType.GEAR,
        might_bonus=might_bonus,
        keywords=keywords,
        tags=tags,
        abilities=abilities,
    )


def _add_gear(
    gs: GameState,
    player_id: str,
    name: str = "Gear",
    might_bonus: int = 2,
    keywords: tuple[KeywordInstance, ...] = (),
    tags: tuple[str, ...] = (),
    abilities: tuple = (),
) -> CardInstance:
    defn = _make_gear_def(name=name, might_bonus=might_bonus, keywords=keywords, tags=tags, abilities=abilities)
    inst = CardInstance.create(defn, player_id, ZoneType.BASE)
    inst.location_id = player_id
    gs.instances[inst.instance_id] = inst
    gs.base_gear.setdefault(player_id, []).append(inst.instance_id)
    return inst


def _add_deck_card(gs: GameState, player_id: str, name: str = "Deck Card") -> CardInstance:
    defn = make_unit_def(name=name)
    inst = CardInstance.create(defn, player_id, ZoneType.MAIN_DECK)
    gs.instances[inst.instance_id] = inst
    gs.players[player_id].main_deck.append(inst.instance_id)
    return inst


# ===========================================================================
# Vision
# ===========================================================================


class TestVision:
    def test_vision_reveals_top_card(self):
        """Vision on a unit should reveal the top card of the controller's deck."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Seer",
            keywords=(KeywordInstance(keyword=Keyword.VISION),),
        )
        deck_card = _add_deck_card(gs, "p1", "Secret Card")

        logs = apply_vision(unit, gs)
        assert any("Secret Card" in msg for msg in logs)

    def test_vision_empty_deck(self):
        """Vision with empty deck should log a message and not crash."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Seer",
            keywords=(KeywordInstance(keyword=Keyword.VISION),),
        )

        logs = apply_vision(unit, gs)
        assert any("empty" in msg for msg in logs)

    def test_vision_no_keyword_returns_empty(self):
        """Unit without Vision should produce no logs."""
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, name="Regular")
        _add_deck_card(gs, "p1")

        logs = apply_vision(unit, gs)
        assert logs == []

    def test_vision_multiple_instances(self):
        """Multiple Vision instances should each trigger separately (rule 743.2)."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Double Seer",
            keywords=(
                KeywordInstance(keyword=Keyword.VISION),
                KeywordInstance(keyword=Keyword.VISION),
            ),
        )
        _add_deck_card(gs, "p1", "Top Card")

        logs = apply_vision(unit, gs)
        # Two vision instances, both see the same top card (rule 743.2.b)
        vision_msgs = [m for m in logs if "sees" in m]
        assert len(vision_msgs) == 2

    def test_vision_recycle_moves_top_to_bottom(self):
        """Vision recycle should move the top card to the bottom of the deck."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Seer",
            keywords=(KeywordInstance(keyword=Keyword.VISION),),
        )
        card1 = _add_deck_card(gs, "p1", "First")
        card2 = _add_deck_card(gs, "p1", "Second")

        logs = apply_vision_recycle(unit, gs, do_recycle=True)
        assert any("recycled" in msg for msg in logs)
        # First card should now be at the bottom
        ps = gs.players["p1"]
        assert ps.main_deck[-1] == card1.instance_id
        assert ps.main_deck[0] == card2.instance_id


# ===========================================================================
# Quick-Draw
# ===========================================================================


class TestQuickDraw:
    def test_quick_draw_auto_attaches_to_unit(self):
        """Quick-Draw gear should auto-attach to a friendly unit on play."""
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, name="Warrior", might=3)
        gear = _add_gear(
            gs, "p1", name="Quick Blade", might_bonus=2,
            keywords=(KeywordInstance(keyword=Keyword.QUICK_DRAW),),
        )

        logs = apply_quick_draw_auto_attach(gear, gs)
        assert any("auto-attaches" in msg for msg in logs)
        assert gear.attached_to == unit.instance_id
        assert gear.instance_id in unit.attached_cards
        assert unit.gear_might_bonus == 2

    def test_quick_draw_no_unit_available(self):
        """Quick-Draw with no units should log but not crash."""
        gs = make_game()
        gear = _add_gear(
            gs, "p1", name="Quick Blade", might_bonus=2,
            keywords=(KeywordInstance(keyword=Keyword.QUICK_DRAW),),
        )

        logs = apply_quick_draw_auto_attach(gear, gs)
        assert any("no unit" in msg for msg in logs)
        assert gear.attached_to is None

    def test_quick_draw_non_keyword_noop(self):
        """Gear without Quick-Draw should not auto-attach."""
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, name="Warrior")
        gear = _add_gear(gs, "p1", name="Normal Sword", might_bonus=2)

        logs = apply_quick_draw_auto_attach(gear, gs)
        assert logs == []
        assert gear.attached_to is None

    def test_quick_draw_removes_from_base_gear(self):
        """Quick-Draw attach should remove gear from base_gear list."""
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, name="Warrior")
        gear = _add_gear(
            gs, "p1", name="Quick Blade", might_bonus=1,
            keywords=(KeywordInstance(keyword=Keyword.QUICK_DRAW),),
        )

        assert gear.instance_id in gs.base_gear["p1"]
        apply_quick_draw_auto_attach(gear, gs)
        assert gear.instance_id not in gs.base_gear["p1"]


# ===========================================================================
# Weaponmaster
# ===========================================================================


class TestWeaponmaster:
    def test_weaponmaster_attaches_equipment(self):
        """Weaponmaster unit should auto-attach equipment on entry."""
        gs = make_game()
        equip_cost = CostDefinition(energy=1, power=((Domain.FURY, 1),))
        gear = _add_gear(
            gs, "p1", name="War Axe", might_bonus=3,
            keywords=(KeywordInstance(keyword=Keyword.EQUIP),),
            tags=("equipment",),
            abilities=(
                AbilityDefinition(
                    ability_id="war_axe_equip",
                    ability_type=AbilityType.ACTIVATED,
                    cost=equip_cost,
                ),
            ),
        )
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Blademaster",
            keywords=(KeywordInstance(keyword=Keyword.WEAPONMASTER),),
        )
        # Give player resources to pay discounted equip cost
        gs.players["p1"].rune_pool.energy = 5
        gs.players["p1"].rune_pool.power[Domain.FURY] = 3

        logs = process_weaponmaster_on_enter(unit, gs)
        assert any("attaches" in msg for msg in logs)
        assert gear.attached_to == unit.instance_id
        assert gear.instance_id in unit.attached_cards
        assert unit.gear_might_bonus == 3

    def test_weaponmaster_no_equipment(self):
        """Weaponmaster with no equipment available should log appropriately."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Blademaster",
            keywords=(KeywordInstance(keyword=Keyword.WEAPONMASTER),),
        )

        logs = process_weaponmaster_on_enter(unit, gs)
        assert any("no equipment" in msg for msg in logs)

    def test_weaponmaster_cannot_afford(self):
        """Weaponmaster should not attach if controller can't pay equip cost."""
        gs = make_game()
        equip_cost = CostDefinition(energy=5, power=((Domain.FURY, 3),))
        gear = _add_gear(
            gs, "p1", name="War Axe", might_bonus=3,
            keywords=(KeywordInstance(keyword=Keyword.EQUIP),),
            tags=("equipment",),
            abilities=(
                AbilityDefinition(
                    ability_id="war_axe_equip",
                    ability_type=AbilityType.ACTIVATED,
                    cost=equip_cost,
                ),
            ),
        )
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Blademaster",
            keywords=(KeywordInstance(keyword=Keyword.WEAPONMASTER),),
        )
        # Give no resources
        gs.players["p1"].rune_pool.energy = 0

        logs = process_weaponmaster_on_enter(unit, gs)
        assert any("cannot afford" in msg for msg in logs)
        assert gear.attached_to is None

    def test_weaponmaster_not_triggered_without_keyword(self):
        """Unit without Weaponmaster should not trigger equipment search."""
        gs = make_game()
        gear = _add_gear(gs, "p1", name="War Axe", tags=("equipment",))
        unit = add_unit(gs, "p1", ZoneType.BASE, name="Regular")

        logs = process_weaponmaster_on_enter(unit, gs)
        assert logs == []
        assert gear.attached_to is None


# ===========================================================================
# Hidden (mark_hidden_ready)
# ===========================================================================


class TestHiddenReady:
    def test_mark_hidden_ready_at_turn_start(self):
        """Facedown cards become ready at the start of the controller's turn."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # Create a facedown card at a battlefield
        defn = make_unit_def(name="Hidden Unit", keywords=(KeywordInstance(keyword=Keyword.HIDDEN),))
        card = CardInstance.create(defn, "p1", ZoneType.FACEDOWN_ZONE)
        card.hidden_at_battlefield = bf_id
        card.hidden_ready = False
        card.facedown = True
        gs.instances[card.instance_id] = card
        bf.facedown_card = card.instance_id
        bf.controller_id = "p1"
        bf.control_status = ControlStatus.CONTROLLED

        # It's p1's turn
        gs.turn_player_id = "p1"

        logs = mark_hidden_ready(gs)
        assert card.hidden_ready is True
        assert any("ready" in msg for msg in logs)

    def test_mark_hidden_ready_not_opponents_turn(self):
        """Facedown cards should NOT become ready on opponent's turn."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        defn = make_unit_def(name="Hidden Unit", keywords=(KeywordInstance(keyword=Keyword.HIDDEN),))
        card = CardInstance.create(defn, "p1", ZoneType.FACEDOWN_ZONE)
        card.hidden_at_battlefield = bf_id
        card.hidden_ready = False
        card.facedown = True
        gs.instances[card.instance_id] = card
        bf.facedown_card = card.instance_id
        bf.controller_id = "p1"
        bf.control_status = ControlStatus.CONTROLLED

        # It's p2's turn
        gs.turn_player_id = "p2"

        logs = mark_hidden_ready(gs)
        assert card.hidden_ready is False


# ===========================================================================
# Backline
# ===========================================================================


class TestBacklineWiring:
    def test_backline_units_protected_by_non_backline(self):
        """Backline unit should be protected when non-Backline units exist."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        tank = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, name="Tank", might=4)
        backline = add_unit(
            gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, name="Archer",
            keywords=(KeywordInstance(keyword=Keyword.BACKLINE),),
        )

        all_friendly = [tank, backline]
        assert check_backline_protection(backline, all_friendly) is True
        assert check_backline_protection(tank, all_friendly) is False

    def test_backline_not_designated_attacker(self):
        """Backline unit should not be designated as attacker in combat."""
        from app.engine.combat import start_combat

        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.control_status = ControlStatus.CONTESTED
        bf.contested_by = "p1"

        backline = add_unit(
            gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, name="Backline Unit",
            keywords=(KeywordInstance(keyword=Keyword.BACKLINE),), might=3,
        )
        attacker = add_unit(
            gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, name="Attacker", might=4,
        )
        defender = add_unit(
            gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, name="Defender", might=5,
        )

        start_combat(gs, bf_id)

        # Backline unit should have NONE role, not ATTACKER
        assert backline.combat_role == CombatRole.NONE
        assert attacker.combat_role == CombatRole.ATTACKER


# ===========================================================================
# Ganking
# ===========================================================================


class TestGanking:
    def test_ganking_allows_bf_to_bf_move(self):
        """Unit with Ganking should be allowed to move between battlefields."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BATTLEFIELD, name="Ganker",
            keywords=(KeywordInstance(keyword=Keyword.GANKING),),
        )
        assert can_gank_move(unit) is True

    def test_no_ganking_blocks_bf_to_bf_move(self):
        """Unit without Ganking should not be allowed bf-to-bf move."""
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BATTLEFIELD, name="Regular")
        assert can_gank_move(unit) is False

    def test_ganking_wired_in_validator(self):
        """Action validator should reject bf-to-bf moves without Ganking."""
        from app.engine.action_validator import validate_action
        from app.engine.enums import ActionType

        gs = make_game()
        bf_ids = list(gs.battlefields.keys())
        bf0 = gs.battlefields[bf_ids[0]]
        bf0.controller_id = "p1"
        bf0.control_status = ControlStatus.CONTROLLED

        # Place a regular unit at bf0
        unit = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_ids[0], name="Regular")

        # Try to move it to bf1 — should fail (no Ganking)
        result = validate_action(gs, "p1", ActionType.MOVE_UNIT, {
            "instance_id": unit.instance_id,
            "destination": {"zone": "battlefield", "id": bf_ids[1]},
        })
        assert not result.ok
        assert "Ganking" in result.error


# ===========================================================================
# Hunt
# ===========================================================================


class TestHunt:
    def test_must_attack_hunt(self):
        """Unit with Hunt must attack if able."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Hunter",
            keywords=(KeywordInstance(keyword=Keyword.HUNT),),
        )
        assert must_attack_hunt(unit) is True

    def test_get_hunt_units_that_must_attack(self):
        """get_hunt_units_that_must_attack returns all ready Hunt units."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        hunt_unit = add_unit(
            gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, name="Hunter",
            keywords=(KeywordInstance(keyword=Keyword.HUNT),),
        )
        normal = add_unit(gs, "p1", ZoneType.BASE, name="Regular")

        hunt_units = get_hunt_units_that_must_attack(gs, "p1")
        ids = [u.instance_id for u in hunt_units]
        assert hunt_unit.instance_id in ids
        assert normal.instance_id not in ids

    def test_exhausted_hunt_unit_excluded(self):
        """Exhausted Hunt units should not be required to attack."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        hunt_unit = add_unit(
            gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, name="Hunter",
            keywords=(KeywordInstance(keyword=Keyword.HUNT),),
        )
        hunt_unit.exhausted = True

        hunt_units = get_hunt_units_that_must_attack(gs, "p1")
        assert hunt_unit.instance_id not in [u.instance_id for u in hunt_units]


# ===========================================================================
# Deflect
# ===========================================================================


class TestDeflect:
    def test_deflect_extra_cost(self):
        """Unit with Deflect should impose extra power cost."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Deflector",
            keywords=(KeywordInstance(keyword=Keyword.DEFLECT, value=2),),
        )
        assert get_deflect_cost(unit) == 2

    def test_deflect_zero_without_keyword(self):
        """Unit without Deflect should have zero extra cost."""
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, name="Regular")
        assert get_deflect_cost(unit) == 0


# ===========================================================================
# Unique
# ===========================================================================


class TestUnique:
    def test_unique_violation_detected(self):
        """Playing a second copy of a Unique card should be flagged."""
        gs = make_game()
        defn = make_unit_def(
            card_id="unique_hero", name="Hero",
            keywords=(KeywordInstance(keyword=Keyword.UNIQUE),),
        )
        # First copy on the board
        inst1 = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[inst1.instance_id] = inst1
        gs.base_units["p1"].append(inst1.instance_id)

        # Second copy trying to play
        inst2 = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[inst2.instance_id] = inst2

        assert check_unique_violation(inst2, gs) is True

    def test_unique_ok_if_no_duplicate(self):
        """First copy of a Unique card should be allowed."""
        gs = make_game()
        defn = make_unit_def(
            card_id="unique_hero", name="Hero",
            keywords=(KeywordInstance(keyword=Keyword.UNIQUE),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[inst.instance_id] = inst

        assert check_unique_violation(inst, gs) is False

    def test_unique_different_controller_ok(self):
        """Unique check only applies to the same controller."""
        gs = make_game()
        defn = make_unit_def(
            card_id="unique_hero", name="Hero",
            keywords=(KeywordInstance(keyword=Keyword.UNIQUE),),
        )
        # p1 has a copy
        inst1 = CardInstance.create(defn, "p1", ZoneType.BASE)
        gs.instances[inst1.instance_id] = inst1
        gs.base_units["p1"].append(inst1.instance_id)

        # p2 tries to play — should be OK
        inst2 = CardInstance.create(defn, "p2", ZoneType.HAND)
        gs.instances[inst2.instance_id] = inst2

        assert check_unique_violation(inst2, gs) is False


# ===========================================================================
# Ambush
# ===========================================================================


class TestAmbushTiming:
    def test_ambush_grants_reaction_timing(self):
        """Ambush should allow playing in Closed State (like Reaction)."""
        kws = [KeywordInstance(keyword=Keyword.AMBUSH)]
        assert can_play_in_state(kws, is_showdown=False, is_closed=True) is True

    def test_ambush_allows_play_in_showdown(self):
        """Ambush should allow playing during Showdown."""
        kws = [KeywordInstance(keyword=Keyword.AMBUSH)]
        assert can_play_in_state(kws, is_showdown=True, is_closed=False) is True

    def test_no_ambush_blocked_in_closed(self):
        """Non-Ambush non-Reaction cards should NOT play in Closed State."""
        kws = []
        assert can_play_in_state(kws, is_showdown=False, is_closed=True) is False


class TestAmbushDestination:
    """Ambush relaxes unit destination rules: can play to any BF where you have units."""

    def test_ambush_to_contested_bf_with_friendly_units(self):
        """Ambush unit can be played to a contested BF where the player has units."""
        from app.engine.action_validator import validate_action

        gs = make_game()
        gs.phase = Phase.ACTION
        gs.turn_player_id = "p1"
        gs.active_player_id = "p1"
        gs.players["p1"].rune_pool.energy = 99

        bf = list(gs.battlefields.values())[0]

        # Place a friendly unit on the BF
        friendly = add_unit(gs, "p1", ZoneType.BATTLEFIELD, name="Ally")
        friendly.location_id = bf.battlefield_id
        friendly.controller_id = "p1"
        bf.units.append(friendly.instance_id)

        # Place an enemy unit (contested)
        enemy = add_unit(gs, "p2", ZoneType.BATTLEFIELD, name="Enemy")
        enemy.location_id = bf.battlefield_id
        enemy.controller_id = "p2"
        bf.units.append(enemy.instance_id)
        bf.control_status = ControlStatus.CONTESTED
        bf.controller_id = None

        # Ambush unit in hand
        ambush_unit = add_unit(
            gs, "p1", ZoneType.HAND, name="Chakram Dancer",
            keywords=(
                KeywordInstance(keyword=Keyword.AMBUSH),
                KeywordInstance(keyword=Keyword.REACTION),
            ),
        )
        gs.players["p1"].hand.append(ambush_unit.instance_id)

        result = validate_action(gs, "p1", ActionType.PLAY_CARD, {
            "instance_id": ambush_unit.instance_id,
            "destination": {"zone": "battlefield", "id": bf.battlefield_id},
        })
        assert result.ok, f"Ambush unit should be playable to contested BF: {result.error}"

    def test_ambush_rejected_to_bf_without_friendly_units(self):
        """Ambush unit cannot play to a BF where the player has NO units."""
        from app.engine.action_validator import validate_action

        gs = make_game()
        gs.phase = Phase.ACTION
        gs.turn_player_id = "p1"
        gs.active_player_id = "p1"
        gs.players["p1"].rune_pool.energy = 99

        bf = list(gs.battlefields.values())[0]

        # Only enemy units on this BF
        enemy = add_unit(gs, "p2", ZoneType.BATTLEFIELD, name="Enemy")
        enemy.location_id = bf.battlefield_id
        enemy.controller_id = "p2"
        bf.units.append(enemy.instance_id)
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p2"

        ambush_unit = add_unit(
            gs, "p1", ZoneType.HAND, name="Ambusher",
            keywords=(
                KeywordInstance(keyword=Keyword.AMBUSH),
                KeywordInstance(keyword=Keyword.REACTION),
            ),
        )
        gs.players["p1"].hand.append(ambush_unit.instance_id)

        result = validate_action(gs, "p1", ActionType.PLAY_CARD, {
            "instance_id": ambush_unit.instance_id,
            "destination": {"zone": "battlefield", "id": bf.battlefield_id},
        })
        assert not result.ok, "Ambush unit should NOT be playable to BF with no friendly units"

    def test_non_ambush_rejected_to_contested_bf(self):
        """Normal unit (no Ambush) still cannot play to a contested BF."""
        from app.engine.action_validator import validate_action

        gs = make_game()
        gs.phase = Phase.ACTION
        gs.turn_player_id = "p1"
        gs.active_player_id = "p1"
        gs.players["p1"].rune_pool.energy = 99

        bf = list(gs.battlefields.values())[0]
        bf.control_status = ControlStatus.CONTESTED
        bf.controller_id = None

        normal_unit = add_unit(gs, "p1", ZoneType.HAND, name="Normal Unit")
        gs.players["p1"].hand.append(normal_unit.instance_id)

        result = validate_action(gs, "p1", ActionType.PLAY_CARD, {
            "instance_id": normal_unit.instance_id,
            "destination": {"zone": "battlefield", "id": bf.battlefield_id},
        })
        assert not result.ok, "Normal unit should NOT be playable to contested BF"

    def test_ambush_to_controlled_bf_always_ok(self):
        """Ambush unit can play to a BF the player controls (normal rule still applies)."""
        from app.engine.action_validator import validate_action

        gs = make_game()
        gs.phase = Phase.ACTION
        gs.turn_player_id = "p1"
        gs.active_player_id = "p1"
        gs.players["p1"].rune_pool.energy = 99

        bf = list(gs.battlefields.values())[0]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        ambush_unit = add_unit(
            gs, "p1", ZoneType.HAND, name="Ambusher",
            keywords=(
                KeywordInstance(keyword=Keyword.AMBUSH),
                KeywordInstance(keyword=Keyword.REACTION),
            ),
        )
        gs.players["p1"].hand.append(ambush_unit.instance_id)

        result = validate_action(gs, "p1", ActionType.PLAY_CARD, {
            "instance_id": ambush_unit.instance_id,
            "destination": {"zone": "battlefield", "id": bf.battlefield_id},
        })
        assert result.ok, f"Ambush to controlled BF should always work: {result.error}"


# ===========================================================================
# Predict
# ===========================================================================


class TestPredict:
    def test_predict_reveals_top_cards(self):
        """Predict should reveal top N cards of the controller's deck."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Oracle",
            keywords=(KeywordInstance(keyword=Keyword.PREDICT, value=2),),
        )
        _add_deck_card(gs, "p1", "Card A")
        _add_deck_card(gs, "p1", "Card B")

        assert get_predict_count(unit) == 2

        logs = process_predict_on_play(unit, gs)
        assert any("Predict 2" in msg for msg in logs)

    def test_predict_empty_deck(self):
        """Predict with empty deck should return appropriate message."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Oracle",
            keywords=(KeywordInstance(keyword=Keyword.PREDICT, value=1),),
        )

        logs = process_predict_on_play(unit, gs)
        assert any("empty" in msg for msg in logs)

    def test_predict_zero_value_noop(self):
        """Predict with value 0 should not trigger."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Oracle",
            keywords=(KeywordInstance(keyword=Keyword.PREDICT, value=0),),
        )
        _add_deck_card(gs, "p1")

        logs = process_predict_on_play(unit, gs)
        assert logs == []


# ===========================================================================
# Level (XP tracking)
# ===========================================================================


class TestLevel:
    def test_add_xp(self):
        """add_xp should increment XP on a Level unit."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Leveler",
            keywords=(KeywordInstance(keyword=Keyword.LEVEL),),
        )

        logs = add_xp(unit, 3)
        assert any("gains 3 XP" in msg for msg in logs)
        assert get_xp(unit) == 3

    def test_add_xp_accumulates(self):
        """Multiple add_xp calls should accumulate."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Leveler",
            keywords=(KeywordInstance(keyword=Keyword.LEVEL),),
        )

        add_xp(unit, 2)
        add_xp(unit, 3)
        assert get_xp(unit) == 5

    def test_add_xp_no_level_keyword(self):
        """add_xp should do nothing if unit lacks Level keyword."""
        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, name="Regular")

        logs = add_xp(unit, 5)
        assert logs == []
        assert get_xp(unit) == 0

    def test_check_level_up(self):
        """check_level_up should return True when XP >= threshold."""
        gs = make_game()
        unit = add_unit(
            gs, "p1", ZoneType.BASE, name="Leveler",
            keywords=(KeywordInstance(keyword=Keyword.LEVEL),),
        )
        add_xp(unit, 5)

        assert check_level_up(unit, 3) is True
        assert check_level_up(unit, 5) is True
        assert check_level_up(unit, 6) is False


# ===========================================================================
# Assault / Shield (passive might bonuses)
# ===========================================================================


class TestAssaultShieldWiring:
    def test_assault_bonus_only_when_attacking(self):
        """Assault should add might only when unit is an attacker."""
        defn = make_unit_def(
            might=3,
            keywords=(KeywordInstance(keyword=Keyword.ASSAULT, value=2),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)

        inst.combat_role = CombatRole.ATTACKER
        assert get_assault_bonus(inst) == 2
        assert inst.effective_might == 5

        inst.combat_role = CombatRole.DEFENDER
        assert get_assault_bonus(inst) == 0
        assert inst.effective_might == 3

        inst.combat_role = CombatRole.NONE
        assert get_assault_bonus(inst) == 0
        assert inst.effective_might == 3

    def test_shield_bonus_only_when_defending(self):
        """Shield should add might only when unit is a defender."""
        defn = make_unit_def(
            might=3,
            keywords=(KeywordInstance(keyword=Keyword.SHIELD, value=2),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)

        inst.combat_role = CombatRole.DEFENDER
        assert get_shield_bonus(inst) == 2
        assert inst.effective_might == 5

        inst.combat_role = CombatRole.ATTACKER
        assert get_shield_bonus(inst) == 0
        assert inst.effective_might == 3

    def test_assault_default_value_is_one(self):
        """Assault with omitted value should default to 1 (rule 733.1.b.3)."""
        defn = make_unit_def(
            might=3,
            keywords=(KeywordInstance(keyword=Keyword.ASSAULT, value=0),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        inst.combat_role = CombatRole.ATTACKER
        # value=0 + default_one -> 1
        assert inst.effective_might == 4

    def test_assault_stacks(self):
        """Multiple Assault sources should sum (rule 733.2)."""
        defn = make_unit_def(
            might=2,
            keywords=(
                KeywordInstance(keyword=Keyword.ASSAULT, value=1),
                KeywordInstance(keyword=Keyword.ASSAULT, value=3),
            ),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        inst.combat_role = CombatRole.ATTACKER
        assert inst.effective_might == 6  # 2 + 1 + 3


# ===========================================================================
# Mighty
# ===========================================================================


class TestMighty:
    def test_mighty_threshold(self):
        """Unit with might >= 5 should be Mighty."""
        defn = make_unit_def(might=5)
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        assert check_mighty(inst) is True
        assert inst.is_mighty is True

    def test_not_mighty_below_threshold(self):
        """Unit with might < 5 should NOT be Mighty."""
        defn = make_unit_def(might=4)
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        assert check_mighty(inst) is False
        assert inst.is_mighty is False


# ===========================================================================
# Temporary
# ===========================================================================


class TestTemporary:
    def test_should_die_after_surviving_turn(self):
        """Temporary unit that survived a turn should die at Beginning Phase."""
        defn = make_unit_def(
            keywords=(KeywordInstance(keyword=Keyword.TEMPORARY),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BASE)
        inst.entered_this_turn = False  # survived a full turn

        assert should_die_temporary(inst) is True

    def test_should_not_die_same_turn(self):
        """Temporary unit that entered this turn should NOT die yet."""
        defn = make_unit_def(
            keywords=(KeywordInstance(keyword=Keyword.TEMPORARY),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BASE)
        inst.entered_this_turn = True

        assert should_die_temporary(inst) is False


# ===========================================================================
# Integration: Chain wiring for Vision, Quick-Draw, Weaponmaster
# ===========================================================================


class TestChainWiring:
    def test_unit_with_vision_triggers_on_entry(self):
        """When a unit with Vision enters via _finalize_permanent, Vision fires."""
        from app.engine.chain import _finalize_permanent

        gs = make_game()
        _add_deck_card(gs, "p1", "Top Card")

        defn = make_unit_def(
            name="Vision Unit", might=2,
            keywords=(KeywordInstance(keyword=Keyword.VISION),),
        )
        card = CardInstance.create(defn, "p1", ZoneType.CHAIN)
        gs.instances[card.instance_id] = card

        logs = _finalize_permanent(gs, card, "p1")
        assert any("Vision" in msg and "sees" in msg for msg in logs)

    def test_gear_with_quick_draw_auto_attaches_on_entry(self):
        """When Quick-Draw gear enters via _finalize_permanent, it auto-attaches."""
        from app.engine.chain import _finalize_permanent

        gs = make_game()
        unit = add_unit(gs, "p1", ZoneType.BASE, name="Warrior")

        defn = _make_gear_def(
            name="Quick Blade", might_bonus=2,
            keywords=(KeywordInstance(keyword=Keyword.QUICK_DRAW),),
        )
        gear = CardInstance.create(defn, "p1", ZoneType.CHAIN)
        gs.instances[gear.instance_id] = gear

        logs = _finalize_permanent(gs, gear, "p1")
        assert any("Quick-Draw" in msg and "auto-attaches" in msg for msg in logs)
        assert gear.attached_to == unit.instance_id


# ===========================================================================
# Hidden: can_hide_card validation
# ===========================================================================


class TestCanHideCard:
    def test_can_hide_on_own_turn(self):
        """Card with Hidden can be hidden on own turn at controlled battlefield."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.controller_id = "p1"
        bf.control_status = ControlStatus.CONTROLLED
        bf.facedown_card = None

        defn = make_unit_def(
            name="Hidden Unit",
            keywords=(KeywordInstance(keyword=Keyword.HIDDEN),),
        )
        card = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[card.instance_id] = card

        gs.turn_player_id = "p1"
        assert can_hide_card(card, gs) is True

    def test_cannot_hide_on_opponent_turn(self):
        """Card with Hidden cannot be hidden on opponent's turn."""
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]
        bf.controller_id = "p1"
        bf.control_status = ControlStatus.CONTROLLED

        defn = make_unit_def(
            name="Hidden Unit",
            keywords=(KeywordInstance(keyword=Keyword.HIDDEN),),
        )
        card = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[card.instance_id] = card

        gs.turn_player_id = "p2"
        assert can_hide_card(card, gs) is False

    def test_cannot_hide_without_keyword(self):
        """Card without Hidden keyword cannot be hidden."""
        gs = make_game()
        defn = make_unit_def(name="Regular Unit")
        card = CardInstance.create(defn, "p1", ZoneType.HAND)
        gs.instances[card.instance_id] = card
        gs.turn_player_id = "p1"
        assert can_hide_card(card, gs) is False
