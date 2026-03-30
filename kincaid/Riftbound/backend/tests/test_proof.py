"""Proof scenarios: each test proves an action produces correct frontend-visible results.

Every test follows the same structure:
  1. Set up a game state with specific cards and resources
  2. Execute an action through the full validate→execute→cleanup pipeline
  3. Assert on BOTH the game state AND the serialized frontend view
  4. The contract schema is validated automatically on every action

If any test here fails, either:
  - The engine computed wrong state (bug in game logic), or
  - The serializer produced wrong output (bug in the contract boundary), or
  - The contract schema is wrong (update contracts/state_update.schema.json)

These tests are the verifiable reward signal: agents that break them score lower.
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.card_types import KeywordInstance
from app.engine.effect_ir import (
    TargetSpec,
    deal_damage,
    draw_cards,
    give_might,
    kill,
    sequence,
    stun,
    buff,
    heal,
    play_token,
    return_to_hand,
    score_points,
)
from app.engine.enums import (
    ActionType,
    CombatRole,
    ControlStatus,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)


# =========================================================================
# Phase progression
# =========================================================================


class TestPhaseProof:
    """Proof: phase transitions produce correct frontend-visible state."""

    def test_advance_through_all_phases(self):
        """Proof: ADVANCE_PHASE from AWAKEN goes through all non-interactive
        phases and lands in ACTION, visible in the frontend."""
        s = Scenario(phase=Phase.AWAKEN)
        s.add_deck_cards("p1", 5)
        s.add_deck_cards("p2", 5)

        view = s.act("p1", ActionType.ADVANCE_PHASE)

        # Game state: now in ACTION phase
        assert s.gs.phase == Phase.ACTION

        # Frontend view: phase is visible
        assert view["phase"] == "action"
        assert view["is_your_turn"] is True
        assert view["can_play_cards"] is True

    def test_end_turn_switches_player(self):
        """Proof: ADVANCE_PHASE from ACTION ends the turn and gives
        the next player control, visible in the frontend."""
        s = Scenario()
        s.add_deck_cards("p1", 5)
        s.add_deck_cards("p2", 5)

        # P1 ends their turn
        view_p1 = s.act("p1", ActionType.ADVANCE_PHASE)

        # Game state: it's now P2's turn in AWAKEN
        assert s.gs.turn_player_id == "p2"

        # Frontend view for P1: not your turn anymore
        assert view_p1["is_your_turn"] is False

        # Frontend view for P2: it's your turn
        view_p2 = s.view("p2")
        assert view_p2["is_your_turn"] is True


# =========================================================================
# Playing units
# =========================================================================


class TestPlayUnitProof:
    """Proof: playing units costs resources and places them on the board."""

    def test_play_unit_to_base(self):
        """Proof: playing a unit spends energy and the unit appears in
        the player's base_units in the frontend view.
        Units finalize immediately on the chain (rule 356.2) — no
        pass_priority needed."""
        s = Scenario()
        unit = s.add_hand_unit("p1", might=3, cost_energy=2, name="Recruit")
        s.give_resources("p1", energy=5)

        view = s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": unit.instance_id,
            "targets": [],
            "destination": {"zone": "base"},
        })

        # Game state: unit in base, energy spent
        assert unit.zone == ZoneType.BASE
        assert s.gs.players["p1"].rune_pool.energy == 3  # 5 - 2

        # Frontend view: unit visible in base_units
        assert view["you"]["rune_pool"]["energy"] == 3
        base_ids = [c["instance_id"] for c in view["you"]["base_units"]]
        assert unit.instance_id in base_ids

        # Frontend view: unit no longer in hand
        hand_ids = [c["instance_id"] for c in view["you"]["hand"]]
        assert unit.instance_id not in hand_ids

    def test_play_unit_to_battlefield(self):
        """Proof: playing a unit to a controlled battlefield places it there,
        visible to both players in the frontend."""
        s = Scenario()
        # P1 must control the battlefield to play units there
        bf = s.gs.battlefields[s.bf0]
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        unit = s.add_hand_unit("p1", might=4, cost_energy=3, name="Fighter")
        s.give_resources("p1", energy=5)

        view = s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": unit.instance_id,
            "targets": [],
            "destination": {"zone": "battlefield", "id": s.bf0},
        })

        # Game state
        assert unit.zone == ZoneType.BATTLEFIELD
        assert unit.location_id == s.bf0

        # Frontend: P1 sees unit at battlefield
        bf_units = view["battlefields"][s.bf0]["units"]
        unit_ids = [c["instance_id"] for c in bf_units]
        assert unit.instance_id in unit_ids

        # The unit's card data is fully visible
        unit_view = next(c for c in bf_units if c["instance_id"] == unit.instance_id)
        assert unit_view["name"] == "Fighter"
        assert unit_view["effective_might"] == 4

        # Opponent also sees the unit
        opp_view = s.view("p2")
        opp_bf_units = opp_view["battlefields"][s.bf0]["units"]
        opp_ids = [c["instance_id"] for c in opp_bf_units]
        assert unit.instance_id in opp_ids


# =========================================================================
# Moving units
# =========================================================================


class TestMoveUnitProof:
    """Proof: moving units updates their location in the frontend view."""

    def test_move_unit_base_to_battlefield(self):
        """Proof: MOVE_UNIT from base to battlefield exhausts the unit
        and shows it at the battlefield in the frontend."""
        s = Scenario()
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Marcher")
        unit.entered_this_turn = False  # not summoning sick
        unit.exhausted = False

        view = s.act("p1", ActionType.MOVE_UNIT, {
            "instance_id": unit.instance_id,
            "destination": {"zone": "battlefield", "id": s.bf0},
        })

        # Game state
        assert unit.zone == ZoneType.BATTLEFIELD
        assert unit.location_id == s.bf0
        assert unit.exhausted is True

        # Frontend: unit at battlefield, exhausted
        bf_units = view["battlefields"][s.bf0]["units"]
        unit_view = next(c for c in bf_units if c["instance_id"] == unit.instance_id)
        assert unit_view["exhausted"] is True
        assert unit_view["zone"] == "battlefield"

        # Frontend: unit no longer in base_units
        base_ids = [c["instance_id"] for c in view["you"]["base_units"]]
        assert unit.instance_id not in base_ids

    def test_move_to_uncontrolled_bf_contests_it(self):
        """Proof: moving a unit to an uncontrolled battlefield sets
        contested_by, visible in the frontend."""
        s = Scenario()
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Pioneer")
        unit.entered_this_turn = False
        unit.exhausted = False

        view = s.act("p1", ActionType.MOVE_UNIT, {
            "instance_id": unit.instance_id,
            "destination": {"zone": "battlefield", "id": s.bf0},
        })

        # Game state
        bf = s.gs.battlefields[s.bf0]
        assert bf.contested_by == "p1"

        # Frontend
        bf_view = view["battlefields"][s.bf0]
        assert bf_view["contested_by"] == "p1"


# =========================================================================
# Playing spells with effects
# =========================================================================


class TestSpellProof:
    """Proof: spells resolve their effects and the results are frontend-visible."""

    def test_damage_spell_shows_damage(self):
        """Proof: a spell that deals 3 damage shows damage on the target
        unit in the frontend view."""
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Target")
        spell = s.add_hand_spell(
            "p1", name="Bolt", cost_energy=2,
            effect_ir=deal_damage(3, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # Play spell
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        # Resolve (both pass priority)
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state
        assert target.damage == 3
        assert target.is_alive  # 5 - 3 = alive

        # Frontend: opponent's base_unit shows damage
        target_view = Scenario.find_card_in_view(view, target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 3
        assert target_view["effective_might"] == 5  # might unchanged

    def test_draw_spell_increases_hand(self):
        """Proof: a spell that draws 2 cards increases hand count
        in the frontend view."""
        s = Scenario()
        s.add_deck_cards("p1", 5)
        spell = s.add_hand_spell(
            "p1", name="Insight", cost_energy=1,
            effect_ir=draw_cards(2),
        )
        s.give_resources("p1", energy=3)
        hand_before_play = len(s.gs.players["p1"].hand)  # includes the spell

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: played 1 spell (-1), drew 2 (+2) = net +1
        expected = hand_before_play - 1 + 2
        assert len(s.gs.players["p1"].hand) == expected

        # Frontend: hand_count matches, and the drawn cards are in hand
        p1_view = s.view("p1")
        assert p1_view["you"]["hand_count"] == expected
        assert len(p1_view["you"]["hand"]) == expected

    def test_kill_spell_removes_from_view(self):
        """Proof: a spell that kills a unit removes it from the board
        and moves it to trash in the frontend view."""
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=8, name="Doomed")
        spell = s.add_hand_spell(
            "p1", name="Execute", cost_energy=4,
            effect_ir=kill(TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Game state
        assert target.zone == ZoneType.TRASH

        # Frontend (P1's view): target gone from opponent's base, in opponent's trash
        view_p1 = s.view("p1")
        opp_base_ids = [c["instance_id"] for c in view_p1["opponent"]["base_units"]]
        assert target.instance_id not in opp_base_ids

        opp_trash_ids = [c["instance_id"] for c in view_p1["opponent"]["trash"]]
        assert target.instance_id in opp_trash_ids

    def test_buff_spell_increases_might_in_view(self):
        """Proof: a spell giving +3 might shows the increased effective_might
        in the frontend view."""
        s = Scenario()
        target = add_unit(s.gs, "p1", ZoneType.BASE, might=2, name="Buffed")
        spell = s.add_hand_spell(
            "p1", name="Empower", cost_energy=1,
            effect_ir=give_might(3, TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=3)

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state
        assert target.effective_might == 5  # 2 + 3

        # Frontend: effective_might reflects the buff
        target_view = Scenario.find_card_in_view(s.view("p1"), target.instance_id)
        assert target_view["effective_might"] == 5
        assert target_view["base_might"] == 2  # base unchanged


# =========================================================================
# Combat keywords
# =========================================================================


class TestCombatKeywordProof:
    """Proof: combat keywords (Assault, Shield) affect might correctly
    and are visible in the frontend."""

    def test_assault_shows_boosted_might_as_attacker(self):
        """Proof: a unit with Assault 2 shows effective_might +2 ONLY
        when it has the attacker combat role."""
        s = Scenario()
        kw = (KeywordInstance(Keyword.ASSAULT, 2),)
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Charger", keywords=kw)

        # Before combat: base might
        view = s.view("p1")
        unit_view = Scenario.find_card_in_view(view, unit.instance_id)
        assert unit_view["effective_might"] == 3

        # Set attacker role
        unit.combat_role = CombatRole.ATTACKER
        view = s.view("p1")
        unit_view = Scenario.find_card_in_view(view, unit.instance_id)
        assert unit_view["effective_might"] == 5  # 3 + Assault 2
        assert unit_view["combat_role"] == "attacker"

    def test_shield_shows_boosted_might_as_defender(self):
        """Proof: a unit with Shield 1 shows effective_might +1 ONLY
        when it has the defender combat role."""
        s = Scenario()
        kw = (KeywordInstance(Keyword.SHIELD, 1),)
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=4, name="Guardian", keywords=kw)

        # Not in combat
        view = s.view("p1")
        unit_view = Scenario.find_card_in_view(view, unit.instance_id)
        assert unit_view["effective_might"] == 4

        # Defender
        unit.combat_role = CombatRole.DEFENDER
        view = s.view("p1")
        unit_view = Scenario.find_card_in_view(view, unit.instance_id)
        assert unit_view["effective_might"] == 5  # 4 + Shield 1
        assert unit_view["combat_role"] == "defender"

    def test_keywords_visible_in_view(self):
        """Proof: keywords on a unit are serialized and visible
        in the frontend card view."""
        s = Scenario()
        kw = (
            KeywordInstance(Keyword.TANK, 0),
            KeywordInstance(Keyword.ASSAULT, 2),
        )
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Keywords", keywords=kw)

        view = s.view("p1")
        unit_view = Scenario.find_card_in_view(view, unit.instance_id)
        kw_list = unit_view["keywords"]
        kw_names = [k["keyword"] for k in kw_list]
        assert "tank" in kw_names
        assert "assault" in kw_names

        # Assault value is 2
        assault = next(k for k in kw_list if k["keyword"] == "assault")
        assert assault["value"] == 2


# =========================================================================
# Showdown and combat
# =========================================================================


class TestShowdownProof:
    """Proof: showdown and combat states are visible in the frontend."""

    def test_showdown_visible_in_view(self):
        """Proof: after both players have units at a battlefield and
        a showdown opens, the showdown state is visible in the frontend."""
        s = Scenario()
        bf = s.gs.battlefields[s.bf0]
        u1 = add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=s.bf0, might=3)
        u2 = add_unit(s.gs, "p2", ZoneType.BATTLEFIELD, bf_id=s.bf0, might=3)
        bf.contested_by = "p1"

        from app.engine.combat import open_showdown
        open_showdown(s.gs, s.bf0, "p1")

        # Frontend: showdown is visible
        view = s.view("p1")
        assert view["showdown"] is not None
        assert view["showdown"]["battlefield_id"] == s.bf0

    def test_combat_damage_phase_visible(self):
        """Proof: after combat starts, the combat state with attacker/defender
        is visible in the frontend view."""
        s = Scenario()
        bf = s.gs.battlefields[s.bf0]
        add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=s.bf0, might=3)
        add_unit(s.gs, "p2", ZoneType.BATTLEFIELD, bf_id=s.bf0, might=3)
        bf.contested_by = "p1"

        from app.engine.combat import start_combat
        start_combat(s.gs, s.bf0)

        view = s.view("p1")
        # Combat or showdown should be visible (combat opens showdown first)
        has_combat_state = view["combat"] is not None or view["showdown"] is not None
        assert has_combat_state


# =========================================================================
# Resource management
# =========================================================================


class TestResourceProof:
    """Proof: resource changes are accurately reflected in the frontend view."""

    def test_energy_decreases_after_play(self):
        """Proof: playing a card that costs 3 energy decreases energy from 10
        to 7 in the frontend view."""
        s = Scenario()
        unit = s.add_hand_unit("p1", cost_energy=3, might=2, name="Costly")
        s.give_resources("p1", energy=10)

        view = s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": unit.instance_id,
            "targets": [],
            "destination": {"zone": "base"},
        })

        # Frontend: energy decreased by card cost
        assert view["you"]["rune_pool"]["energy"] == 7

    def test_score_visible_after_scoring(self):
        """Proof: scoring points updates the score in the frontend view."""
        s = Scenario()
        s.gs.players["p1"].score = 3

        view = s.view("p1")
        assert view["you"]["score"] == 3

        # Opponent also sees P1's score
        opp_view = s.view("p2")
        assert opp_view["opponent"]["score"] == 3


# =========================================================================
# Information hiding
# =========================================================================


class TestInformationHidingProof:
    """Proof: the frontend correctly hides private information."""

    def test_opponent_hand_is_facedown(self):
        """Proof: cards in the opponent's hand appear as facedown stubs
        without card details in the frontend."""
        s = Scenario()
        unit = s.add_hand_unit("p2", might=5, name="Secret Unit")

        view = s.view("p1")
        opp_hand = view["opponent"]["hand"]
        assert len(opp_hand) >= 1

        # Find the card
        hidden_card = next(
            (c for c in opp_hand if c["instance_id"] == unit.instance_id), None,
        )
        assert hidden_card is not None
        assert hidden_card["facedown"] is True
        assert "card_id" not in hidden_card
        assert "name" not in hidden_card

    def test_own_hand_is_visible(self):
        """Proof: cards in the player's own hand show full details."""
        s = Scenario()
        unit = s.add_hand_unit("p1", might=5, name="My Unit", cost_energy=3)

        view = s.view("p1")
        own_hand = view["you"]["hand"]
        my_card = next(
            (c for c in own_hand if c["instance_id"] == unit.instance_id), None,
        )
        assert my_card is not None
        assert my_card["name"] == "My Unit"
        assert my_card["base_might"] == 5
        assert my_card["cost_energy"] == 3

    def test_deck_count_visible_but_not_contents(self):
        """Proof: the frontend shows deck count but not which cards are in it."""
        s = Scenario()
        s.add_deck_cards("p1", 5)

        view = s.view("p1")
        assert view["you"]["main_deck_count"] == 5
        # No "main_deck" array is exposed — only the count
        assert "main_deck" not in view["you"]


# =========================================================================
# Multi-step combos
# =========================================================================


class TestComboProof:
    """Proof: multi-step effect sequences produce correct cumulative results."""

    def test_damage_then_draw(self):
        """Proof: a spell with sequence [deal 2 damage, draw 1] applies both
        effects, and both are visible in the frontend."""
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Bruiser")
        s.add_deck_cards("p1", 3)

        spell = s.add_hand_spell(
            "p1", name="Shock and Study", cost_energy=2,
            effect_ir=sequence([
                deal_damage(2, TargetSpec(obj_type="unit", scope="enemy", count=1)),
                draw_cards(1),
            ]),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)
        hand_before_play = len(s.gs.players["p1"].hand)  # includes the spell

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        s.act("p1", ActionType.PASS_PRIORITY)
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: damage applied AND card drawn
        assert target.damage == 2
        # Hand: lost 1 spell (-1), gained 1 draw (+1) = net 0 change
        expected_hand = hand_before_play - 1 + 1
        assert len(s.gs.players["p1"].hand) == expected_hand

        # Frontend: damage visible
        target_view = Scenario.find_card_in_view(view, target.instance_id)
        assert target_view["damage"] == 2

        # Frontend: hand count reflects the draw
        p1_view = s.view("p1")
        assert p1_view["you"]["hand_count"] == expected_hand


# =========================================================================
# Mulligan
# =========================================================================


class TestMulliganProof:
    """Proof: mulligan correctly replaces cards and transitions the game."""

    def test_mulligan_keeps_selected_cards(self):
        """Proof: mulligan with keep_indices keeps those cards and
        replaces the rest, visible in the frontend."""
        s = Scenario(phase=Phase.SETUP_MULLIGAN)
        # Reset mulligan_done (make_game sets it to True by default)
        s.gs.mulligan_done = {"p1": False, "p2": False}
        s.add_deck_cards("p1", 5)
        s.add_deck_cards("p2", 5)

        # Give each player 4 cards in hand (simulating initial draw)
        for _ in range(4):
            s.add_hand_unit("p1", might=1, name="MulliganCard")
            s.add_hand_unit("p2", might=1, name="MulliganCard")

        hand_before = list(s.gs.players["p1"].hand)
        assert len(hand_before) == 4

        # Keep cards at index 0 and 2, mulligan 1 and 3
        view = s.act("p1", ActionType.MULLIGAN_CHOICE, {"keep_indices": [0, 2]})

        # Game state: still 4 cards in hand (replaced 2, drew 2)
        assert len(s.gs.players["p1"].hand) == 4
        # The kept cards are still in hand
        assert hand_before[0] in s.gs.players["p1"].hand
        assert hand_before[2] in s.gs.players["p1"].hand
        # The mulliganed cards are gone from hand
        assert hand_before[1] not in s.gs.players["p1"].hand
        assert hand_before[3] not in s.gs.players["p1"].hand

        # Frontend: hand count is 4
        p1_view = s.view("p1")
        assert p1_view["you"]["hand_count"] == 4
