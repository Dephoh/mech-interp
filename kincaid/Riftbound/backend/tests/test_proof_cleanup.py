"""Proof scenarios: turn-end cleanup removes dead units, clears temporary state.

Cleanup is the critical phase where:
  - Dead units (damage >= might) go to trash
  - Might modifiers clear at end of turn (Expiration Step, rule 317.3)
  - Stun clears at end of turn (Expiration Step via clear_turn_state)
  - Temporary units die at start of the next turn's Beginning Phase

Each test triggers end-of-turn via ADVANCE_PHASE from ACTION, then verifies
both internal game state and serialized frontend views.
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.card_types import KeywordInstance
from app.engine.enums import (
    ActionType,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)


# =========================================================================
# Dead unit cleanup
# =========================================================================


class TestDeadUnitCleanup:
    """Proof: units with lethal damage are removed during cleanup."""

    def test_dead_unit_cleaned_up(self):
        """Proof: a unit with damage >= might is moved to trash by cleanup,
        and disappears from the board in both game state and frontend view."""
        s = Scenario()
        # Place a unit in P1's base with lethal damage (damage >= might)
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Doomed Soldier")
        unit.damage = 3  # exactly lethal: damage == might

        # Verify unit is alive in game state before cleanup
        assert unit.zone == ZoneType.BASE
        assert unit.instance_id in s.gs.base_units["p1"]

        # Any action triggers cleanup. Pass priority is simplest.
        s.add_deck_cards("p1", 5)
        s.add_deck_cards("p2", 5)
        view = s.act("p1", ActionType.ADVANCE_PHASE)

        # Game state: unit moved to trash
        assert unit.zone == ZoneType.TRASH
        assert unit.instance_id in s.gs.players["p1"].trash
        assert unit.instance_id not in s.gs.base_units.get("p1", [])

        # Frontend view: unit appears in trash, not in base_units.
        # After ADVANCE_PHASE from ACTION, it's now P2's turn, so check
        # from P2's perspective (P1 is the opponent).
        view_p2 = s.view("p2")
        opp_base_ids = [c["instance_id"] for c in view_p2["opponent"]["base_units"]]
        assert unit.instance_id not in opp_base_ids

        opp_trash_ids = [c["instance_id"] for c in view_p2["opponent"]["trash"]]
        assert unit.instance_id in opp_trash_ids

        # Also verify from P1's own view
        view_p1 = s.view("p1")
        own_base_ids = [c["instance_id"] for c in view_p1["you"]["base_units"]]
        assert unit.instance_id not in own_base_ids

        own_trash_ids = [c["instance_id"] for c in view_p1["you"]["trash"]]
        assert unit.instance_id in own_trash_ids


# =========================================================================
# Might modifier expiration
# =========================================================================


class TestMightModifierCleanup:
    """Proof: transient might modifiers expire at end of turn."""

    def test_might_modifiers_clear_on_turn_end(self):
        """Proof: a +3 might modifier applied to a unit is visible in the
        frontend as effective_might=6, then clears when the turn ends.
        After the turn switches to P2, the unit's effective_might returns
        to its base value of 3."""
        s = Scenario()
        s.add_deck_cards("p1", 5)
        s.add_deck_cards("p2", 5)

        # Place a unit in P1's base and give it a +3 might modifier
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Buffed Warrior")
        unit.might_modifiers.append(3)

        # Verify the modifier is active in game state
        assert unit.effective_might == 6  # 3 base + 3 modifier

        # Frontend view before turn end: effective_might includes modifier
        view_before = s.view("p1")
        unit_view = Scenario.find_card_in_view(view_before, unit.instance_id)
        assert unit_view is not None
        assert unit_view["effective_might"] == 6

        # End the turn (ADVANCE_PHASE from ACTION)
        # This triggers _execute_end_of_turn which calls clear_turn_state,
        # then _start_next_turn switches to P2.
        s.act("p1", ActionType.ADVANCE_PHASE)

        # Game state: might_modifiers cleared by clear_turn_state
        assert unit.effective_might == 3  # back to base
        assert len(unit.might_modifiers) == 0

        # Frontend view after turn end: effective_might is back to base
        # Check from P2's perspective (P1's unit is the opponent's)
        view_p2 = s.view("p2")
        unit_view_after = Scenario.find_card_in_view(view_p2, unit.instance_id)
        assert unit_view_after is not None
        assert unit_view_after["effective_might"] == 3

        # Also verify from P1's own view
        view_p1 = s.view("p1")
        unit_view_p1 = Scenario.find_card_in_view(view_p1, unit.instance_id)
        assert unit_view_p1 is not None
        assert unit_view_p1["effective_might"] == 3


# =========================================================================
# Stun expiration
# =========================================================================


class TestStunCleanup:
    """Proof: stun clears at end of turn via clear_turn_state."""

    def test_stun_clears_on_turn_end(self):
        """Proof: a stunned unit shows stunned=True in the frontend,
        then after the turn ends, stunned=False in the new turn's view."""
        s = Scenario()
        s.add_deck_cards("p1", 5)
        s.add_deck_cards("p2", 5)

        # Place a unit and stun it
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=4, name="Stunned Guard")
        unit.stunned = True

        # Game state: unit is stunned
        assert unit.stunned is True

        # Frontend view before turn end: stunned is True
        view_before = s.view("p1")
        unit_view = Scenario.find_card_in_view(view_before, unit.instance_id)
        assert unit_view is not None
        assert unit_view["stunned"] is True

        # End the turn
        s.act("p1", ActionType.ADVANCE_PHASE)

        # Game state: stun cleared by clear_turn_state in _execute_end_of_turn
        assert unit.stunned is False

        # Frontend view after turn end: stunned is False
        # Check from P2's perspective
        view_p2 = s.view("p2")
        unit_view_after = Scenario.find_card_in_view(view_p2, unit.instance_id)
        assert unit_view_after is not None
        assert unit_view_after["stunned"] is False

        # Also verify from P1's own view
        view_p1 = s.view("p1")
        unit_view_p1 = Scenario.find_card_in_view(view_p1, unit.instance_id)
        assert unit_view_p1 is not None
        assert unit_view_p1["stunned"] is False


# =========================================================================
# Temporary unit expiration
# =========================================================================


class TestTemporaryUnitCleanup:
    """Proof: units with the Temporary keyword die at the start of their
    controller's next Beginning Phase."""

    def test_temporary_unit_dies_on_turn_end(self):
        """Proof: a Temporary unit owned by P2 is killed during P2's
        Beginning Phase (which runs automatically when P1 ends their turn
        and P2's turn starts). The unit moves to trash in both game state
        and frontend view.

        Flow: P1 ADVANCE_PHASE from ACTION ->
              _execute_end_of_turn (P1's turn ends) ->
              _start_next_turn (switches to P2) ->
              advance_phase from AWAKEN ->
              _execute_awaken (clears entered_this_turn for P2's units) ->
              _execute_beginning (kill_temporary_units for P2) ->
              channel -> draw -> ACTION
        """
        s = Scenario()
        s.add_deck_cards("p1", 5)
        s.add_deck_cards("p2", 5)

        # Create a Temporary unit for P2
        temp_kw = (KeywordInstance(Keyword.TEMPORARY, 0),)
        temp_unit = add_unit(
            s.gs, "p2", ZoneType.BASE, might=2,
            name="Ephemeral Spirit", keywords=temp_kw,
        )
        # Not protected by newness — entered_this_turn=False means
        # should_die_temporary will return True
        temp_unit.entered_this_turn = False

        # Verify unit is alive and has Temporary keyword
        assert temp_unit.zone == ZoneType.BASE
        assert temp_unit.has_keyword(Keyword.TEMPORARY)
        assert temp_unit.instance_id in s.gs.base_units["p2"]

        # P1 ends their turn, which triggers P2's full turn start sequence
        s.act("p1", ActionType.ADVANCE_PHASE)

        # Game state: temporary unit was killed during P2's Beginning Phase
        assert temp_unit.zone == ZoneType.TRASH
        assert temp_unit.instance_id in s.gs.players["p2"].trash
        assert temp_unit.instance_id not in s.gs.base_units.get("p2", [])

        # Frontend view from P2 (the current turn player): unit in trash
        view_p2 = s.view("p2")
        own_base_ids = [c["instance_id"] for c in view_p2["you"]["base_units"]]
        assert temp_unit.instance_id not in own_base_ids

        own_trash_ids = [c["instance_id"] for c in view_p2["you"]["trash"]]
        assert temp_unit.instance_id in own_trash_ids

        # Frontend view from P1 (opponent): unit in opponent's trash
        view_p1 = s.view("p1")
        opp_base_ids = [c["instance_id"] for c in view_p1["opponent"]["base_units"]]
        assert temp_unit.instance_id not in opp_base_ids

        opp_trash_ids = [c["instance_id"] for c in view_p1["opponent"]["trash"]]
        assert temp_unit.instance_id in opp_trash_ids
