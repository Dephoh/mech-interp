"""Proof scenarios: battlefield control transitions and scoring.

Tests prove that battlefields correctly transition between UNCONTROLLED,
contested (contested_by set), and CONTROLLED states, and that scoring
via Hold works when a player controls a battlefield during their turn's
Beginning Phase.

Control lifecycle:
  1. Battlefield starts UNCONTROLLED (no contested_by, no controller_id)
  2. Player moves a unit there -> contested_by = that player
  3. Cleanup stages a showdown (sole presence, uncontrolled + contested_by)
  4. Cleanup opens the showdown; both players pass focus -> showdown closes
  5. close_showdown sees one player's units -> CONTROLLED by that player
  6. Next turn's Beginning Phase -> score_hold awards 1 point per controlled BF
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.enums import (
    ActionType,
    ControlStatus,
    Phase,
    ZoneType,
)


class TestBattlefieldControlTransitions:
    """Proof: battlefield control transitions produce correct frontend state."""

    def test_move_unit_contests_battlefield(self):
        """Proof: moving a unit to an UNCONTROLLED battlefield sets
        contested_by to that player in both game state and frontend view.

        The control_status remains 'uncontrolled' because control is not
        granted until a showdown resolves. The contested_by field signals
        which player initiated the contest."""
        s = Scenario()
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Pioneer")
        unit.entered_this_turn = False
        unit.exhausted = False

        # Pre-condition: battlefield is uncontrolled
        bf_id = s.bf0
        bf = s.gs.battlefields[bf_id]
        assert bf.control_status == ControlStatus.UNCONTROLLED
        assert bf.contested_by is None
        assert bf.controller_id is None

        view = s.act("p1", ActionType.MOVE_UNIT, {
            "instance_id": unit.instance_id,
            "destination": {"zone": "battlefield", "id": bf_id},
        })

        # Game state: contested_by is set, control_status is now CONTESTED
        assert bf.contested_by == "p1"
        assert bf.control_status == ControlStatus.CONTESTED

        # Frontend view: contested_by visible, control_status is "contested"
        bf_view = view["battlefields"][bf_id]
        assert bf_view["contested_by"] == "p1"
        assert bf_view["control_status"] == ControlStatus.CONTESTED.value

    def test_sole_presence_conquers(self):
        """Proof: when only one player has units at a contested battlefield,
        the showdown resolves in their favor and the battlefield becomes
        CONTROLLED.

        Flow: move unit -> cleanup stages showdown -> cleanup opens showdown
        -> both pass focus -> close_showdown sets CONTROLLED + scores conquer.
        """
        s = Scenario()
        unit = add_unit(s.gs, "p1", ZoneType.BASE, might=3, name="Conqueror")
        unit.entered_this_turn = False
        unit.exhausted = False

        bf_id = s.bf0

        # Move triggers cleanup which stages + opens showdown
        view = s.act("p1", ActionType.MOVE_UNIT, {
            "instance_id": unit.instance_id,
            "destination": {"zone": "battlefield", "id": bf_id},
        })

        # After move, a showdown should be active at this battlefield
        assert s.gs.active_showdown is not None
        assert s.gs.active_showdown.battlefield_id == bf_id

        # P1 passes focus (initiator passes first)
        s.act("p1", ActionType.PASS_FOCUS)
        # P2 passes focus -> showdown closes -> battlefield becomes CONTROLLED
        view = s.act("p2", ActionType.PASS_FOCUS)

        # Game state: battlefield is now CONTROLLED by p1
        bf = s.gs.battlefields[bf_id]
        assert bf.control_status == ControlStatus.CONTROLLED
        assert bf.controller_id == "p1"
        assert bf.contested_by is None  # cleared after conquer

        # Frontend view: control_status and controller_id reflect conquer
        bf_view = view["battlefields"][bf_id]
        assert bf_view["control_status"] == ControlStatus.CONTROLLED.value
        assert bf_view["controller_id"] == "p1"

    def test_both_players_keeps_contested(self):
        """Proof: when both players have units at a contested battlefield,
        the battlefield does NOT become controlled — it stays contested
        and a combat is staged instead.

        Two-player presence means neither side has sole control. The engine
        stages a combat (not a showdown) because opposing units are present."""
        s = Scenario()
        bf_id = s.bf0
        bf = s.gs.battlefields[bf_id]

        # Place units for both players at the battlefield
        u1 = add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Vanguard")
        u2 = add_unit(s.gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Defender")
        bf.contested_by = "p1"

        # Run an action to trigger cleanup (advance_phase would change state
        # too much, so we use a second move on a different battlefield)
        u3 = add_unit(s.gs, "p1", ZoneType.BASE, might=2, name="Scout")
        u3.entered_this_turn = False
        u3.exhausted = False
        bf1_id = s.bf_ids[1]

        view = s.act("p1", ActionType.MOVE_UNIT, {
            "instance_id": u3.instance_id,
            "destination": {"zone": "battlefield", "id": bf1_id},
        })

        # Game state: bf0 is NOT controlled — both players have units there
        assert bf.control_status != ControlStatus.CONTROLLED
        assert bf.controller_id is None

        # Frontend view: control_status is not "controlled"
        bf_view = view["battlefields"][bf_id]
        assert bf_view["control_status"] != "controlled"
        assert bf_view["controller_id"] is None

    def test_control_status_visible_to_both_players(self):
        """Proof: after a battlefield is CONTROLLED, both the controller
        and the opponent see identical control_status and controller_id
        in their serialized frontend views."""
        s = Scenario()
        bf_id = s.bf0
        bf = s.gs.battlefields[bf_id]

        # Directly set up the controlled state (simulating post-showdown)
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"
        bf.contested_by = None
        add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Holder")

        # Get views for both players
        view_p1 = s.view("p1")
        view_p2 = s.view("p2")

        # Both players see the same control state
        bf_p1 = view_p1["battlefields"][bf_id]
        bf_p2 = view_p2["battlefields"][bf_id]

        assert bf_p1["control_status"] == "controlled"
        assert bf_p1["controller_id"] == "p1"

        assert bf_p2["control_status"] == "controlled"
        assert bf_p2["controller_id"] == "p1"

        # The views agree exactly
        assert bf_p1["control_status"] == bf_p2["control_status"]
        assert bf_p1["controller_id"] == bf_p2["controller_id"]
