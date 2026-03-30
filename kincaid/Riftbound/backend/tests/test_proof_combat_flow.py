"""Proof scenarios for the FULL COMBAT FLOW — the most complex multi-step
sequence in Riftbound.

Combat flow (single-unit matchups auto-resolve):
  1. start_combat(gs, bf_id) opens a combat showdown
  2. Both players pass focus (PASS_FOCUS) to close the showdown
  3. Showdown closes -> execute_combat_damage is called
  4. With 1v1 units, damage is auto-assigned on both sides
  5. apply_combat_damage deals simultaneous damage
  6. resolve_combat kills lethal-damaged units, determines control

These tests prove the full pipeline produces correct frontend-visible results.
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.combat import start_combat
from app.engine.enums import (
    ActionType,
    CombatRole,
    ControlStatus,
    Phase,
    ZoneType,
)


class TestCombatFlowProof:
    """Proof: full combat flow produces correct frontend-visible results."""

    def test_combat_attacker_wins(self):
        """Proof: when the attacker's unit (might=5) fights a defender's
        unit (might=2), the defender dies (in trash) and the attacker
        survives at the battlefield, visible in the frontend view.

        Flow: start_combat -> showdown opens -> both pass focus ->
        showdown closes -> damage auto-assigned (1v1) -> damage applied ->
        weaker unit killed, stronger survives.
        """
        s = Scenario()
        bf_id = s.bf0
        bf = s.gs.battlefields[bf_id]
        bf.contested_by = "p1"

        # P1 attacker (might=5), P2 defender (might=2)
        u1 = add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=5, name="Attacker")
        u2 = add_unit(s.gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="Defender")

        # Start combat -- opens a showdown
        start_combat(s.gs, bf_id)

        # Verify showdown is open (combat phase = "showdown")
        assert s.gs.active_showdown is not None
        assert s.gs.active_showdown.is_combat_showdown is True

        # Both players pass focus to close the showdown
        # Attacker (initiator) has focus first
        s.act("p1", ActionType.PASS_FOCUS)
        # Defender passes focus -- showdown closes, combat resolves
        s.act("p2", ActionType.PASS_FOCUS)

        # -- Game state assertions --
        # Combat fully resolved (auto-assigned for 1v1)
        assert s.gs.active_combat is None
        assert s.gs.active_showdown is None

        # Defender (might=2) took 5 damage -> dead
        assert u2.zone == ZoneType.TRASH

        # Attacker (might=5) took 2 damage -> survives, healed after combat
        assert u1.zone == ZoneType.BATTLEFIELD
        assert u1.damage == 0  # healed after combat resolution

        # -- Frontend view assertions --
        view_p1 = s.view("p1")

        # Defender is in P2's trash
        defender_view = Scenario.find_card_in_view(view_p1, u2.instance_id)
        assert defender_view is not None
        assert defender_view["zone"] == "trash"

        # Attacker is still at the battlefield
        bf_view = view_p1["battlefields"][bf_id]
        bf_unit_ids = [c["instance_id"] for c in bf_view["units"]]
        assert u1.instance_id in bf_unit_ids
        assert u2.instance_id not in bf_unit_ids

    def test_combat_mutual_kill(self):
        """Proof: when both units have equal might (3 vs 3), both die
        simultaneously and appear in their owners' trash in the frontend.

        Rule 443.1.d.1.a: combat damage is applied simultaneously.
        Rule 444.2.d: if no units remain, battlefield becomes uncontrolled.
        """
        s = Scenario()
        bf_id = s.bf0
        bf = s.gs.battlefields[bf_id]
        bf.contested_by = "p1"

        u1 = add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Warrior A")
        u2 = add_unit(s.gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=3, name="Warrior B")

        # Start combat and resolve showdown
        start_combat(s.gs, bf_id)
        s.act("p1", ActionType.PASS_FOCUS)
        s.act("p2", ActionType.PASS_FOCUS)

        # -- Game state assertions --
        assert s.gs.active_combat is None

        # Both units dead
        assert u1.zone == ZoneType.TRASH
        assert u2.zone == ZoneType.TRASH

        # Battlefield uncontrolled (no units remain, rule 444.2.d)
        assert bf.control_status == ControlStatus.UNCONTROLLED
        assert bf.controller_id is None

        # -- Frontend view assertions --
        view_p1 = s.view("p1")
        view_p2 = s.view("p2")

        # P1's unit is in P1's trash
        p1_trash_ids = [c["instance_id"] for c in view_p1["you"]["trash"]]
        assert u1.instance_id in p1_trash_ids

        # P2's unit is in P2's trash (visible from P1's view as opponent trash)
        p2_trash_from_p1 = [c["instance_id"] for c in view_p1["opponent"]["trash"]]
        assert u2.instance_id in p2_trash_from_p1

        # P2 also sees their own unit in trash
        p2_own_trash = [c["instance_id"] for c in view_p2["you"]["trash"]]
        assert u2.instance_id in p2_own_trash

        # Neither unit is at the battlefield anymore
        bf_view = view_p1["battlefields"][bf_id]
        assert len(bf_view["units"]) == 0

    def test_combat_state_visible_during_showdown(self):
        """Proof: after start_combat opens a showdown, the frontend view
        exposes both the combat state (attacker/defender/phase) and the
        showdown state (focus player, is_combat_showdown).
        """
        s = Scenario()
        bf_id = s.bf0
        bf = s.gs.battlefields[bf_id]
        bf.contested_by = "p1"

        u1 = add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=4, name="Scout")
        u2 = add_unit(s.gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=4, name="Guard")

        start_combat(s.gs, bf_id)

        # -- Frontend view before any focus passes --
        view = s.view("p1")

        # Combat state is visible
        assert view["combat"] is not None
        assert view["combat"]["battlefield_id"] == bf_id
        assert view["combat"]["attacker_id"] == "p1"
        assert view["combat"]["defender_id"] == "p2"
        assert view["combat"]["phase"] == "showdown"

        # Showdown state is also visible (combat opens a showdown sub-phase)
        assert view["showdown"] is not None
        assert view["showdown"]["battlefield_id"] == bf_id
        assert view["showdown"]["is_combat_showdown"] is True
        assert view["showdown"]["focus_player_id"] == "p1"

        # Units have combat roles visible in the view
        u1_view = Scenario.find_card_in_view(view, u1.instance_id)
        u2_view = Scenario.find_card_in_view(view, u2.instance_id)
        assert u1_view["combat_role"] == "attacker"
        assert u2_view["combat_role"] == "defender"

        # After first pass, focus moves to the opponent
        s.act("p1", ActionType.PASS_FOCUS)
        view_after = s.view("p2")
        assert view_after["showdown"]["focus_player_id"] == "p2"

        # Clean up: close the showdown so combat resolves
        s.act("p2", ActionType.PASS_FOCUS)
        view_done = s.view("p1")
        assert view_done["combat"] is None
        assert view_done["showdown"] is None

    def test_battlefield_control_after_combat(self):
        """Proof: after combat where the attacker wins and the defender
        has no surviving units, the battlefield control_status changes
        to CONTROLLED with the attacker as controller, visible in the
        frontend view.

        Rule 444.2: battlefield control determined after combat.
        """
        s = Scenario()
        bf_id = s.bf0
        bf = s.gs.battlefields[bf_id]
        bf.contested_by = "p1"
        # Battlefield starts uncontrolled
        bf.control_status = ControlStatus.UNCONTROLLED
        bf.controller_id = None

        # P1 attacker (might=6) crushes P2 defender (might=2)
        u1 = add_unit(s.gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id, might=6, name="Conqueror")
        u2 = add_unit(s.gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id, might=2, name="Sentry")

        # Full combat flow
        start_combat(s.gs, bf_id)
        s.act("p1", ActionType.PASS_FOCUS)
        s.act("p2", ActionType.PASS_FOCUS)

        # -- Game state assertions --
        assert bf.control_status == ControlStatus.CONTROLLED
        assert bf.controller_id == "p1"
        assert bf.contested_by is None

        # -- Frontend view assertions --
        view_p1 = s.view("p1")
        bf_view = view_p1["battlefields"][bf_id]
        assert bf_view["control_status"] == "controlled"
        assert bf_view["controller_id"] == "p1"
        assert bf_view["contested_by"] is None

        # Opponent also sees the control change
        view_p2 = s.view("p2")
        bf_view_p2 = view_p2["battlefields"][bf_id]
        assert bf_view_p2["control_status"] == "controlled"
        assert bf_view_p2["controller_id"] == "p1"

        # The surviving attacker is at the battlefield
        assert u1.zone == ZoneType.BATTLEFIELD
        bf_unit_ids = [c["instance_id"] for c in bf_view["units"]]
        assert u1.instance_id in bf_unit_ids

        # The dead defender is in P2's trash
        assert u2.zone == ZoneType.TRASH
        p2_trash = [c["instance_id"] for c in view_p2["you"]["trash"]]
        assert u2.instance_id in p2_trash
