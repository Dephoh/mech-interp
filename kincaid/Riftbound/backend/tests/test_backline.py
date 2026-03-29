"""Tests for the Backline keyword (rules 756-757).

Backline means "I cannot attack." / "I cannot be declared as an attacker."
- Backline units CANNOT be declared as attackers.
- Backline units CAN defend.
- Backline units CAN still be at battlefields and contribute to control.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from helpers import add_unit, load_card_db, make_game

from app.engine.card_db import CardDB
from app.engine.card_types import CardInstance, KeywordInstance
from app.engine.combat import start_combat
from app.engine.enums import (
    CardType,
    CombatRole,
    ControlStatus,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.scoring import score_hold
from app.engine.trigger_system import GameEvent, fire_event


# ---------------------------------------------------------------------------
# Card IDs used across tests
# ---------------------------------------------------------------------------
ENTHUSIASTIC_PROMOTER = "unl-043-219"  # Backline, 3E, 2 might, on_hold buff
BLASTCONE_FAE = "ogn-097-298"         # No backline, 2E, 2 might


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_db_once():
    """Ensure CardDB is populated (idempotent)."""
    if CardDB.get_or_none(ENTHUSIASTIC_PROMOTER) is None:
        load_card_db()


def _place_real_unit(
    gs, card_id: str, player_id: str, bf_id: str,
) -> CardInstance:
    """Place a unit from the real card database onto a battlefield."""
    _load_db_once()
    defn = CardDB.get(card_id)
    inst = CardInstance.create(defn, player_id, ZoneType.BATTLEFIELD, location_id=bf_id)
    inst.exhausted = False
    inst.entered_this_turn = False
    gs.instances[inst.instance_id] = inst
    gs.battlefields[bf_id].units.append(inst.instance_id)
    return inst


def _setup_combat_game():
    """Create a game with one battlefield, contested by p1."""
    gs = make_game(num_battlefields=1)
    bf_id = list(gs.battlefields.keys())[0]
    bf = gs.battlefields[bf_id]
    bf.control_status = ControlStatus.CONTESTED
    bf.contested_by = "p1"
    return gs, bf_id, bf


# ---------------------------------------------------------------------------
# Test 1: Backline unit excluded from attackers
# ---------------------------------------------------------------------------


class TestBacklineExcludedFromAttackers:
    """Place a backline unit + a normal unit on P1's side. P2 has units too.
    P1 attacks. After combat starts the normal unit should be ATTACKER
    while the backline unit should NOT."""

    def test_backline_unit_excluded_from_attackers(self):
        gs, bf_id, bf = _setup_combat_game()

        # Attacker side (p1): one backline, one normal
        promoter = _place_real_unit(gs, ENTHUSIASTIC_PROMOTER, "p1", bf_id)
        fae = _place_real_unit(gs, BLASTCONE_FAE, "p1", bf_id)

        # Defender side (p2): one normal
        defender = add_unit(
            gs, "p2", zone=ZoneType.BATTLEFIELD, bf_id=bf_id,
            might=3, name="Defender",
        )

        logs = start_combat(gs, bf_id)

        # Normal unit must be ATTACKER
        assert fae.combat_role == CombatRole.ATTACKER, (
            f"Blastcone Fae should be ATTACKER, got {fae.combat_role}"
        )
        # Backline unit must NOT be ATTACKER
        assert promoter.combat_role != CombatRole.ATTACKER, (
            f"Enthusiastic Promoter (Backline) should NOT be ATTACKER, got {promoter.combat_role}"
        )


# ---------------------------------------------------------------------------
# Test 2: Backline unit defends normally
# ---------------------------------------------------------------------------


class TestBacklineDefendsNormally:
    """Place a backline unit on P2's side. P1 attacks. The backline unit
    must be assigned DEFENDER and participate in damage normally."""

    def test_backline_unit_defends_normally(self):
        gs, bf_id, bf = _setup_combat_game()

        # Attacker (p1)
        attacker = _place_real_unit(gs, BLASTCONE_FAE, "p1", bf_id)

        # Defender (p2) — the backline unit
        promoter = _place_real_unit(gs, ENTHUSIASTIC_PROMOTER, "p2", bf_id)

        logs = start_combat(gs, bf_id)

        # Backline unit on defending side must be DEFENDER
        assert promoter.combat_role == CombatRole.DEFENDER, (
            f"Backline defender should be DEFENDER, got {promoter.combat_role}"
        )
        # Attacker should be ATTACKER
        assert attacker.combat_role == CombatRole.ATTACKER, (
            f"Blastcone Fae should be ATTACKER, got {attacker.combat_role}"
        )

        # Verify the defender's might is used for combat damage (2 might)
        assert promoter.combat_might == promoter.definition.base_might, (
            f"Backline defender combat_might should be {promoter.definition.base_might}, "
            f"got {promoter.combat_might}"
        )


# ---------------------------------------------------------------------------
# Test 3: Only backline units on attacking side — no valid attackers
# ---------------------------------------------------------------------------


class TestBacklineOnlyNoAttackers:
    """If P1 has ONLY backline units at the battlefield, there should be
    no valid attackers after combat starts."""

    def test_backline_only_units_no_combat(self):
        gs, bf_id, bf = _setup_combat_game()

        # Attacker side (p1): ONLY backline units
        promoter1 = _place_real_unit(gs, ENTHUSIASTIC_PROMOTER, "p1", bf_id)
        # Add a second one built from helpers for variety
        promoter2 = add_unit(
            gs, "p1", zone=ZoneType.BATTLEFIELD, bf_id=bf_id,
            might=2, name="Backline Support",
            keywords=(KeywordInstance(keyword=Keyword.BACKLINE),),
        )

        # Defender side (p2)
        defender = add_unit(
            gs, "p2", zone=ZoneType.BATTLEFIELD, bf_id=bf_id,
            might=3, name="Defender",
        )

        logs = start_combat(gs, bf_id)

        # Neither p1 unit should be ATTACKER
        assert promoter1.combat_role != CombatRole.ATTACKER, (
            "First backline unit should not be ATTACKER"
        )
        assert promoter2.combat_role != CombatRole.ATTACKER, (
            "Second backline unit should not be ATTACKER"
        )

        # Verify no unit on p1's side received the ATTACKER role
        p1_units_at_bf = [
            gs.instances[uid] for uid in bf.units
            if gs.instances[uid].controller_id == "p1"
        ]
        attackers = [u for u in p1_units_at_bf if u.combat_role == CombatRole.ATTACKER]
        assert len(attackers) == 0, (
            f"Expected 0 attackers when all units have Backline, found {len(attackers)}"
        )


# ---------------------------------------------------------------------------
# Test 4: Backline unit still contributes to battlefield control
# ---------------------------------------------------------------------------


class TestBacklineControlsBattlefield:
    """A backline unit placed at a battlefield should contribute to
    control — the battlefield should become controlled for that player
    (or at least contested if both sides are present)."""

    def test_backline_unit_still_controls_battlefield(self):
        gs = make_game(num_battlefields=1)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # Start uncontrolled, no units
        assert bf.control_status == ControlStatus.UNCONTROLLED

        # Place a backline unit on p1's side
        promoter = _place_real_unit(gs, ENTHUSIASTIC_PROMOTER, "p1", bf_id)

        # Manually set the battlefield to controlled by p1 as a deploy would
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        # Verify the unit is at the battlefield
        assert promoter.instance_id in bf.units
        assert promoter.location_id == bf_id

        # The battlefield is controlled by p1; verify Hold scoring works
        gs.turn_player_id = "p1"
        gs.active_player_id = "p1"
        gs.phase = Phase.BEGINNING
        hold_logs = score_hold(gs, "p1")

        # p1 should score a point from holding
        assert gs.players["p1"].score >= 1, (
            f"p1 should have scored from holding with backline unit present, "
            f"score = {gs.players['p1'].score}"
        )

    def test_backline_unit_at_contested_battlefield(self):
        """When both players have units at a battlefield (one with backline),
        the battlefield should be contestable / have units from both sides."""
        gs = make_game(num_battlefields=1)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # Place backline unit for p1 and normal unit for p2
        promoter = _place_real_unit(gs, ENTHUSIASTIC_PROMOTER, "p1", bf_id)
        defender = add_unit(
            gs, "p2", zone=ZoneType.BATTLEFIELD, bf_id=bf_id,
            might=3, name="Defender",
        )

        # Both players should have units at this battlefield
        units_by_player = bf.units_by_player(gs.instances)
        assert "p1" in units_by_player, "p1 should have units at the battlefield"
        assert "p2" in units_by_player, "p2 should have units at the battlefield"
        assert len(units_by_player["p1"]) >= 1
        assert len(units_by_player["p2"]) >= 1


# ---------------------------------------------------------------------------
# Test 5: Backline unit's triggered ability still fires during hold
# ---------------------------------------------------------------------------


class TestBacklineWithAbilities:
    """Enthusiastic Promoter (unl-043-219) has 'When I hold, Buff all units
    here'. Even though the unit has Backline, its on_hold ability should
    still fire during the hold step."""

    def test_backline_with_abilities(self):
        gs = make_game(num_battlefields=1)
        bf_id = list(gs.battlefields.keys())[0]
        bf = gs.battlefields[bf_id]

        # p1 controls the battlefield
        bf.control_status = ControlStatus.CONTROLLED
        bf.controller_id = "p1"

        # Place Enthusiastic Promoter for p1 at the battlefield
        promoter = _place_real_unit(gs, ENTHUSIASTIC_PROMOTER, "p1", bf_id)

        # Place another friendly unit at the same battlefield to receive the buff
        ally = add_unit(
            gs, "p1", zone=ZoneType.BATTLEFIELD, bf_id=bf_id,
            might=3, name="Ally Unit",
        )
        ally.entered_this_turn = False

        # Verify the promoter has backline
        assert promoter.has_keyword(Keyword.BACKLINE), (
            "Enthusiastic Promoter should have the Backline keyword"
        )

        # Verify the promoter has an on_hold triggered ability
        hold_abilities = [
            ab for ab in promoter.definition.abilities
            if ab.trigger_condition == "on_hold"
        ]
        assert len(hold_abilities) > 0, (
            "Enthusiastic Promoter should have an on_hold triggered ability"
        )

        # Fire a HOLD event for this battlefield (simulating the Beginning
        # Phase scoring step where hold triggers fire)
        gs.turn_player_id = "p1"
        gs.active_player_id = "p1"

        trigger_logs = fire_event(gs, GameEvent.HOLD, {
            "player_id": "p1",
            "battlefield_id": bf_id,
        })

        # The on_hold ability should have triggered and been pushed onto the chain
        assert len(trigger_logs) > 0, (
            "Enthusiastic Promoter's on_hold ability should fire during hold step "
            "even though it has Backline"
        )

        # Verify the trigger was from the promoter
        promoter_triggered = any(
            promoter.name in log for log in trigger_logs
        )
        assert promoter_triggered, (
            f"Expected Enthusiastic Promoter to appear in trigger logs, "
            f"got: {trigger_logs}"
        )

        # Verify a chain item was pushed for the ability
        assert not gs.chain.is_empty, (
            "The on_hold triggered ability should have pushed an item onto the chain"
        )

        # Verify the chain item is from the promoter
        chain_item = gs.chain.peek()
        assert chain_item is not None
        assert chain_item.source_instance_id == promoter.instance_id, (
            f"Chain item source should be the promoter, got {chain_item.source_instance_id}"
        )
