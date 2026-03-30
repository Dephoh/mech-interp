"""Proof scenarios for the COUNTER effect (counterspell).

These tests prove that counter-spells work correctly through the full
validate -> execute -> cleanup pipeline, including chain priority flow
and Reaction timing.

Chain flow for counter:
  1. P1 plays a damage spell -> P1 has priority
  2. P1 passes priority -> P2 gets priority (chain is closed)
  3. P2 plays counter-spell (Reaction keyword allows play in closed state)
     targeting the damage spell on chain -> P2 has priority
  4. P2 passes priority -> P1 gets priority
  5. P1 passes priority -> both passed, top of chain (counter) resolves
  6. Counter removes the damage spell from chain -> damage spell goes to
     trash without resolving, target takes no damage
  7. Chain is now empty -> state returns to Neutral Open
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.card_types import KeywordInstance
from app.engine.effect_ir import (
    TargetSpec,
    counter,
    deal_damage,
)
from app.engine.enums import (
    ActionType,
    Keyword,
    ZoneType,
)


class TestCounterProof:
    """Proof: counter-spells remove their target from the chain before
    it resolves, preventing its effects."""

    def test_counter_removes_spell_from_chain(self):
        """Proof: P1 plays a damage spell, P2 counters it with a Reaction
        counter-spell. The damage spell is removed from the chain, goes to
        trash without resolving, and the target unit takes no damage."""
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Guarded")
        s.give_resources("p1", energy=10)
        s.give_resources("p2", energy=10)

        # P1's damage spell: deal 4 damage to an enemy unit
        damage_spell = s.add_hand_spell(
            "p1", name="Bolt", cost_energy=2,
            effect_ir=deal_damage(4, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )

        # P2's counter-spell: counter a spell on the chain (Reaction keyword)
        counter_spell = s.add_hand_spell(
            "p2", name="Deny", cost_energy=1,
            effect_ir=counter(TargetSpec(obj_type="spell", zone="chain", count=1)),
            targets_required=1,
            keywords=(KeywordInstance(Keyword.REACTION, 0),),
        )

        # Step 1: P1 plays the damage spell targeting the unit
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": damage_spell.instance_id,
            "targets": [target.instance_id],
        })
        # Chain is now closed with the damage spell; P1 has priority
        assert not s.gs.chain.is_empty
        assert s.gs.active_player_id == "p1"

        # Step 2: P1 passes priority -> P2 gets priority
        s.act("p1", ActionType.PASS_PRIORITY)
        assert s.gs.active_player_id == "p2"

        # Step 3: P2 plays counter-spell (Reaction) targeting the damage
        # spell on chain. The target is the damage spell's card_instance_id.
        s.act("p2", ActionType.PLAY_CARD, {
            "instance_id": counter_spell.instance_id,
            "targets": [damage_spell.instance_id],
        })
        # Counter is now on top of chain; P2 has priority
        assert s.gs.active_player_id == "p2"

        # Step 4: P2 passes priority -> P1 gets priority
        s.act("p2", ActionType.PASS_PRIORITY)
        assert s.gs.active_player_id == "p1"

        # Step 5: P1 passes priority -> both passed, counter resolves (LIFO)
        # The counter removes the damage spell from the chain.
        # Then the counter spell itself goes to trash.
        # Chain becomes empty -> Neutral Open.
        view = s.act("p1", ActionType.PASS_PRIORITY)

        # Game state: damage spell was countered -> in trash, never resolved
        assert damage_spell.zone == ZoneType.TRASH
        assert target.damage == 0, "Target should have taken no damage"
        assert target.is_alive

        # Game state: counter spell also in trash (it resolved normally)
        assert counter_spell.zone == ZoneType.TRASH

        # Game state: chain is empty
        assert s.gs.chain.is_empty

        # Frontend view: target unit shows zero damage
        target_view = Scenario.find_card_in_view(s.view("p2"), target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 0

        # Frontend view: both spells are in their owners' trash
        p1_view = s.view("p1")
        p1_trash_ids = [c["instance_id"] for c in p1_view["you"]["trash"]]
        assert damage_spell.instance_id in p1_trash_ids

        p2_view = s.view("p2")
        p2_trash_ids = [c["instance_id"] for c in p2_view["you"]["trash"]]
        assert counter_spell.instance_id in p2_trash_ids

    def test_uncountered_spell_resolves(self):
        """Proof (control test): when no counter is played, a damage spell
        resolves normally and deals damage to the target."""
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Victim")
        s.give_resources("p1", energy=10)

        # P1's damage spell: deal 4 damage to an enemy unit
        damage_spell = s.add_hand_spell(
            "p1", name="Bolt", cost_energy=2,
            effect_ir=deal_damage(4, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )

        # Step 1: P1 plays the damage spell targeting the unit
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": damage_spell.instance_id,
            "targets": [target.instance_id],
        })
        assert not s.gs.chain.is_empty

        # Step 2: P1 passes priority -> P2 gets priority
        s.act("p1", ActionType.PASS_PRIORITY)

        # Step 3: P2 passes priority (no counter) -> both passed, spell resolves
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: spell resolved -> target took 4 damage
        assert target.damage == 4
        assert target.is_alive  # 5 might - 4 damage = still alive

        # Game state: spell went to trash after resolving
        assert damage_spell.zone == ZoneType.TRASH

        # Game state: chain is empty
        assert s.gs.chain.is_empty

        # Frontend view: target unit shows 4 damage
        target_view = Scenario.find_card_in_view(view, target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 4
        assert target_view["effective_might"] == 5  # might unchanged

        # Frontend view: spell is in P1's trash
        p1_view = s.view("p1")
        p1_trash_ids = [c["instance_id"] for c in p1_view["you"]["trash"]]
        assert damage_spell.instance_id in p1_trash_ids
