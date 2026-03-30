"""Proof scenarios: chain resolution order (LIFO) and priority passing.

Every test proves that the chain resolves items in last-in-first-out order
and that priority passing works correctly per rules 326-336.

Chain rules verified:
  - Items resolve LIFO (last played resolves first)
  - After P1 plays a spell, P1 has priority
  - P1 must pass, then P2 gets priority
  - P2 can play a Reaction spell (adding to the chain) -- then P2 has priority
  - When all players pass consecutively, the TOP item resolves
  - After resolution, if chain still has items, priority resets
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.card_types import KeywordInstance
from app.engine.effect_ir import (
    TargetSpec,
    deal_damage,
    give_might,
)
from app.engine.enums import (
    ActionType,
    Keyword,
    Phase,
    ZoneType,
)


class TestChainResolutionProof:
    """Proof: chain resolves items in LIFO order with correct priority flow."""

    def test_single_spell_resolves_on_both_pass(self):
        """Proof: P1 plays a damage spell, P1 passes, P2 passes.
        The spell resolves (damage dealt) and the chain is empty afterward.

        Rule 335: When all players pass consecutively, resolve the top item.
        Rule 336: After resolution, if chain is empty, return to Open State.
        """
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Target Dummy")
        spell = s.add_hand_spell(
            "p1", name="Zap", cost_energy=2,
            effect_ir=deal_damage(3, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # P1 plays the spell -- spell goes on chain, P1 gets priority
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })
        assert not s.gs.chain.is_empty, "Spell should be on the chain"
        assert s.gs.active_player_id == "p1", "P1 should have priority after playing"

        # P1 passes priority -- P2 now has priority
        s.act("p1", ActionType.PASS_PRIORITY)
        assert s.gs.active_player_id == "p2", "P2 should have priority after P1 passes"

        # P2 passes priority -- both passed, spell resolves
        view = s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: damage applied, chain empty
        assert target.damage == 3
        assert target.is_alive  # 5 might - 3 damage = alive
        assert s.gs.chain.is_empty

        # Frontend view: chain is empty, damage visible
        assert view["chain"]["is_empty"] is True
        assert len(view["chain"]["items"]) == 0

        target_view = Scenario.find_card_in_view(view, target.instance_id)
        assert target_view is not None
        assert target_view["damage"] == 3

    def test_lifo_order_reaction_resolves_first(self):
        """Proof: P1 plays a damage spell dealing 2, P1 passes, P2 plays a
        reaction spell that buffs (+3 might) their unit. Both pass. The
        reaction (top of chain) resolves FIRST, giving the buff BEFORE the
        damage spell resolves.

        Rule 336: Newest item resolves first (LIFO).
        Rule 334: After playing a Reaction, the controller gets priority.

        Verify: the unit has the +3 might buff AND has taken 2 damage.
        The buff resolved first (LIFO), then the damage.
        """
        s = Scenario()
        # P2's unit: base might 4
        p2_unit = add_unit(s.gs, "p2", ZoneType.BASE, might=4, name="Tough Guard")

        # P1's damage spell: deal 2 damage to enemy unit
        damage_spell = s.add_hand_spell(
            "p1", name="Firebolt", cost_energy=2,
            effect_ir=deal_damage(2, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=5)

        # P2's reaction spell: give +3 might to friendly unit
        buff_spell = s.add_hand_spell(
            "p2", name="Fortify", cost_energy=1,
            effect_ir=give_might(3, TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
            keywords=(KeywordInstance(Keyword.REACTION, 0),),
        )
        s.give_resources("p2", energy=5)

        # Step 1: P1 plays the damage spell (goes on chain, P1 has priority)
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": damage_spell.instance_id,
            "targets": [p2_unit.instance_id],
        })
        assert not s.gs.chain.is_empty
        assert len(s.gs.chain.stack) == 1, "One item on chain"

        # Step 2: P1 passes priority (P2 gets priority)
        s.act("p1", ActionType.PASS_PRIORITY)
        assert s.gs.active_player_id == "p2"

        # Step 3: P2 plays the reaction spell (adds to chain on TOP, P2 has priority)
        s.act("p2", ActionType.PLAY_CARD, {
            "instance_id": buff_spell.instance_id,
            "targets": [p2_unit.instance_id],
        })
        assert len(s.gs.chain.stack) == 2, "Two items on chain"
        assert s.gs.active_player_id == "p2", "P2 has priority after playing reaction"

        # Step 4: P2 passes priority (P1 gets priority)
        s.act("p2", ActionType.PASS_PRIORITY)
        assert s.gs.active_player_id == "p1"

        # Step 5: P1 passes priority -- both passed, TOP item resolves (the buff)
        # The buff resolves first (LIFO), giving +3 might to the unit
        s.act("p1", ActionType.PASS_PRIORITY)

        # After the buff resolves, chain still has the damage spell.
        # Priority resets. Both need to pass again for the damage to resolve.
        # The engine resolves the top item when both pass. After that resolution,
        # if the chain is not empty, passes reset (rule 336.4).
        # So now we need another round of passes for the damage spell.

        # Check that the buff resolved: unit should have +3 might
        assert p2_unit.effective_might == 7, (
            f"After buff resolves, effective might should be 4+3=7, "
            f"got {p2_unit.effective_might}"
        )

        # Now resolve the damage spell: both pass again
        s.act(s.gs.active_player_id, ActionType.PASS_PRIORITY)
        view = s.act(s.gs.active_player_id, ActionType.PASS_PRIORITY)

        # Game state: buff applied (+3), damage applied (2)
        assert p2_unit.damage == 2, f"Expected 2 damage, got {p2_unit.damage}"
        assert p2_unit.effective_might == 7, (
            f"Buff should still be active: expected 7, got {p2_unit.effective_might}"
        )
        assert p2_unit.is_alive  # 7 might - 2 damage = alive
        assert s.gs.chain.is_empty

        # Frontend view: chain empty, buff and damage both visible
        p2_view = s.view("p2")
        unit_view = Scenario.find_card_in_view(p2_view, p2_unit.instance_id)
        assert unit_view is not None
        assert unit_view["damage"] == 2
        assert unit_view["effective_might"] == 7  # base 4 + buff 3

    def test_chain_state_visible_in_view(self):
        """Proof: while a spell is on the chain, the frontend view shows
        the chain is non-empty with the spell visible. After resolution,
        the chain is empty.

        Rule 356.3: Spells linger on chain as Finalized Chain Items.
        Rule 336.2: After resolution, if chain is empty -> Open State.
        """
        s = Scenario()
        target = add_unit(s.gs, "p2", ZoneType.BASE, might=5, name="Sentinel")
        spell = s.add_hand_spell(
            "p1", name="Lightning", cost_energy=1,
            effect_ir=deal_damage(1, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )
        s.give_resources("p1", energy=3)

        # Before play: chain should be empty
        view_before = s.view("p1")
        assert view_before["chain"]["is_empty"] is True
        assert len(view_before["chain"]["items"]) == 0

        # Play the spell -- it goes on the chain
        view_on_chain = s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [target.instance_id],
        })

        # During chain: chain is non-empty, spell is visible
        assert view_on_chain["chain"]["is_empty"] is False
        assert len(view_on_chain["chain"]["items"]) >= 1

        # The spell should be in the chain items
        chain_card_ids = [
            item["card"]["instance_id"]
            for item in view_on_chain["chain"]["items"]
            if "card" in item
        ]
        assert spell.instance_id in chain_card_ids, (
            f"Spell {spell.instance_id} should be visible in chain items"
        )

        # The chain item should show the spell's name
        spell_item = next(
            item for item in view_on_chain["chain"]["items"]
            if "card" in item and item["card"]["instance_id"] == spell.instance_id
        )
        assert spell_item["card"]["name"] == "Lightning"
        assert spell_item["controller_id"] == "p1"

        # Both pass -- spell resolves, chain empties
        s.act("p1", ActionType.PASS_PRIORITY)
        view_after = s.act("p2", ActionType.PASS_PRIORITY)

        assert view_after["chain"]["is_empty"] is True
        assert len(view_after["chain"]["items"]) == 0
