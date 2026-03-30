"""Proof scenarios: Deathknell triggered abilities fire when a unit dies.

Deathknell (rule 734): When a unit with the Deathknell keyword dies
(damage >= might causes cleanup to send it to trash), its on_death
ability triggers and goes on the chain.

Key engine flow for damage-based death:
  1. Spell deals lethal damage to the unit.
  2. Cleanup runs _kill_dead_units: detects damage >= effective_might.
  3. Phase 1: fire_event(UNIT_DIED) fires while unit is still on board,
     pushing the on_death triggered ability onto the chain.
  4. Phase 2: unit moves to trash, board state updated.
  5. Both players pass priority to resolve the triggered ability on chain.
  6. The deathknell effect (e.g. draw_cards) executes.
"""

from __future__ import annotations

import pytest
from helpers import Scenario, add_unit

from app.engine.card_types import AbilityDefinition, CardDefinition, CardInstance, KeywordInstance
from app.engine.effect_ir import TargetSpec, deal_damage, draw_cards
from app.engine.enums import (
    AbilityType,
    ActionType,
    Keyword,
    Phase,
    ZoneType,
    CardType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_deathknell_unit(
    s: Scenario,
    player_id: str,
    might: int = 2,
    effect_ir: dict | None = None,
    ability_id: str = "dk_draw",
    text: str = "Deathknell: Draw 1.",
) -> CardInstance:
    """Create a unit with Deathknell and an on_death ability, placed in base."""
    if effect_ir is None:
        effect_ir = draw_cards(1)
    dk_ability = AbilityDefinition(
        ability_id=ability_id,
        ability_type=AbilityType.TRIGGERED,
        trigger_condition="on_death",
        effect_ir=effect_ir,
        text=text,
    )
    dk_unit_def = CardDefinition(
        card_id="dk_unit",
        name="Deathknell Unit",
        card_type=CardType.UNIT,
        base_might=might,
        keywords=(KeywordInstance(Keyword.DEATHKNELL, 0),),
        abilities=(dk_ability,),
    )
    dk_unit = CardInstance.create(dk_unit_def, player_id, ZoneType.BASE)
    s.gs.instances[dk_unit.instance_id] = dk_unit
    s.gs.base_units[player_id].append(dk_unit.instance_id)
    return dk_unit


class TestDeathknellProof:
    """Proof: Deathknell triggered abilities fire when a unit dies from
    lethal damage and produce correct frontend-visible results."""

    def test_deathknell_draw_on_lethal_damage(self):
        """Proof: A unit with Deathknell: draw 1 that takes lethal damage
        triggers its ability. After the trigger resolves, the controller
        has drawn a card, visible in both game state and frontend view.

        Flow:
          - P1 plays a damage spell targeting P1's own deathknell unit
          - Spell resolves (both pass priority), dealing lethal damage
          - Cleanup detects the dead unit, fires UNIT_DIED, pushes
            the deathknell trigger onto the chain
          - Both pass priority again to resolve the deathknell trigger
          - The draw_cards(1) effect fires for the unit's controller
        """
        s = Scenario()
        dk_unit = _make_deathknell_unit(s, "p1", might=2)
        s.add_deck_cards("p1", 5)
        s.give_resources("p1", energy=10)

        hand_before = len(s.gs.players["p1"].hand)

        # A damage spell that deals 5 to a friendly unit (lethal for might=2)
        spell = s.add_hand_spell(
            "p1", name="Self Destruct", cost_energy=1,
            effect_ir=deal_damage(5, TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
        )
        # Play the spell targeting the deathknell unit
        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [dk_unit.instance_id],
        })

        # Resolve the damage spell (both pass priority)
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # After spell resolves and cleanup runs, the deathknell trigger
        # should be on the chain. The unit took lethal damage, cleanup
        # detects it, fires UNIT_DIED, and pushes the on_death ability.
        # The unit should now be in trash.
        assert dk_unit.zone == ZoneType.TRASH

        # If the chain has items (the deathknell trigger), resolve them
        if not s.gs.chain.is_empty:
            s.act("p1", ActionType.PASS_PRIORITY)
            s.act("p2", ActionType.PASS_PRIORITY)

        # Game state: spell left hand (-1), deathknell drew 1 (+1) => net +1
        # from hand_before (measured before add_hand_spell added the spell)
        assert len(s.gs.players["p1"].hand) == hand_before + 1

        # Frontend view: hand count reflects the draw
        view = s.view("p1")
        assert view["you"]["hand_count"] == hand_before + 1

        # Frontend view: deathknell unit is in p1's trash
        trash_ids = [c["instance_id"] for c in view["you"]["trash"]]
        assert dk_unit.instance_id in trash_ids

        # Frontend view: deathknell unit is NOT in base
        base_ids = [c["instance_id"] for c in view["you"]["base_units"]]
        assert dk_unit.instance_id not in base_ids

    def test_deathknell_fires_from_enemy_damage(self):
        """Proof: A deathknell unit owned by P2 that takes lethal damage
        from P1's spell triggers the deathknell for P2 (the dying unit's
        controller). P2 draws a card, not P1.

        This tests the cross-player scenario: the opponent kills your
        unit, but YOUR deathknell triggers for YOUR benefit.
        """
        s = Scenario()
        dk_unit = _make_deathknell_unit(s, "p2", might=3)
        s.add_deck_cards("p2", 5)
        s.give_resources("p1", energy=10)

        p2_hand_before = len(s.gs.players["p2"].hand)
        p1_hand_before = len(s.gs.players["p1"].hand)

        # P1 plays a damage spell targeting P2's deathknell unit
        spell = s.add_hand_spell(
            "p1", name="Fireball", cost_energy=2,
            effect_ir=deal_damage(5, TargetSpec(obj_type="unit", scope="enemy", count=1)),
            targets_required=1,
        )

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [dk_unit.instance_id],
        })

        # Resolve the damage spell
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Unit should be dead
        assert dk_unit.zone == ZoneType.TRASH

        # Resolve deathknell trigger if on the chain
        if not s.gs.chain.is_empty:
            s.act("p1", ActionType.PASS_PRIORITY)
            s.act("p2", ActionType.PASS_PRIORITY)

        # P2 (dying unit's controller) drew 1 card from deathknell
        assert len(s.gs.players["p2"].hand) == p2_hand_before + 1

        # P1 did NOT draw from the deathknell (they lost the spell: -1)
        assert len(s.gs.players["p1"].hand) == p1_hand_before  # spell added then played

        # Frontend: P2's view shows the drawn card
        view_p2 = s.view("p2")
        assert view_p2["you"]["hand_count"] == p2_hand_before + 1

        # Frontend: dk_unit is in P2's trash
        trash_ids = [c["instance_id"] for c in view_p2["you"]["trash"]]
        assert dk_unit.instance_id in trash_ids

    def test_no_deathknell_without_keyword(self):
        """Proof: A normal unit without Deathknell does NOT trigger any
        draw effect when it dies from lethal damage. The hand count
        remains unchanged (no spurious trigger).
        """
        s = Scenario()
        # Create a plain unit with no keywords and no abilities
        plain_unit = add_unit(s.gs, "p1", ZoneType.BASE, might=2, name="Plain Unit")
        s.add_deck_cards("p1", 5)
        s.give_resources("p1", energy=10)

        hand_before = len(s.gs.players["p1"].hand)

        # Damage spell to kill the plain unit
        spell = s.add_hand_spell(
            "p1", name="Self Destruct", cost_energy=1,
            effect_ir=deal_damage(5, TargetSpec(obj_type="unit", scope="friendly", count=1)),
            targets_required=1,
        )

        s.act("p1", ActionType.PLAY_CARD, {
            "instance_id": spell.instance_id,
            "targets": [plain_unit.instance_id],
        })

        # Resolve the damage spell
        s.act("p1", ActionType.PASS_PRIORITY)
        s.act("p2", ActionType.PASS_PRIORITY)

        # Plain unit should be dead
        assert plain_unit.zone == ZoneType.TRASH

        # No deathknell trigger should be on the chain
        assert s.gs.chain.is_empty

        # Hand count: spell was added (+1 from add_hand_spell), then played (-1)
        # No draw triggered => hand_before
        assert len(s.gs.players["p1"].hand) == hand_before

        # Frontend: hand count unchanged
        view = s.view("p1")
        assert view["you"]["hand_count"] == hand_before

        # Frontend: plain unit is in trash
        trash_ids = [c["instance_id"] for c in view["you"]["trash"]]
        assert plain_unit.instance_id in trash_ids
