"""Integration tests for the Action keyword timing restrictions (rule 732).

Rule 732: A card with Action "may only be played during the controlling
player's Action Phase and only while the Chain is empty."
  - 732.1.a: Grants permission to play during the controller's Action Phase.
  - 732.1.b: Grants permission to play during Showdowns (Open state).
  - Action does NOT grant Reaction permission (cannot play in Closed states).

These tests go beyond validation — they exercise full play-resolve cycles
and verify that game state mutations actually happen.
"""

import pytest
from helpers import add_unit, make_game, make_spell_def, make_unit_def

from app.engine.action_executor import execute_action
from app.engine.action_validator import validate_action, ValidationResult
from app.engine.card_types import (
    AbilityDefinition,
    CardDefinition,
    CardInstance,
    KeywordInstance,
)
from app.engine.chain import push_card_to_chain, resolve_top_item
from app.engine.enums import (
    AbilityType,
    ActionType,
    CardType,
    Domain,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.game_state import ChainItem, ShowdownState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_action_spell(
    gs,
    owner_id: str,
    *,
    effect_ir: dict | None = None,
    targets_required: int = 0,
    cost_energy: int = 1,
    cost_power: tuple[tuple[Domain, int], ...] = (),
) -> CardInstance:
    """Create an Action spell in the player's hand.

    Defaults to a deal_damage IR so resolution has an observable effect.
    """
    if effect_ir is None:
        effect_ir = {"type": "deal_damage", "amount": 2}

    abilities = (
        AbilityDefinition(
            ability_id="action_spell_ab",
            ability_type=AbilityType.ACTIVATED,
            timing="action",
            effect_ir=effect_ir,
            targets_required=targets_required,
        ),
    )
    defn = CardDefinition(
        card_id="test_action_spell",
        name="Test Action Spell",
        card_type=CardType.SPELL,
        keywords=(KeywordInstance(keyword=Keyword.ACTION),),
        abilities=abilities,
        cost_energy=cost_energy,
        cost_power=cost_power,
    )
    inst = CardInstance.create(defn, owner_id, ZoneType.HAND)
    gs.instances[inst.instance_id] = inst
    gs.players[owner_id].hand.append(inst.instance_id)
    return inst


def _make_reaction_spell(
    gs,
    owner_id: str,
    *,
    effect_ir: dict | None = None,
    cost_energy: int = 1,
) -> CardInstance:
    """Create a Reaction spell in the player's hand."""
    if effect_ir is None:
        effect_ir = {"type": "draw_cards", "amount": 1}

    abilities = (
        AbilityDefinition(
            ability_id="reaction_spell_ab",
            ability_type=AbilityType.ACTIVATED,
            timing="reaction",
            effect_ir=effect_ir,
        ),
    )
    defn = CardDefinition(
        card_id="test_reaction_spell",
        name="Test Reaction Spell",
        card_type=CardType.SPELL,
        keywords=(KeywordInstance(keyword=Keyword.REACTION),),
        abilities=abilities,
        cost_energy=cost_energy,
    )
    inst = CardInstance.create(defn, owner_id, ZoneType.HAND)
    gs.instances[inst.instance_id] = inst
    gs.players[owner_id].hand.append(inst.instance_id)
    return inst


def _make_action_unit(
    gs,
    owner_id: str,
    *,
    might: int = 4,
    cost_energy: int = 3,
) -> CardInstance:
    """Create a unit with the Action keyword in the player's hand."""
    defn = make_unit_def(
        card_id="test_action_unit",
        name="Test Action Unit",
        might=might,
        keywords=(KeywordInstance(keyword=Keyword.ACTION),),
        cost_energy=cost_energy,
    )
    inst = CardInstance.create(defn, owner_id, ZoneType.HAND)
    gs.instances[inst.instance_id] = inst
    gs.players[owner_id].hand.append(inst.instance_id)
    return inst


def _fund_player(gs, player_id: str, energy: int = 10, power: dict | None = None):
    """Give a player enough resources to pay any cost."""
    gs.players[player_id].rune_pool.add_energy(energy)
    if power:
        for domain, amount in power.items():
            gs.players[player_id].rune_pool.add_power(domain, amount)


# ===========================================================================
# Test 1: Action spell plays and resolves during Action Phase
# ===========================================================================


class TestActionSpellPlaysAndResolves:
    """Play an Action spell during the turn player's Action Phase, fully
    resolve it through the chain, and verify the effect mutates game state."""

    def test_action_spell_plays_and_resolves_in_action_phase(self):
        """Full lifecycle: validate -> execute -> push to chain -> resolve.

        Uses a deal_damage spell targeting an enemy unit. Verifies that
        damage is actually applied after chain resolution.
        """
        gs = make_game(phase=Phase.ACTION)
        _fund_player(gs, "p1")

        bf_id = list(gs.battlefields.keys())[0]
        enemy = add_unit(gs, "p2", zone=ZoneType.BATTLEFIELD, bf_id=bf_id, might=5, name="Target Dummy")

        # Create action spell that deals 2 damage to target
        spell = _make_action_spell(
            gs, "p1",
            effect_ir={"type": "deal_damage", "amount": 2},
            targets_required=1,
        )

        # Step 1: Validation passes
        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": spell.instance_id, "targets": [enemy.instance_id]},
        )
        assert result.ok, f"Validation failed: {result.error}"

        # Step 2: Execute (removes from hand, pays cost, pushes to chain)
        execute_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": spell.instance_id, "targets": [enemy.instance_id]},
        )

        # Spell should be on the chain now
        assert not gs.chain.is_empty, "Spell should be on the chain"
        assert spell.instance_id not in gs.players["p1"].hand

        # Step 3: Resolve the chain
        logs = resolve_top_item(gs)

        # Step 4: Verify effect — enemy should have taken 2 damage
        assert enemy.damage == 2, f"Enemy should have 2 damage, has {enemy.damage}"
        # Spell should be in trash after resolution
        assert spell.zone == ZoneType.TRASH
        assert spell.instance_id in gs.players["p1"].trash
        # Chain should be empty
        assert gs.chain.is_empty


# ===========================================================================
# Test 2: Action spell rejected when chain is active
# ===========================================================================


class TestActionRejectedDuringChain:
    """Action cards cannot be played while the chain has items (rule 732).
    In Neutral Closed state, only Reaction-timed cards are permitted."""

    def test_action_spell_rejected_when_chain_active(self):
        """Put something on the chain, then attempt to play an Action card.
        The validator must reject it because the turn state is Neutral Closed.

        We give p1 priority (active_player_id) so the rejection is specifically
        about Action timing, not about lacking priority.
        """
        gs = make_game(phase=Phase.ACTION)
        _fund_player(gs, "p1")
        _fund_player(gs, "p2")

        # Populate chain with a dummy item (simulating p2 played a reaction)
        dummy = ChainItem.create(controller_id="p2")
        gs.chain.push(dummy)
        assert not gs.chain.is_empty

        # Give p1 priority so the timing check is what rejects the action card
        gs.active_player_id = "p1"

        action_spell = _make_action_spell(gs, "p1")

        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": action_spell.instance_id},
        )
        assert not result.ok, "Action card should be rejected when chain is non-empty"
        assert "cannot be played in current state" in result.error.lower()


# ===========================================================================
# Test 3: Action spell rejected on opponent's turn
# ===========================================================================


class TestActionRejectedOnOpponentTurn:
    """In Neutral Open, only the turn player can play cards (rule 316.2.b).
    P2 cannot play an Action spell during P1's turn."""

    def test_action_spell_rejected_on_opponent_turn(self):
        gs = make_game(phase=Phase.ACTION)  # p1 is turn player
        _fund_player(gs, "p2")

        spell = _make_action_spell(gs, "p2")

        result = validate_action(
            gs, "p2", ActionType.PLAY_CARD,
            {"instance_id": spell.instance_id},
        )
        assert not result.ok, "Action card should be rejected on opponent's turn"
        assert "not your turn" in result.error.lower()


# ===========================================================================
# Test 4: Reaction spell CAN play while chain has items
# ===========================================================================


class TestReactionPlaysWhileChainActive:
    """A Reaction spell (no Action keyword) is allowed in Closed state.
    This confirms the restriction is specific to Action cards."""

    def test_non_action_spell_plays_during_chain(self):
        """Push a dummy item to create Neutral Closed state, then play a
        Reaction spell from the active player — should be accepted."""
        gs = make_game(phase=Phase.ACTION)
        _fund_player(gs, "p1")
        _fund_player(gs, "p2")

        # Populate chain so state becomes Neutral Closed
        dummy = ChainItem.create(controller_id="p1")
        gs.chain.push(dummy)
        assert not gs.chain.is_empty

        # p2 has priority in Closed state (they are the non-turn player
        # who needs to respond). Set active_player_id to p2.
        gs.active_player_id = "p2"

        reaction = _make_reaction_spell(gs, "p2")

        result = validate_action(
            gs, "p2", ActionType.PLAY_CARD,
            {"instance_id": reaction.instance_id},
        )
        assert result.ok, f"Reaction spell should be allowed in Closed state: {result.error}"

    def test_action_rejected_but_reaction_accepted_same_chain(self):
        """Side-by-side: same chain state, Action rejected, Reaction accepted."""
        gs = make_game(phase=Phase.ACTION)
        _fund_player(gs, "p1")
        _fund_player(gs, "p2")

        dummy = ChainItem.create(controller_id="p1")
        gs.chain.push(dummy)
        gs.active_player_id = "p2"

        action_spell = _make_action_spell(gs, "p2")
        reaction_spell = _make_reaction_spell(gs, "p2")

        # Action: rejected
        r1 = validate_action(
            gs, "p2", ActionType.PLAY_CARD,
            {"instance_id": action_spell.instance_id},
        )
        assert not r1.ok

        # Reaction: accepted
        r2 = validate_action(
            gs, "p2", ActionType.PLAY_CARD,
            {"instance_id": reaction_spell.instance_id},
        )
        assert r2.ok, f"Reaction should be OK: {r2.error}"


# ===========================================================================
# Test 5: Action unit plays in Action Phase
# ===========================================================================


class TestActionUnitPlaysInActionPhase:
    """A unit with the Action keyword can be played during the controller's
    Action Phase with an empty chain (same timing as Action spells)."""

    def test_action_unit_plays_in_action_phase(self):
        gs = make_game(phase=Phase.ACTION)
        _fund_player(gs, "p1")

        unit = _make_action_unit(gs, "p1")

        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": unit.instance_id},
        )
        assert result.ok, f"Action unit should play in Action Phase: {result.error}"

    def test_action_unit_enters_base_after_execution(self):
        """Execute the play and verify the unit actually enters the base."""
        gs = make_game(phase=Phase.ACTION)
        _fund_player(gs, "p1")

        unit = _make_action_unit(gs, "p1")

        execute_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": unit.instance_id},
        )

        # Units finalize immediately (rule 356.2) — should be in base
        assert unit.zone == ZoneType.BASE
        assert unit.instance_id in gs.base_units["p1"]
        assert unit.instance_id not in gs.players["p1"].hand


# ===========================================================================
# Test 6: Action unit rejected during chain
# ===========================================================================


class TestActionUnitRejectedDuringChain:
    """A unit with the Action keyword cannot be played while the chain
    has items (Neutral Closed state)."""

    def test_action_unit_rejected_during_chain(self):
        gs = make_game(phase=Phase.ACTION)
        _fund_player(gs, "p1")

        dummy = ChainItem.create(controller_id="p2")
        gs.chain.push(dummy)

        unit = _make_action_unit(gs, "p1")

        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": unit.instance_id},
        )
        assert not result.ok, "Action unit should be rejected when chain is non-empty"
        assert "cannot be played in current state" in result.error.lower()


# ===========================================================================
# Test 7: Action in Showdown (Open vs Closed)
# ===========================================================================


class TestActionInShowdown:
    """Rule 732.1.b: Action cards CAN play in Showdown Open state.
    They CANNOT play in Showdown Closed (no Reaction permission)."""

    def test_action_card_plays_in_showdown_open(self):
        gs = make_game(phase=Phase.SHOWDOWN)
        _fund_player(gs, "p1")

        bf_id = list(gs.battlefields.keys())[0]
        gs.active_showdown = ShowdownState(
            battlefield_id=bf_id,
            initiator_id="p1",
            focus_player_id="p1",
        )

        spell = _make_action_spell(gs, "p1")

        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": spell.instance_id},
        )
        assert result.ok, f"Action card should play in Showdown Open: {result.error}"

    def test_action_card_rejected_in_showdown_closed(self):
        gs = make_game(phase=Phase.SHOWDOWN)
        _fund_player(gs, "p1")
        gs.active_player_id = "p1"

        bf_id = list(gs.battlefields.keys())[0]
        gs.active_showdown = ShowdownState(
            battlefield_id=bf_id,
            initiator_id="p1",
            focus_player_id="p1",
        )
        # Add chain item to make it Showdown Closed
        dummy = ChainItem.create(controller_id="p2")
        gs.chain.push(dummy)

        spell = _make_action_spell(gs, "p1")

        result = validate_action(
            gs, "p1", ActionType.PLAY_CARD,
            {"instance_id": spell.instance_id},
        )
        assert not result.ok, "Action card should be rejected in Showdown Closed"
        assert "cannot be played in current state" in result.error.lower()
