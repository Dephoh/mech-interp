"""Shared test helpers: build minimal GameState fixtures."""

from __future__ import annotations

import sys
import os

# Ensure the app package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engine.card_db import CardDB
from app.engine.card_types import CardDefinition, CardInstance
from app.engine.enums import (
    CardType,
    CombatRole,
    ControlStatus,
    Domain,
    GameMode,
    Keyword,
    Phase,
    ZoneType,
)
from app.engine.card_types import AbilityDefinition, KeywordInstance
from app.engine.game_state import (
    BattlefieldState,
    GameState,
    PlayerState,
    create_game,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_card_db():
    CardDB.load_all(DATA_DIR)


def make_unit_def(
    card_id: str = "test_unit",
    name: str = "Test Unit",
    might: int = 3,
    domains: tuple[Domain, ...] = (Domain.FURY,),
    keywords: tuple[KeywordInstance, ...] = (),
    cost_energy: int = 2,
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        name=name,
        card_type=CardType.UNIT,
        domains=domains,
        base_might=might,
        keywords=keywords,
        cost_energy=cost_energy,
    )


def make_spell_def(
    card_id: str = "test_spell",
    name: str = "Test Spell",
    keywords: tuple[KeywordInstance, ...] = (),
    effect_ir: dict | None = None,
    cost_energy: int = 1,
) -> CardDefinition:
    abilities = ()
    if effect_ir:
        abilities = (
            AbilityDefinition(
                ability_id=f"{card_id}_ab",
                ability_type="activated",
                effect_ir=effect_ir,
            ),
        )
    return CardDefinition(
        card_id=card_id,
        name=name,
        card_type=CardType.SPELL,
        keywords=keywords,
        abilities=abilities,
        cost_energy=cost_energy,
    )


def make_game(
    num_battlefields: int = 2,
    phase: Phase = Phase.ACTION,
    game_mode: GameMode = GameMode.STANDARD_1V1,
    num_players: int | None = None,
    teams: dict[str, int] | None = None,
) -> GameState:
    """Create a minimal game state for testing.

    Args:
        num_battlefields: How many battlefields to create.
        phase: Starting phase.
        game_mode: Game mode enum value.
        num_players: Override player count (default: derived from game_mode).
        teams: Optional team mapping (player_id -> team_id) for 2v2.
    """
    from app.engine.game_state import _mode_config

    mcfg = _mode_config(game_mode)
    player_count = num_players or mcfg["players"]

    gs = GameState(game_id="test", game_mode=game_mode)
    gs.victory_score = mcfg["victory"]

    player_ids = [f"p{i+1}" for i in range(player_count)]
    for pid in player_ids:
        ps = PlayerState(player_id=pid, display_name=f"Player {pid[1:]}")
        ps.is_first_turn = False
        gs.players[pid] = ps
        gs.mulligan_done[pid] = True
        gs.base_units[pid] = []
        gs.base_gear[pid] = []
        gs.base_runes[pid] = []

    gs.player_order = player_ids
    gs.turn_player_id = player_ids[0]
    gs.active_player_id = player_ids[0]
    gs.phase = phase
    gs.turn_number = 1

    # Teams
    if teams:
        gs.teams = dict(teams)
    elif mcfg["teams"]:
        # Default: first half team 0, second half team 1
        for i, pid in enumerate(player_ids):
            gs.teams[pid] = i % 2

    # Add battlefields
    for i in range(num_battlefields):
        bf_def = CardDefinition(
            card_id=f"bf_{i}",
            name=f"Battlefield {i}",
            card_type=CardType.BATTLEFIELD,
        )
        bf_inst = CardInstance.create(bf_def, "neutral", ZoneType.BATTLEFIELD)
        gs.instances[bf_inst.instance_id] = bf_inst
        gs.battlefields[bf_inst.instance_id] = BattlefieldState(
            battlefield_id=bf_inst.instance_id,
            card_instance_id=bf_inst.instance_id,
        )

    return gs


def add_unit(
    gs: GameState,
    player_id: str,
    zone: ZoneType = ZoneType.BASE,
    might: int = 3,
    name: str = "Unit",
    keywords: tuple[KeywordInstance, ...] = (),
    bf_id: str | None = None,
) -> CardInstance:
    """Add a unit to the game and return it."""
    defn = make_unit_def(name=name, might=might, keywords=keywords)
    inst = CardInstance.create(defn, player_id, zone)
    inst.exhausted = False
    gs.instances[inst.instance_id] = inst

    if zone == ZoneType.BASE:
        gs.base_units[player_id].append(inst.instance_id)
    elif zone == ZoneType.BATTLEFIELD and bf_id:
        bf = gs.battlefields[bf_id]
        bf.units.append(inst.instance_id)
        inst.location_id = bf_id

    return inst


# ---------------------------------------------------------------------------
# Scenario — structured proof that actions produce correct frontend results
# ---------------------------------------------------------------------------

import json
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "contracts" / "state_update.schema.json"

_schema_cache: dict | None = None


def _load_schema() -> dict:
    global _schema_cache
    if _schema_cache is None:
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            _schema_cache = json.load(f)
    return _schema_cache


class Scenario:
    """A game scenario that proves actions produce correct frontend-visible results.

    Every call to ``act()`` executes an action through the same pipeline the
    real server uses (validate → execute → cleanup), then serializes the
    state for both players and validates the output against the contract
    schema.  If the serialized state drifts from the schema, the test fails
    immediately — before any manual assertions.

    Usage in a test::

        def test_playing_a_unit_costs_energy(self):
            s = Scenario()
            unit = s.add_hand_unit("p1", might=2, cost_energy=3)
            s.gs.players["p1"].rune_pool.energy = 5

            view = s.act("p1", ActionType.PLAY_CARD, {
                "instance_id": unit.instance_id,
                "targets": [],
                "destination": {"zone": "base"},
            })

            # Assert game state
            assert s.gs.players["p1"].rune_pool.energy == 2  # 5 - 3

            # Assert frontend view
            assert view["you"]["rune_pool"]["energy"] == 2
    """

    def __init__(self, num_battlefields: int = 2, phase: Phase = Phase.ACTION):
        from app.engine.action_executor import execute_action
        from app.engine.action_validator import validate_action
        from app.protocol.serializers import serialize_for_player

        self.gs = make_game(num_battlefields=num_battlefields, phase=phase)
        self.schema = _load_schema()
        self._execute_action = execute_action
        self._validate_action = validate_action
        self._serialize = serialize_for_player
        self._action_log: list[dict] = []

    # -- Action execution ------------------------------------------------

    def act(
        self,
        player_id: str,
        action_type: "ActionType",
        payload: dict | None = None,
    ) -> dict:
        """Execute an action and return the actor's validated frontend view.

        Steps:
          1. Validate the action (same checks the real server runs)
          2. Execute the action (mutates game state, runs cleanup)
          3. Serialize state for BOTH players
          4. Validate both serialized views against the contract schema
          5. Return the acting player's view

        Raises ``AssertionError`` if validation rejects the action.
        Raises ``jsonschema.ValidationError`` if the serialized state
        violates the contract.
        """
        import jsonschema as _js

        payload = payload or {}

        # 1. Validate
        result = self._validate_action(
            self.gs, player_id, action_type, payload,
        )
        assert result.ok, (
            f"Action {action_type.value} rejected: {result.error} "
            f"(phase={self.gs.phase.value}, active={self.gs.active_player_id})"
        )

        # 2. Execute
        logs = self._execute_action(
            self.gs, player_id, action_type, payload,
        )
        self._action_log.append({
            "player": player_id,
            "action": action_type.value,
            "payload": payload,
            "logs": logs,
        })

        # 3 & 4. Serialize + validate both views
        views: dict[str, dict] = {}
        for pid in self.gs.player_order:
            view = self._serialize(self.gs, pid)
            _js.validate(instance=view, schema=self.schema)
            views[pid] = view

        # 5. Return actor's view
        return views[player_id]

    def view(self, player_id: str) -> dict:
        """Get the current serialized view for a player (no action taken)."""
        return self._serialize(self.gs, player_id)

    # -- Setup helpers ---------------------------------------------------

    @property
    def bf_ids(self) -> list[str]:
        """All battlefield instance IDs."""
        return list(self.gs.battlefields.keys())

    @property
    def bf0(self) -> str:
        """First battlefield ID (convenience)."""
        return self.bf_ids[0]

    def add_hand_unit(
        self,
        player_id: str,
        might: int = 3,
        name: str = "Hand Unit",
        cost_energy: int = 2,
        keywords: tuple[KeywordInstance, ...] = (),
        abilities: tuple = (),
    ) -> CardInstance:
        """Create a unit in a player's hand (ready to be played)."""
        defn = CardDefinition(
            card_id=f"test_{name.lower().replace(' ', '_')}",
            name=name,
            card_type=CardType.UNIT,
            base_might=might,
            cost_energy=cost_energy,
            keywords=keywords,
            abilities=abilities,
        )
        inst = CardInstance.create(defn, player_id, ZoneType.HAND)
        inst.location_id = player_id
        self.gs.instances[inst.instance_id] = inst
        self.gs.players[player_id].hand.append(inst.instance_id)
        return inst

    def add_hand_spell(
        self,
        player_id: str,
        name: str = "Hand Spell",
        cost_energy: int = 2,
        effect_ir: dict | None = None,
        targets_required: int = 0,
        keywords: tuple[KeywordInstance, ...] = (),
    ) -> CardInstance:
        """Create a spell in a player's hand (ready to be played)."""
        from app.engine.enums import AbilityType as _AT
        abilities = ()
        if effect_ir is not None:
            abilities = (
                AbilityDefinition(
                    ability_id=f"test_{name.lower().replace(' ', '_')}_ab",
                    ability_type=_AT.ACTIVATED,
                    effect_ir=effect_ir,
                    targets_required=targets_required,
                ),
            )
        defn = CardDefinition(
            card_id=f"test_{name.lower().replace(' ', '_')}",
            name=name,
            card_type=CardType.SPELL,
            cost_energy=cost_energy,
            keywords=keywords,
            abilities=abilities,
        )
        inst = CardInstance.create(defn, player_id, ZoneType.HAND)
        inst.location_id = player_id
        self.gs.instances[inst.instance_id] = inst
        self.gs.players[player_id].hand.append(inst.instance_id)
        return inst

    def add_hand_gear(
        self,
        player_id: str,
        name: str = "Hand Gear",
        cost_energy: int = 1,
        might_bonus: int = 2,
    ) -> CardInstance:
        """Create a gear card in a player's hand."""
        defn = CardDefinition(
            card_id=f"test_{name.lower().replace(' ', '_')}",
            name=name,
            card_type=CardType.GEAR,
            cost_energy=cost_energy,
            might_bonus=might_bonus,
        )
        inst = CardInstance.create(defn, player_id, ZoneType.HAND)
        inst.location_id = player_id
        self.gs.instances[inst.instance_id] = inst
        self.gs.players[player_id].hand.append(inst.instance_id)
        return inst

    def add_deck_cards(self, player_id: str, count: int = 5) -> list[CardInstance]:
        """Add filler cards to a player's deck so they can draw."""
        cards = []
        for i in range(count):
            defn = make_unit_def(name=f"Deck Filler {i}")
            inst = CardInstance.create(defn, player_id, ZoneType.MAIN_DECK)
            self.gs.instances[inst.instance_id] = inst
            self.gs.players[player_id].main_deck.append(inst.instance_id)
            cards.append(inst)
        return cards

    def give_resources(
        self,
        player_id: str,
        energy: int = 10,
        power: dict[Domain, int] | None = None,
    ) -> None:
        """Give a player resources to spend."""
        ps = self.gs.players[player_id]
        ps.rune_pool.energy = energy
        if power:
            for dom, amt in power.items():
                ps.rune_pool.power[dom] = amt

    # -- View query helpers ----------------------------------------------

    @staticmethod
    def find_card_in_view(view: dict, instance_id: str) -> dict | None:
        """Search a player's view for a card by instance_id.

        Looks in: hand, base_units, base_gear, rune_board, trash, legend,
        champion, all battlefield units, all facedown cards, chain items.
        Returns the card dict if found, None otherwise.
        """
        # Player zones
        for zone_key in ("hand", "base_units", "base_gear", "rune_board", "trash"):
            for card in view["you"].get(zone_key, []):
                if card.get("instance_id") == instance_id:
                    return card
            for card in view["opponent"].get(zone_key, []):
                if card.get("instance_id") == instance_id:
                    return card

        # Legend / champion
        for side in ("you", "opponent"):
            for slot in ("legend", "champion"):
                c = view[side].get(slot)
                if c and c.get("instance_id") == instance_id:
                    return c

        # Battlefields
        for bf in view["battlefields"].values():
            for card in bf.get("units", []):
                if card.get("instance_id") == instance_id:
                    return card
            fd = bf.get("facedown_card")
            if fd and fd.get("instance_id") == instance_id:
                return fd

        # Chain
        for item in view["chain"].get("items", []):
            c = item.get("card")
            if c and c.get("instance_id") == instance_id:
                return c

        return None
