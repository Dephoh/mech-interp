"""Central game state: all mutable data for one game."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

from .card_types import CardDefinition, CardInstance
from .enums import (
    CombatRole,
    ControlStatus,
    Domain,
    Phase,
    TurnState,
    ZoneType,
)


@dataclass
class RunePool:
    """A player's available resources to spend on costs."""

    energy: int = 0
    power: dict[Domain, int] = field(default_factory=dict)

    def add_energy(self, amount: int) -> None:
        self.energy += amount

    def add_power(self, domain: Domain, amount: int = 1) -> None:
        self.power[domain] = self.power.get(domain, 0) + amount

    def total_power(self) -> int:
        return sum(self.power.values())

    def can_pay(self, energy: int, power: dict[Domain, int]) -> bool:
        if self.energy < energy:
            return False
        for domain, needed in power.items():
            if self.power.get(domain, 0) < needed:
                return False
        return True

    def spend(self, energy: int, power: dict[Domain, int]) -> None:
        self.energy -= energy
        for domain, amount in power.items():
            self.power[domain] -= amount

    def empty(self) -> None:
        self.energy = 0
        self.power.clear()

    def clone(self) -> RunePool:
        return RunePool(energy=self.energy, power=dict(self.power))


@dataclass
class PlayerState:
    player_id: str
    display_name: str
    score: int = 0

    # Card zones (lists of instance_ids)
    hand: list[str] = field(default_factory=list)
    main_deck: list[str] = field(default_factory=list)
    rune_deck: list[str] = field(default_factory=list)
    trash: list[str] = field(default_factory=list)
    banishment: list[str] = field(default_factory=list)

    # Special zones (single instance_id or None)
    legend_zone: str | None = None
    champion_zone: str | None = None

    rune_pool: RunePool = field(default_factory=RunePool)

    # Per-turn tracking
    battlefields_scored_this_turn: set[str] = field(default_factory=set)
    played_main_deck_this_turn: int = 0
    is_first_turn: bool = True
    goes_second: bool = False  # gets extra rune on first channel

    def reset_turn_tracking(self) -> None:
        self.battlefields_scored_this_turn.clear()
        self.played_main_deck_this_turn = 0


@dataclass
class BattlefieldState:
    battlefield_id: str
    card_instance_id: str  # the battlefield card's instance_id
    control_status: ControlStatus = ControlStatus.UNCONTROLLED
    controller_id: str | None = None
    contested_by: str | None = None  # player_id who applied contested

    # Units present here (instance_ids)
    units: list[str] = field(default_factory=list)
    # Facedown (Hidden) card
    facedown_card: str | None = None
    # Track who has scored this battlefield this turn
    scored_this_turn_by: str | None = None

    # Staged showdown / combat flags
    showdown_staged: bool = False
    combat_staged: bool = False

    def units_by_player(self, instances: dict[str, CardInstance]) -> dict[str, list[str]]:
        """Group unit instance_ids by controller_id."""
        result: dict[str, list[str]] = {}
        for uid in self.units:
            inst = instances.get(uid)
            if inst:
                result.setdefault(inst.controller_id, []).append(uid)
        return result


@dataclass
class ChainItem:
    item_id: str
    source_instance_id: str | None  # the card/permanent that created this
    ability_id: str | None  # None if the item IS the card being played
    card_instance_id: str | None  # set if item is a spell/unit/gear being played
    controller_id: str
    pending: bool = True  # True until finalized
    targets: list[str] = field(default_factory=list)

    @staticmethod
    def create(
        controller_id: str,
        source_instance_id: str | None = None,
        ability_id: str | None = None,
        card_instance_id: str | None = None,
        targets: list[str] | None = None,
    ) -> ChainItem:
        return ChainItem(
            item_id=str(uuid.uuid4()),
            source_instance_id=source_instance_id,
            ability_id=ability_id,
            card_instance_id=card_instance_id,
            controller_id=controller_id,
            targets=targets or [],
        )


@dataclass
class ChainState:
    stack: list[ChainItem] = field(default_factory=list)
    passed_players: set[str] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return len(self.stack) == 0

    @property
    def is_closed(self) -> bool:
        return not self.is_empty

    def push(self, item: ChainItem) -> None:
        self.stack.append(item)
        self.passed_players.clear()

    def pop(self) -> ChainItem:
        return self.stack.pop()

    def peek(self) -> ChainItem | None:
        return self.stack[-1] if self.stack else None

    def reset_passes(self) -> None:
        self.passed_players.clear()


@dataclass
class CombatState:
    battlefield_id: str
    attacker_id: str
    defender_id: str
    phase: str = "showdown"  # "showdown" | "waiting_assignments" | "resolution"

    # Damage assignments: player_id -> {target_instance_id: damage}
    attacker_assignment: dict[str, int] | None = None
    defender_assignment: dict[str, int] | None = None


@dataclass
class ShowdownState:
    battlefield_id: str
    initiator_id: str  # player who applied contested
    focus_player_id: str  # who currently has focus
    passed_players: set[str] = field(default_factory=set)
    is_combat_showdown: bool = False  # True if part of Combat flow


@dataclass
class GameLog:
    entries: list[dict] = field(default_factory=list)

    def add(self, message: str, **kwargs: object) -> None:
        entry = {"message": message, **kwargs}
        self.entries.append(entry)


@dataclass
class GameState:
    """The complete, authoritative state of one game."""

    game_id: str
    players: dict[str, PlayerState] = field(default_factory=dict)
    player_order: list[str] = field(default_factory=list)  # [p0_id, p1_id]
    turn_player_id: str = ""
    active_player_id: str = ""  # who has priority/focus right now
    phase: Phase = Phase.SETUP_MULLIGAN
    turn_number: int = 0

    # All card instances in the game
    instances: dict[str, CardInstance] = field(default_factory=dict)

    # Board state
    battlefields: dict[str, BattlefieldState] = field(default_factory=dict)
    base_units: dict[str, list[str]] = field(default_factory=dict)  # player_id -> instance_ids
    base_gear: dict[str, list[str]] = field(default_factory=dict)
    base_runes: dict[str, list[str]] = field(default_factory=dict)  # channeled runes

    # Chain
    chain: ChainState = field(default_factory=ChainState)

    # Combat / Showdown
    active_combat: CombatState | None = None
    active_showdown: ShowdownState | None = None
    pending_combats: list[str] = field(default_factory=list)  # battlefield_ids
    pending_showdowns: list[str] = field(default_factory=list)

    # Game result
    game_over: bool = False
    winner_id: str | None = None

    # Mulligan tracking
    mulligan_done: dict[str, bool] = field(default_factory=dict)

    # Log
    log: GameLog = field(default_factory=GameLog)

    # Mode of play constants
    victory_score: int = 8

    def get_turn_state(self) -> TurnState:
        in_showdown = (
            self.active_showdown is not None
            or self.phase in (Phase.SHOWDOWN, Phase.COMBAT_DAMAGE, Phase.COMBAT_RESOLUTION)
        )
        chain_exists = not self.chain.is_empty
        if in_showdown:
            return TurnState.SHOWDOWN_CLOSED if chain_exists else TurnState.SHOWDOWN_OPEN
        return TurnState.NEUTRAL_CLOSED if chain_exists else TurnState.NEUTRAL_OPEN

    def opponent_id(self, player_id: str) -> str:
        for pid in self.player_order:
            if pid != player_id:
                return pid
        raise ValueError(f"No opponent for {player_id}")

    def get_instance(self, instance_id: str) -> CardInstance | None:
        return self.instances.get(instance_id)

    def get_battlefield_for_unit(self, instance_id: str) -> BattlefieldState | None:
        for bf in self.battlefields.values():
            if instance_id in bf.units:
                return bf
        return None

    def units_at_battlefield(self, bf_id: str, player_id: str | None = None) -> list[CardInstance]:
        bf = self.battlefields.get(bf_id)
        if not bf:
            return []
        units = [self.instances[uid] for uid in bf.units if uid in self.instances]
        if player_id:
            units = [u for u in units if u.controller_id == player_id]
        return units


def create_game(
    game_id: str,
    player_configs: list[dict],
    card_db: dict[str, CardDefinition],
) -> GameState:
    """
    Initialize a new game from player configs.

    Each player_config: {
        "player_id": str,
        "display_name": str,
        "legend_id": str,         # card_id of legend
        "champion_id": str,       # card_id of chosen champion
        "main_deck": [card_id, ...],  # 40 card_ids
        "rune_deck": [card_id, ...],  # 12 card_ids
        "battlefields": [card_id, card_id, card_id],  # 3 battlefield card_ids
    }
    """
    gs = GameState(game_id=game_id)

    # Determine turn order (random)
    configs = list(player_configs)
    random.shuffle(configs)
    gs.player_order = [c["player_id"] for c in configs]
    gs.turn_player_id = gs.player_order[0]
    gs.active_player_id = gs.player_order[0]

    for i, config in enumerate(configs):
        pid = config["player_id"]
        ps = PlayerState(
            player_id=pid,
            display_name=config["display_name"],
            goes_second=(i == 1),
        )
        gs.players[pid] = ps
        gs.mulligan_done[pid] = False
        gs.base_units[pid] = []
        gs.base_gear[pid] = []
        gs.base_runes[pid] = []

        # Legend
        legend_def = card_db[config["legend_id"]]
        legend_inst = CardInstance.create(legend_def, pid, ZoneType.LEGEND_ZONE)
        gs.instances[legend_inst.instance_id] = legend_inst
        ps.legend_zone = legend_inst.instance_id

        # Chosen champion
        champ_def = card_db[config["champion_id"]]
        champ_inst = CardInstance.create(champ_def, pid, ZoneType.CHAMPION_ZONE)
        gs.instances[champ_inst.instance_id] = champ_inst
        ps.champion_zone = champ_inst.instance_id

        # Main deck (shuffle)
        deck_ids = list(config["main_deck"])
        random.shuffle(deck_ids)
        for cid in deck_ids:
            card_def = card_db[cid]
            inst = CardInstance.create(card_def, pid, ZoneType.MAIN_DECK)
            gs.instances[inst.instance_id] = inst
            ps.main_deck.append(inst.instance_id)

        # Rune deck (shuffle)
        rune_ids = list(config["rune_deck"])
        random.shuffle(rune_ids)
        for cid in rune_ids:
            card_def = card_db[cid]
            inst = CardInstance.create(card_def, pid, ZoneType.RUNE_DECK)
            gs.instances[inst.instance_id] = inst
            ps.rune_deck.append(inst.instance_id)

        # Battlefields: randomly select 1 of 3 (1v1 Duel mode)
        bf_card_ids = config["battlefields"]
        chosen_bf_cid = random.choice(bf_card_ids)
        bf_def = card_db[chosen_bf_cid]
        bf_inst = CardInstance.create(bf_def, pid, ZoneType.BATTLEFIELD)
        gs.instances[bf_inst.instance_id] = bf_inst
        bf_state = BattlefieldState(
            battlefield_id=bf_inst.instance_id,
            card_instance_id=bf_inst.instance_id,
        )
        gs.battlefields[bf_inst.instance_id] = bf_state

    # Draw 4 cards each
    for pid in gs.player_order:
        _draw_cards(gs, pid, 4)

    gs.phase = Phase.SETUP_MULLIGAN
    gs.log.add("Game started. Mulligan phase.")
    return gs


def _draw_cards(gs: GameState, player_id: str, count: int) -> list[str]:
    """Draw cards from main deck to hand. Returns drawn instance_ids."""
    ps = gs.players[player_id]
    drawn: list[str] = []
    for _ in range(count):
        if not ps.main_deck:
            # Burn out handled elsewhere
            break
        iid = ps.main_deck.pop(0)  # top of deck
        inst = gs.instances[iid]
        inst.zone = ZoneType.HAND
        inst.location_id = player_id
        ps.hand.append(iid)
        drawn.append(iid)
    return drawn
