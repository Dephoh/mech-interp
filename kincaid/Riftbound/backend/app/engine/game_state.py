"""Central game state: all mutable data for one game."""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from .card_types import CardDefinition, CardInstance
from .enums import (
    AbilityType,
    CardType,
    CombatRole,
    ControlStatus,
    Domain,
    Phase,
    TurnState,
    ZoneType,
)

logger = logging.getLogger("riftbound.game_state")


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
class ActiveModifier:
    """A continuous modifier currently active on the board.

    Tracked on GameState so derived stats can be recalculated dynamically
    without mutating base CardDefinition data.
    """

    source_instance_id: str      # card that owns this modifier
    ability_id: str              # which ability on the source
    stat: str                    # "might", "shield", "assault"
    amount: int                  # signed adjustment
    target_spec: dict[str, Any]  # serialized TargetSpec dict
    exclude_self: bool = False
    duration: str = "continuous"


@dataclass
class ActiveReplacement:
    """A replacement effect currently active on the board.

    When the watched event is about to occur and the condition evaluates
    to true, the original effect is replaced with `alternative_ir`.
    """

    source_instance_id: str       # card that owns this replacement
    ability_id: str               # which ability on the source
    watch_event: str              # primitive type or game event to intercept
    condition: dict[str, Any]     # serialized ConditionSpec
    alternative_ir: dict[str, Any]  # IR subtree to execute instead
    target_spec: dict[str, Any]   # who this replacement protects


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

    # Active continuous modifiers (auras) and replacement effects
    active_modifiers: list[ActiveModifier] = field(default_factory=list)
    active_replacements: list[ActiveReplacement] = field(default_factory=list)

    # Choice state machine (set when effect_resolver hits a player_choice node)
    pending_choice: dict[str, Any] | None = None

    # Resolution context: did the last primitive/optional in a sequence succeed?
    last_effect_succeeded: bool = False

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

    # ------------------------------------------------------------------
    # Legend protections (rules 170.2.a, 170.3, 170.4)
    # ------------------------------------------------------------------

    @staticmethod
    def can_play_target(target: CardInstance) -> bool:
        """Return True if the target card can be played.

        Rule 170.2.a: Legends are not played during the course of regular play.
        """
        return target.can_be_played

    @staticmethod
    def can_kill_target(target: CardInstance) -> bool:
        """Return True if the target can be killed.

        Rule 170.3: Legends cannot be Killed during the course of regular play.
        """
        if target.is_legend:
            return False
        return True

    @staticmethod
    def can_move_target(target: CardInstance) -> bool:
        """Return True if the target can be moved.

        Rule 170.4: Legends cannot be Moved.
        """
        if target.is_legend:
            return False
        return True

    # ------------------------------------------------------------------
    # Zone transition helpers (rule 719.5)
    # ------------------------------------------------------------------

    def move_to_zone(
        self,
        instance_id: str,
        new_zone: ZoneType,
        new_location: str | None = None,
    ) -> list[str]:
        """Move a card to a new zone, handling detachment and state reset.

        Rule 719.5: When a Top-Most Card changes zones from a board zone
        to a non-board zone, all Attached cards Detach, remaining in their
        current zones.

        Returns log messages.
        """
        inst = self.get_instance(instance_id)
        if not inst:
            return []

        logs: list[str] = []
        was_on_board = inst.is_on_board

        # Rule 719.5: detach all attached cards if leaving the board
        if was_on_board and new_zone not in CardInstance._BOARD_ZONES:
            if inst.attached_cards:
                detached = inst.detach_all(self.instances)
                for did in detached:
                    gear = self.get_instance(did)
                    if gear:
                        logs.append(f"{gear.name} detached from {inst.name} (left board)")

        # Reset mutable state when leaving play
        if was_on_board and new_zone not in CardInstance._BOARD_ZONES:
            inst.reset_on_zone_exit()

        inst.zone = new_zone
        inst.location_id = new_location
        return logs

    # ------------------------------------------------------------------
    # Modifier registration & recalculation
    # ------------------------------------------------------------------

    def register_modifier(self, source: CardInstance, ability_id: str, ir_node: dict) -> None:
        """Register an active modifier from a card entering the board."""
        mod = ActiveModifier(
            source_instance_id=source.instance_id,
            ability_id=ability_id,
            stat=ir_node.get("stat", "might"),
            amount=ir_node.get("amount", 0),
            target_spec=ir_node.get("target", {}),
            exclude_self=ir_node.get("exclude_self", False),
            duration=ir_node.get("duration", "continuous"),
        )
        self.active_modifiers.append(mod)

    def register_replacement(self, source: CardInstance, ability_id: str, ir_node: dict) -> None:
        """Register an active replacement effect from a card entering the board."""
        rep = ActiveReplacement(
            source_instance_id=source.instance_id,
            ability_id=ability_id,
            watch_event=ir_node.get("watch_event", ""),
            condition=ir_node.get("condition", {"cond_type": "always"}),
            alternative_ir=ir_node.get("alternative_ir", {}),
            target_spec=ir_node.get("target", {}),
        )
        self.active_replacements.append(rep)

    def unregister_card(self, instance_id: str) -> None:
        """Remove all modifiers and replacements owned by a card leaving the board."""
        self.active_modifiers = [
            m for m in self.active_modifiers if m.source_instance_id != instance_id
        ]
        self.active_replacements = [
            r for r in self.active_replacements if r.source_instance_id != instance_id
        ]

    def recalculate_modifiers(self) -> None:
        """Recompute aura_might_bonus on every card from active continuous modifiers.

        Called during cleanup and after any board change. Never mutates
        base card definitions — only sets the derived `aura_might_bonus` field
        on CardInstance.
        """
        from .effect_ir import TargetSpec

        # 1. Zero all aura and gear bonuses
        for inst in self.instances.values():
            inst.aura_might_bonus = 0
            inst.gear_might_bonus = 0

        # 2. Prune modifiers whose source has left the board
        board_ids = self._get_board_instance_ids()
        self.active_modifiers = [
            m for m in self.active_modifiers if m.source_instance_id in board_ids
        ]

        # Also prune replacements
        self.active_replacements = [
            r for r in self.active_replacements if r.source_instance_id in board_ids
        ]

        # 3. Apply each active modifier to matching targets
        for mod in self.active_modifiers:
            if mod.stat != "might":
                continue  # only might recalculation for now

            source = self.get_instance(mod.source_instance_id)
            if not source:
                continue

            spec = TargetSpec.from_dict(mod.target_spec)
            matched = self._resolve_modifier_targets(spec, source, mod.exclude_self)
            for iid in matched:
                inst = self.get_instance(iid)
                if inst:
                    inst.aura_might_bonus += mod.amount

        # 4. Apply gear attachment might bonuses
        for inst in self.instances.values():
            if inst.attached_cards:
                for gear_id in inst.attached_cards:
                    gear = self.get_instance(gear_id)
                    if gear and gear.definition.might_bonus:
                        inst.gear_might_bonus += gear.definition.might_bonus

    def _get_board_instance_ids(self) -> set[str]:
        """Return instance_ids of all cards currently on the board.

        Includes: base units/gear/runes, battlefield units, battlefield cards
        themselves, legend zone, and champion zone.
        """
        ids: set[str] = set()
        for uid_list in self.base_units.values():
            ids.update(uid_list)
        for bf in self.battlefields.values():
            ids.update(bf.units)
            # Battlefield cards themselves are board objects
            ids.add(bf.card_instance_id)
            # Facedown cards at battlefields
            if bf.facedown_card:
                ids.add(bf.facedown_card)
        for gid_list in self.base_gear.values():
            ids.update(gid_list)
        for rid_list in self.base_runes.values():
            ids.update(rid_list)
        for ps in self.players.values():
            if ps.legend_zone:
                ids.add(ps.legend_zone)
            if ps.champion_zone:
                ids.add(ps.champion_zone)
        return ids

    def _resolve_modifier_targets(
        self,
        spec: Any,  # TargetSpec
        source: CardInstance,
        exclude_self: bool,
    ) -> list[str]:
        """Resolve which board objects a modifier applies to."""
        controller = source.controller_id
        candidates: list[str] = []

        if spec.obj_type in ("unit", "permanent", "card"):
            for uid_list in self.base_units.values():
                candidates.extend(uid_list)
            for bf in self.battlefields.values():
                candidates.extend(bf.units)

        if spec.obj_type in ("gear", "permanent", "card"):
            for gid_list in self.base_gear.values():
                candidates.extend(gid_list)

        matched: list[str] = []
        for cid in candidates:
            card = self.get_instance(cid)
            if not card:
                continue
            if exclude_self and cid == source.instance_id:
                continue
            if spec.scope == "friendly" and card.controller_id != controller:
                continue
            if spec.scope == "enemy" and card.controller_id == controller:
                continue
            if spec.scope == "self" and cid != source.instance_id:
                continue
            if spec.location == "here" and source.location_id:
                if card.location_id != source.location_id:
                    continue
            matched.append(cid)

        return matched


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
