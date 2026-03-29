"""Card definitions (immutable templates) and card instances (mutable runtime state)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .enums import (
    AbilityType,
    CardType,
    CombatRole,
    Domain,
    Keyword,
    SuperType,
    ZoneType,
)


# ---------------------------------------------------------------------------
# Immutable definitions (loaded from JSON, never mutated during play)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeywordInstance:
    """A keyword with an optional numeric value (e.g. Assault 2, Shield 1)."""

    keyword: Keyword
    value: int = 0  # used for ASSAULT[X], SHIELD[X], DEFLECT[X], REPEAT cost, etc.


@dataclass(frozen=True)
class CostDefinition:
    energy: int = 0
    power: tuple[tuple[Domain, int], ...] = ()  # ((Domain.FURY, 1),)
    exhaust_source: bool = False
    additional_costs: tuple[str, ...] = ()  # e.g. ("kill_friendly_unit",)

    def power_dict(self) -> dict[Domain, int]:
        return dict(self.power)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CostDefinition:
        power = tuple(
            (Domain(k), v) for k, v in d.get("power", {}).items()
        )
        return CostDefinition(
            energy=d.get("energy", 0),
            power=power,
            exhaust_source=d.get("exhaust_source", False),
            additional_costs=tuple(d.get("additional_costs", ())),
        )


@dataclass(frozen=True)
class AbilityDefinition:
    ability_id: str
    ability_type: AbilityType
    trigger_condition: str | None = None  # e.g. "on_play", "on_conquer", "on_death"
    cost: CostDefinition | None = None
    effect_script: str | None = None  # registered function name in effects.py
    effect_ir: dict | None = None     # composable effect IR tree (see effect_ir.py)
    timing: str = "default"  # "default", "action", "reaction"
    text: str = ""
    # Target validation (rule 352.7): how many targets must be chosen
    targets_required: int = 0  # 0 = no targets needed, 1+ = must have that many valid targets
    target_type: str = ""     # e.g. "unit", "unit_at_battlefield", "friendly_unit", "spell_on_chain"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> AbilityDefinition:
        cost = CostDefinition.from_dict(d["cost"]) if d.get("cost") else None
        return AbilityDefinition(
            ability_id=d["ability_id"],
            ability_type=AbilityType(d["ability_type"]),
            trigger_condition=d.get("trigger_condition"),
            cost=cost,
            effect_script=d.get("effect_script"),
            effect_ir=d.get("effect_ir"),
            timing=d.get("timing", "default"),
            text=d.get("text", ""),
            targets_required=d.get("targets_required", 0),
            target_type=d.get("target_type", ""),
        )


@dataclass(frozen=True)
class CardDefinition:
    """Immutable card template loaded from the card database."""

    card_id: str
    name: str
    card_type: CardType
    domains: tuple[Domain, ...] = ()
    supertypes: tuple[SuperType, ...] = ()
    cost_energy: int = 0
    cost_power: tuple[tuple[Domain, int], ...] = ()
    base_might: int = 0
    keywords: tuple[KeywordInstance, ...] = ()
    abilities: tuple[AbilityDefinition, ...] = ()
    tags: tuple[str, ...] = ()
    text: str = ""
    effect_text: str = ""
    might_bonus: int = 0  # for cards with Might Bonus (attachment)
    art_path: str = ""

    def cost_power_dict(self) -> dict[Domain, int]:
        return dict(self.cost_power)

    def has_keyword(self, kw: Keyword) -> bool:
        return any(k.keyword == kw for k in self.keywords)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CardDefinition:
        domains = tuple(Domain(x) for x in d.get("domains", []))
        supertypes = tuple(SuperType(x) for x in d.get("supertypes", []))
        cost_power = tuple(
            (Domain(k), v) for k, v in d.get("cost_power", {}).items()
        )
        keywords = tuple(
            KeywordInstance(
                keyword=Keyword(kw["keyword"]),
                value=kw.get("value", 0),
            )
            for kw in d.get("keywords", [])
        )
        abilities = tuple(
            AbilityDefinition.from_dict(ab)
            for ab in d.get("abilities", [])
        )
        return CardDefinition(
            card_id=d["card_id"],
            name=d["name"],
            card_type=CardType(d["card_type"]),
            domains=domains,
            supertypes=supertypes,
            cost_energy=d.get("cost_energy", 0),
            cost_power=cost_power,
            base_might=d.get("base_might", 0),
            keywords=keywords,
            abilities=abilities,
            tags=tuple(d.get("tags", [])),
            text=d.get("text", ""),
            effect_text=d.get("effect_text", ""),
            might_bonus=d.get("might_bonus", 0),
            art_path=d.get("art_path", ""),
        )


# ---------------------------------------------------------------------------
# Mutable runtime instance (one per physical card in a game)
# ---------------------------------------------------------------------------


@dataclass
class CardInstance:
    """A single card in play, with mutable game state."""

    instance_id: str
    definition: CardDefinition
    owner_id: str
    controller_id: str
    zone: ZoneType
    location_id: str | None = None  # battlefield_id or player_id

    exhausted: bool = False
    damage: int = 0
    buff_counter: bool = False  # max 1 buff per unit
    stunned: bool = False
    combat_role: CombatRole = CombatRole.NONE
    facedown: bool = False
    entered_this_turn: bool = True
    hidden_at_battlefield: str | None = None  # battlefield_id if hidden
    hidden_ready: bool = False  # True after surviving one turn facedown
    accelerated: bool = False  # True if Accelerate cost was paid
    attached_to: str | None = None  # instance_id of top-most card
    attached_cards: list[str] = field(default_factory=list)  # instance_ids attached to this

    # Transient keyword grants (expire end of turn)
    granted_keywords: list[KeywordInstance] = field(default_factory=list)
    # Transient might modifiers (expire end of turn)
    might_modifiers: list[int] = field(default_factory=list)
    # Continuous modifier bonus (recalculated dynamically from active auras)
    aura_might_bonus: int = 0
    # Gear attachment might bonus (recalculated dynamically from attached gear)
    gear_might_bonus: int = 0

    @staticmethod
    def create(
        definition: CardDefinition,
        owner_id: str,
        zone: ZoneType,
        location_id: str | None = None,
    ) -> CardInstance:
        return CardInstance(
            instance_id=str(uuid.uuid4()),
            definition=definition,
            owner_id=owner_id,
            controller_id=owner_id,
            zone=zone,
            location_id=location_id,
        )

    @property
    def card_type(self) -> CardType:
        return self.definition.card_type

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def card_id(self) -> str:
        return self.definition.card_id

    def all_keywords(self) -> list[KeywordInstance]:
        return list(self.definition.keywords) + self.granted_keywords

    def has_keyword(self, kw: Keyword) -> bool:
        return any(k.keyword == kw for k in self.all_keywords())

    # Keywords where an omitted value (0) is presumed to be 1
    # per rules 733.1.b.3 (Assault) and 740.1.b.3 (Shield).
    _DEFAULT_ONE_KEYWORDS = frozenset({Keyword.ASSAULT, Keyword.SHIELD})

    def keyword_value(self, kw: Keyword) -> int:
        """Sum all values for a given keyword (e.g. multiple Assault sources).

        For Assault and Shield, a stored value of 0 means the X was omitted
        and is presumed to be 1 (rules 733.1.b.3, 740.1.b.3).
        """
        default_one = kw in self._DEFAULT_ONE_KEYWORDS
        total = 0
        for k in self.all_keywords():
            if k.keyword == kw:
                total += k.value if k.value != 0 else (1 if default_one else 0)
        return total

    # ------------------------------------------------------------------
    # Board zones (rule 710 vs 711)
    # ------------------------------------------------------------------

    _BOARD_ZONES = frozenset({
        ZoneType.BASE,
        ZoneType.BATTLEFIELD,
        ZoneType.FACEDOWN_ZONE,
        ZoneType.RUNE_BOARD,
        ZoneType.LEGEND_ZONE,
        ZoneType.CHAMPION_ZONE,
    })

    @property
    def is_on_board(self) -> bool:
        """True if the card is in a board zone (rule 710)."""
        return self.zone in self._BOARD_ZONES

    @property
    def inherent_might(self) -> int:
        """Printed/base might with no modifiers (rule 711 — non-board evaluation)."""
        return self.definition.base_might

    @property
    def effective_might(self) -> int:
        """Current might accounting for buffs, modifiers, auras, Assault/Shield.

        Rule 710: Units on the board are evaluated according to their current might.
        Rule 711: Units in non-board zones are evaluated according to their inherent might.
        """
        # Non-board zones use inherent (printed) might only (rule 711)
        if not self.is_on_board:
            return max(0, self.definition.base_might)

        base = self.definition.base_might
        # Buff counter (rule 703: each buff +1 might)
        if self.buff_counter:
            base += 1
        # Attachment might bonuses from gear (rule 718.4)
        base += self.gear_might_bonus
        # Transient modifiers (spell effects etc.)
        base += sum(self.might_modifiers)
        # Continuous aura modifiers (recalculated by GameState.recalculate_modifiers)
        base += self.aura_might_bonus
        # Assault (only while attacker, rule 733)
        if self.combat_role == CombatRole.ATTACKER:
            base += self.keyword_value(Keyword.ASSAULT)
        # Shield (only while defender, rule 740)
        if self.combat_role == CombatRole.DEFENDER:
            base += self.keyword_value(Keyword.SHIELD)
        return max(0, base)

    @property
    def combat_might(self) -> int:
        """Might contributed to combat damage (0 if stunned)."""
        if self.stunned:
            return 0
        return self.effective_might

    @property
    def is_alive(self) -> bool:
        """True if unit is on the board and has no lethal damage.

        Rule 415: A unit in the trash or banishment is not alive.
        A unit with 0 effective might is alive (cannot be killed by damage).
        """
        if self.card_type != CardType.UNIT:
            return True
        # A unit in trash or banishment is dead
        if self.zone in (ZoneType.TRASH, ZoneType.BANISHMENT):
            return False
        return self.damage < self.effective_might or self.effective_might == 0

    # ------------------------------------------------------------------
    # Mighty status (rules 707-709)
    # ------------------------------------------------------------------

    @property
    def is_mighty(self) -> bool:
        """True if this unit's might is 5 or greater (rule 708)."""
        return self.effective_might >= 5

    def would_become_mighty(self, might_delta: int) -> bool:
        """True if applying might_delta would cause this unit to *become* Mighty.

        Rule 709: A unit 'becomes Mighty' at the moment its Might changes
        from being less than 5 to being 5 or greater.
        """
        current = self.effective_might
        new = current + might_delta
        return current < 5 and new >= 5

    # ------------------------------------------------------------------
    # Legend / Token identity helpers (rules 170-179)
    # ------------------------------------------------------------------

    @property
    def is_legend(self) -> bool:
        """True if this card is a Legend (rule 170)."""
        return self.card_type == CardType.LEGEND

    @property
    def is_token(self) -> bool:
        """True if this card is a Token (rule 179)."""
        return SuperType.TOKEN in self.definition.supertypes

    @property
    def is_permanent(self) -> bool:
        """True if this card is a permanent (unit or gear on the board).

        Legends are NOT permanents (rule 171)."""
        return self.card_type in (CardType.UNIT, CardType.GEAR) and not self.is_legend

    # ------------------------------------------------------------------
    # Attachment / Inactive text (rules 718, 720-724)
    # ------------------------------------------------------------------

    @property
    def is_attached(self) -> bool:
        """True if this card is attached to another card (rule 718)."""
        return self.attached_to is not None

    @property
    def rules_text_active(self) -> bool:
        """True if this card's own rules text is active.

        Rule 718.2: While attached, the card's Rules Text is Inactive.
        Rule 723: Rules Text is never Inactive by default.
        """
        return not self.is_attached

    @property
    def effect_text_active(self) -> bool:
        """True if this card's effect text is active.

        Rule 724: Effect Text is Inactive unless the card is Attached.
        """
        return self.is_attached

    # ------------------------------------------------------------------
    # Playability (rule 170.2.a — Legends cannot be played)
    # ------------------------------------------------------------------

    @property
    def can_be_played(self) -> bool:
        """True if this card is allowed to be played during regular play.

        Rule 170.2.a: Legends are not played during the course of regular play.
        Rule 170.2.b: Legends are established at start and remain in place.
        """
        if self.is_legend:
            return False
        return True

    # ------------------------------------------------------------------
    # Token cost overrides (rules 179.2.a, 179.2.b)
    # ------------------------------------------------------------------

    @property
    def play_cost_energy(self) -> int:
        """Energy cost to play this card. Tokens have no costs (rule 179.2.a)."""
        if self.is_token:
            return 0
        return self.definition.cost_energy

    @property
    def play_cost_power(self) -> dict[Domain, int]:
        """Power cost to play this card. Tokens have no costs (rule 179.2.a)."""
        if self.is_token:
            return {}
        return self.definition.cost_power_dict()

    # ------------------------------------------------------------------
    # Detachment (rule 719.5)
    # ------------------------------------------------------------------

    def detach_all(self, instances: dict[str, CardInstance]) -> list[str]:
        """Detach all cards attached to this top-most card.

        Rule 719.5: When a Top-Most Card changes zones from a board zone
        to a non-board zone, all Attached cards Detach from it, remaining
        in their current zones.

        Returns instance_ids of detached cards.
        """
        detached: list[str] = []
        for gear_id in list(self.attached_cards):
            gear = instances.get(gear_id)
            if gear:
                gear.attached_to = None
                detached.append(gear_id)
        self.attached_cards.clear()
        return detached

    # ------------------------------------------------------------------
    # Privacy (rules 127.2-127.5)
    # ------------------------------------------------------------------

    @property
    def privacy_level(self) -> str:
        """Return the privacy level of this card based on its zone.

        Rule 127.3: Secret — neither player may look at the face.
        Rule 127.4: Private — only the controller (board) or owner (other zones).
        Rule 127.5: Public — any player may look at the face.
        """
        if self.zone in (ZoneType.MAIN_DECK, ZoneType.RUNE_DECK):
            # Cards in decks are Secret (127.3)
            return "secret"
        if self.zone == ZoneType.HAND:
            # Cards in hand are Private to the owner (127.4)
            return "private"
        if self.zone == ZoneType.FACEDOWN_ZONE:
            # Facedown cards are Private to the controller (127.4, 128.4)
            return "private"
        if self.zone in (ZoneType.TRASH, ZoneType.BANISHMENT):
            # Cards in trash/banishment are face-up, Public (129.6)
            return "public"
        if self.zone in (
            ZoneType.BASE, ZoneType.BATTLEFIELD, ZoneType.RUNE_BOARD,
            ZoneType.LEGEND_ZONE, ZoneType.CHAMPION_ZONE, ZoneType.CHAIN,
        ):
            # Cards on the board / chain are face-up, Public (129.4-129.5)
            return "public"
        return "public"

    def clear_turn_state(self) -> None:
        """Called at end of turn to clear transient state."""
        self.granted_keywords.clear()
        self.might_modifiers.clear()
        self.entered_this_turn = False
        self.stunned = False  # stun clears at Ending Step

    def reset_on_zone_exit(self) -> None:
        """Clear all mutable state when a card leaves play.

        Rule 705: If a Unit leaves play, remove all Buffs from it.
        Rule 705.1: Champions do not retain Buffs in the Champion Zone.
        Rule 719.5: When a Top-Most Card changes zones from board to non-board,
                     all attached cards detach (handled by caller via detach_all).
        """
        self.damage = 0
        self.buff_counter = False
        self.exhausted = False
        self.stunned = False
        self.combat_role = CombatRole.NONE
        self.facedown = False
        self.hidden_at_battlefield = None
        self.hidden_ready = False
        self.granted_keywords.clear()
        self.might_modifiers.clear()
        self.aura_might_bonus = 0
        self.gear_might_bonus = 0
        self.entered_this_turn = False
        # Clear attachment state (rule 719.5)
        self.attached_to = None
        self.attached_cards.clear()

    def heal(self) -> None:
        self.damage = 0
