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

    def keyword_value(self, kw: Keyword) -> int:
        """Sum all values for a given keyword (e.g. multiple Assault sources)."""
        return sum(k.value for k in self.all_keywords() if k.keyword == kw)

    @property
    def effective_might(self) -> int:
        """Current might accounting for buffs, modifiers, Assault/Shield, but NOT stun."""
        base = self.definition.base_might
        # Buff counter
        if self.buff_counter:
            base += 1
        # Attachment might bonuses
        # (handled externally by the engine looking at attached_cards)
        # Transient modifiers
        base += sum(self.might_modifiers)
        # Assault (only while attacker)
        if self.combat_role == CombatRole.ATTACKER:
            base += self.keyword_value(Keyword.ASSAULT)
        # Shield (only while defender)
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
        """True if unit has no lethal damage marked."""
        if self.card_type != CardType.UNIT:
            return True
        return self.damage < self.effective_might or self.effective_might == 0

    def clear_turn_state(self) -> None:
        """Called at end of turn to clear transient state."""
        self.granted_keywords.clear()
        self.might_modifiers.clear()
        self.entered_this_turn = False
        self.stunned = False  # stun clears at Ending Step

    def heal(self) -> None:
        self.damage = 0
