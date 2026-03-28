"""Effect Intermediate Representation (IR) for composable card abilities.

Effect IR nodes are plain dicts with a "type" key, keeping them JSON-serializable
and easy to compose in the card pipeline. TargetSpec and ConditionSpec are
dataclasses providing structure for targeting and conditions.

The resolver in effect_resolver.py walks IR trees and calls primitives in
effect_primitives.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Primitive node type constants (leaf nodes)
# ---------------------------------------------------------------------------

DEAL_DAMAGE = "deal_damage"
DRAW_CARDS = "draw_cards"
GIVE_MIGHT = "give_might"
BUFF = "buff"
STUN = "stun"
HEAL = "heal"
KILL = "kill"
MOVE = "move"
READY = "ready"
EXHAUST = "exhaust"
DISCARD = "discard"
BANISH = "banish"
COUNTER = "counter"
RETURN_TO_HAND = "return_to_hand"
RETURN_TO_DECK = "return_to_deck"
RECYCLE = "recycle"
PLAY_TOKEN = "play_token"
ADD_ENERGY = "add_energy"
ADD_POWER = "add_power"
CHANNEL_RUNE = "channel_rune"
ATTACH = "attach"
DETACH = "detach"
SCORE_POINTS = "score_points"
GAIN_XP = "gain_xp"
SPEND_XP = "spend_xp"
LOOK_AT_TOP = "look_at_top"
GIVE_KEYWORD = "give_keyword"
RESTRICT = "restrict"

# Composition node type constants (branch nodes)
SEQUENCE = "sequence"
CONDITIONAL = "conditional"
FOR_EACH = "for_each"
CHOOSE_ONE = "choose_one"
OPTIONAL = "optional"
REPEAT_EFFECT = "repeat_effect"

# All valid node types
PRIMITIVE_TYPES = frozenset({
    DEAL_DAMAGE, DRAW_CARDS, GIVE_MIGHT, BUFF, STUN, HEAL, KILL, MOVE,
    READY, EXHAUST, DISCARD, BANISH, COUNTER, RETURN_TO_HAND, RETURN_TO_DECK,
    RECYCLE, PLAY_TOKEN, ADD_ENERGY, ADD_POWER, CHANNEL_RUNE, ATTACH, DETACH,
    SCORE_POINTS, GAIN_XP, SPEND_XP, LOOK_AT_TOP, GIVE_KEYWORD, RESTRICT,
})

COMPOSITION_TYPES = frozenset({
    SEQUENCE, CONDITIONAL, FOR_EACH, CHOOSE_ONE, OPTIONAL, REPEAT_EFFECT,
})

ALL_NODE_TYPES = PRIMITIVE_TYPES | COMPOSITION_TYPES


# ---------------------------------------------------------------------------
# TargetSpec — describes what a target must match
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSpec:
    """Specification for resolving or validating targets.

    Attributes:
        obj_type: What kind of object ("unit", "gear", "spell", "rune",
                  "card", "player", "permanent").
        scope: Ownership filter — "friendly", "enemy", "any", "self".
        zone: Zone constraint — "base", "battlefield", "chain", "hand",
              "trash", "any".
        location: Location constraint — "here" (same BF as source),
                  "any", or a specific battlefield_id.
        count: How many targets — positive int, or -1 for "all".
        filters: Additional filter predicates (see FilterSpec).
        chooser: Who picks the target — "controller", "opponent", "auto".
    """

    obj_type: str = "unit"
    scope: str = "any"
    zone: str = "any"
    location: str = "any"
    count: int = 1
    filters: tuple[FilterSpec, ...] = ()
    chooser: str = "controller"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "obj_type": self.obj_type,
            "scope": self.scope,
            "zone": self.zone,
            "location": self.location,
            "count": self.count,
            "chooser": self.chooser,
        }
        if self.filters:
            d["filters"] = [f.to_dict() for f in self.filters]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> TargetSpec:
        filters = tuple(
            FilterSpec.from_dict(f) for f in d.get("filters", [])
        )
        return TargetSpec(
            obj_type=d.get("obj_type", "unit"),
            scope=d.get("scope", "any"),
            zone=d.get("zone", "any"),
            location=d.get("location", "any"),
            count=d.get("count", 1),
            filters=filters,
            chooser=d.get("chooser", "controller"),
        )


# Pre-built target specs for common patterns
SELF_TARGET = TargetSpec(obj_type="unit", scope="self", count=1)
FRIENDLY_UNIT = TargetSpec(obj_type="unit", scope="friendly")
ENEMY_UNIT = TargetSpec(obj_type="unit", scope="enemy")
ANY_UNIT = TargetSpec(obj_type="unit", scope="any")
ANY_UNIT_HERE = TargetSpec(obj_type="unit", scope="any", location="here")
FRIENDLY_UNIT_HERE = TargetSpec(obj_type="unit", scope="friendly", location="here")
ENEMY_UNIT_HERE = TargetSpec(obj_type="unit", scope="enemy", location="here")
SPELL_ON_CHAIN = TargetSpec(obj_type="spell", zone="chain")
FRIENDLY_GEAR = TargetSpec(obj_type="gear", scope="friendly")
ALL_UNITS = TargetSpec(obj_type="unit", scope="any", count=-1)
ALL_FRIENDLY_UNITS = TargetSpec(obj_type="unit", scope="friendly", count=-1)
ALL_ENEMY_UNITS = TargetSpec(obj_type="unit", scope="enemy", count=-1)


# ---------------------------------------------------------------------------
# FilterSpec — additional predicates on targets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterSpec:
    """A single filter predicate applied to potential targets.

    Attributes:
        field: What to check — "keyword", "might", "tag", "card_type",
               "domain", "name", "controller", "is_exhausted".
        op: Comparison — "eq", "neq", "lte", "gte", "has", "not_has".
        value: The value to compare against.
    """

    field: str
    op: str = "eq"
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> FilterSpec:
        return FilterSpec(
            field=d["field"],
            op=d.get("op", "eq"),
            value=d.get("value"),
        )


# Common filter constructors
def has_keyword(kw: str) -> FilterSpec:
    return FilterSpec(field="keyword", op="has", value=kw)

def might_lte(n: int) -> FilterSpec:
    return FilterSpec(field="might", op="lte", value=n)

def might_gte(n: int) -> FilterSpec:
    return FilterSpec(field="might", op="gte", value=n)

def has_tag(tag: str) -> FilterSpec:
    return FilterSpec(field="tag", op="has", value=tag)

def is_exhausted() -> FilterSpec:
    return FilterSpec(field="is_exhausted", op="eq", value=True)

def is_ready() -> FilterSpec:
    return FilterSpec(field="is_exhausted", op="eq", value=False)


# ---------------------------------------------------------------------------
# ConditionSpec — for conditional nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionSpec:
    """A condition to evaluate for conditional effect nodes.

    Attributes:
        cond_type: What kind of condition — "legion", "mighty",
                   "xp_gte", "has_keyword", "unit_count_gte",
                   "controls_battlefield", "has_buff", "is_attacker",
                   "is_defender", "card_played_this_turn", "hand_count_lte".
        params: Condition-specific parameters.
    """

    cond_type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"cond_type": self.cond_type}
        if self.params:
            d["params"] = self.params
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ConditionSpec:
        return ConditionSpec(
            cond_type=d["cond_type"],
            params=d.get("params", {}),
        )


# Common condition constructors
def legion_condition() -> ConditionSpec:
    return ConditionSpec(cond_type="legion")

def mighty_condition() -> ConditionSpec:
    return ConditionSpec(cond_type="mighty")

def xp_gte_condition(n: int) -> ConditionSpec:
    return ConditionSpec(cond_type="xp_gte", params={"threshold": n})


# ---------------------------------------------------------------------------
# DestinationSpec — for move effects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DestinationSpec:
    """Where to move a target.

    Attributes:
        zone: Target zone — "base", "battlefield", "hand", "trash", "deck_top",
              "deck_bottom", "banishment".
        location: Specific location — "owner" (owner's base), "here"
                  (source's battlefield), or a specific id.
    """

    zone: str = "base"
    location: str = "owner"

    def to_dict(self) -> dict[str, Any]:
        return {"zone": self.zone, "location": self.location}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DestinationSpec:
        return DestinationSpec(
            zone=d.get("zone", "base"),
            location=d.get("location", "owner"),
        )


# ---------------------------------------------------------------------------
# Node constructors — convenience functions for building IR trees
# ---------------------------------------------------------------------------

def deal_damage(amount: int, target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": DEAL_DAMAGE, "amount": amount}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def draw_cards(count: int = 1, player: str = "controller") -> dict:
    return {"type": DRAW_CARDS, "count": count, "player": player}

def give_might(amount: int, target: TargetSpec | dict | None = None,
               duration: str = "turn") -> dict:
    node: dict[str, Any] = {"type": GIVE_MIGHT, "amount": amount, "duration": duration}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def buff(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": BUFF}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def stun(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": STUN}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def heal(amount: int | str = "all", target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": HEAL, "amount": amount}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def kill(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": KILL}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def move(target: TargetSpec | dict | None = None,
         destination: DestinationSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": MOVE}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    if destination is not None:
        node["destination"] = destination.to_dict() if isinstance(destination, DestinationSpec) else destination
    return node

def ready(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": READY}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def exhaust_target(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": EXHAUST}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def discard(count: int = 1, player: str = "controller") -> dict:
    return {"type": DISCARD, "count": count, "player": player}

def banish(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": BANISH}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def counter(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": COUNTER}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def return_to_hand(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": RETURN_TO_HAND}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def return_to_deck(target: TargetSpec | dict | None = None,
                   position: str = "bottom") -> dict:
    node: dict[str, Any] = {"type": RETURN_TO_DECK, "position": position}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def recycle(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": RECYCLE}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def play_token(name: str = "Recruit", might: int = 1,
               keywords: list[dict] | None = None,
               temporary: bool = False,
               ready_on_enter: bool = False) -> dict:
    node: dict[str, Any] = {
        "type": PLAY_TOKEN,
        "name": name,
        "might": might,
        "temporary": temporary,
        "ready_on_enter": ready_on_enter,
    }
    if keywords:
        node["keywords"] = keywords
    return node

def add_energy(amount: int = 1, player: str = "controller") -> dict:
    return {"type": ADD_ENERGY, "amount": amount, "player": player}

def add_power(domain: str, amount: int = 1, player: str = "controller") -> dict:
    return {"type": ADD_POWER, "domain": domain, "amount": amount, "player": player}

def channel_rune(count: int = 1) -> dict:
    return {"type": CHANNEL_RUNE, "count": count}

def attach(gear: TargetSpec | dict | None = None,
           unit: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": ATTACH}
    if gear is not None:
        node["gear"] = gear.to_dict() if isinstance(gear, TargetSpec) else gear
    if unit is not None:
        node["unit"] = unit.to_dict() if isinstance(unit, TargetSpec) else unit
    return node

def detach(target: TargetSpec | dict | None = None) -> dict:
    node: dict[str, Any] = {"type": DETACH}
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node

def score_points(amount: int = 1, player: str = "controller") -> dict:
    return {"type": SCORE_POINTS, "amount": amount, "player": player}

def gain_xp(amount: int = 1) -> dict:
    return {"type": GAIN_XP, "amount": amount}

def spend_xp(amount: int) -> dict:
    return {"type": SPEND_XP, "amount": amount}

def look_at_top(count: int = 1, may_recycle: bool = True) -> dict:
    return {"type": LOOK_AT_TOP, "count": count, "may_recycle": may_recycle}


# Composition constructors

def sequence(steps: list[dict]) -> dict:
    return {"type": SEQUENCE, "steps": steps}

def conditional(condition: ConditionSpec | dict,
                then: dict,
                else_: dict | None = None) -> dict:
    cond = condition.to_dict() if isinstance(condition, ConditionSpec) else condition
    node: dict[str, Any] = {"type": CONDITIONAL, "condition": cond, "then": then}
    if else_ is not None:
        node["else"] = else_
    return node

def for_each(targets: TargetSpec | dict, effect: dict) -> dict:
    t = targets.to_dict() if isinstance(targets, TargetSpec) else targets
    return {"type": FOR_EACH, "targets": t, "effect": effect}

def choose_one(options: list[dict]) -> dict:
    return {"type": CHOOSE_ONE, "options": options}

def optional(effect: dict) -> dict:
    return {"type": OPTIONAL, "effect": effect}

def repeat_effect(effect: dict) -> dict:
    return {"type": REPEAT_EFFECT, "effect": effect}
