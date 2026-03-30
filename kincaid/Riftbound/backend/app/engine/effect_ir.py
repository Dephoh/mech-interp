"""Effect Intermediate Representation (IR) for composable card abilities.

Effect IR nodes are plain dicts with a "type" key, keeping them JSON-serializable
and easy to compose in the card pipeline. TargetSpec and ConditionSpec are
dataclasses providing structure for targeting and conditions.

The resolver in effect_resolver.py walks IR trees and calls primitives in
effect_primitives.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Effect Layer classification (rules 454)
# ---------------------------------------------------------------------------

class EffectLayer(str, Enum):
    """The three layers in which continuous effects are applied (rule 454).

    Applied in order:
      1. TRAIT_ALTERING  — grants/removes/replaces inherent traits (name, type,
                           tags, controller, cost, domain, Might *assignment*)
      2. ABILITY_ALTERING — grants/removes/replaces abilities, keywords, rules text
      3. ARITHMETIC       — numeric increases then decreases to Might, cost, etc.
    """
    TRAIT_ALTERING = "trait_altering"
    ABILITY_ALTERING = "ability_altering"
    ARITHMETIC = "arithmetic"


def classify_modifier_layer(node: dict[str, Any]) -> EffectLayer:
    """Determine which layer a modifier IR node belongs to (rule 454).

    - stat in {"name", "type", "tag", "controller", "cost", "domain",
      "might_set"} -> Trait-Altering (454.1)
    - stat in {"keyword", "ability", "rules_text"} -> Ability-Altering (454.2)
    - stat in {"might", "energy_cost", "power_cost"} -> Arithmetic (454.3)
    """
    stat = node.get("stat", "might")
    if stat in ("name", "type", "tag", "controller", "cost", "domain", "might_set"):
        return EffectLayer.TRAIT_ALTERING
    if stat in ("keyword", "ability", "rules_text"):
        return EffectLayer.ABILITY_ALTERING
    # Default: arithmetic (might, energy_cost, power_cost, shield, assault, etc.)
    return EffectLayer.ARITHMETIC


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
SPEND_ENERGY = "spend_energy"
SPEND_POWER = "spend_power"
FIGHT = "fight"
CHALLENGE = "challenge"
GAIN_CONTROL = "gain_control"
SET_STAT = "set_stat"
PLAY = "play"
PLAY_CARD = "play_card"
REVEAL_UNTIL = "reveal_until"
REVERT_BATTLEFIELD = "revert_battlefield"
TRIGGER_KEYWORD = "trigger_keyword"
GRANT_ACTIVATED_ABILITY = "grant_activated_ability"
SHARE_ABILITIES = "share_abilities"

# Composition node type constants (branch nodes)
SEQUENCE = "sequence"
CONDITIONAL = "conditional"
FOR_EACH = "for_each"
CHOOSE_ONE = "choose_one"
OPTIONAL = "optional"
REPEAT_EFFECT = "repeat_effect"
ARRAY = "array"  # alias for choose_one — multiple option branches

# Rule-breaker node type constants (stateful / interactive nodes)
MODIFIER = "modifier"
REPLACEMENT = "replacement"
PLAYER_CHOICE = "player_choice"
DELAYED_TRIGGER = "delayed_trigger"  # rules 382-385
TRIGGERED_EFFECT = "triggered_effect"  # register a triggered ability

# All valid node types
PRIMITIVE_TYPES = frozenset({
    DEAL_DAMAGE, DRAW_CARDS, GIVE_MIGHT, BUFF, STUN, HEAL, KILL, MOVE,
    READY, EXHAUST, DISCARD, BANISH, COUNTER, RETURN_TO_HAND, RETURN_TO_DECK,
    RECYCLE, PLAY_TOKEN, ADD_ENERGY, ADD_POWER, CHANNEL_RUNE, ATTACH, DETACH,
    SCORE_POINTS, GAIN_XP, SPEND_XP, LOOK_AT_TOP, GIVE_KEYWORD, RESTRICT,
    SPEND_ENERGY, SPEND_POWER, FIGHT, CHALLENGE, GAIN_CONTROL, SET_STAT,
    PLAY, PLAY_CARD, REVEAL_UNTIL, REVERT_BATTLEFIELD, TRIGGER_KEYWORD,
    GRANT_ACTIVATED_ABILITY, SHARE_ABILITIES,
})

COMPOSITION_TYPES = frozenset({
    SEQUENCE, CONDITIONAL, FOR_EACH, CHOOSE_ONE, OPTIONAL, REPEAT_EFFECT,
    ARRAY,
})

RULE_BREAKER_TYPES = frozenset({
    MODIFIER, REPLACEMENT, PLAYER_CHOICE, DELAYED_TRIGGER, TRIGGERED_EFFECT,
})

# Descriptor types: used as sub-nodes (e.g. trigger descriptors inside
# triggered_effect) but not directly executable as standalone effects.
EVENT = "event"
DESCRIPTOR_TYPES = frozenset({EVENT})

ALL_NODE_TYPES = PRIMITIVE_TYPES | COMPOSITION_TYPES | RULE_BREAKER_TYPES | DESCRIPTOR_TYPES


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


# ---------------------------------------------------------------------------
# Rule-breaker node constructors — stateful / interactive IR nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModifierSpec:
    """A continuous stat adjustment applied to matching targets.

    While the source permanent is on the board, all targets matching the
    filter have their stats adjusted. The engine recalculates derived stats
    dynamically — base card data in card_db is never mutated.

    Attributes:
        stat: Which stat to modify — "might", "shield", "assault".
        amount: Signed integer adjustment (e.g., +1, -2).
        target: Who receives the modifier.
        duration: "continuous" (while source is on board), "turn", or "permanent".
        exclude_self: If True, the source permanent is not affected.
        layer: Which effect layer this modifier belongs to (rule 454).
    """

    stat: str = "might"
    amount: int = 0
    target: TargetSpec = field(default_factory=lambda: TargetSpec(scope="friendly"))
    duration: str = "continuous"
    exclude_self: bool = False
    layer: str = ""  # auto-classified if empty

    def to_dict(self) -> dict[str, Any]:
        return {
            "stat": self.stat,
            "amount": self.amount,
            "target": self.target.to_dict() if isinstance(self.target, TargetSpec) else self.target,
            "duration": self.duration,
            "exclude_self": self.exclude_self,
            "layer": self.layer,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ModifierSpec:
        target = d.get("target", {})
        if isinstance(target, dict):
            target = TargetSpec.from_dict(target)
        return ModifierSpec(
            stat=d.get("stat", "might"),
            amount=d.get("amount", 0),
            target=target,
            duration=d.get("duration", "continuous"),
            exclude_self=d.get("exclude_self", False),
            layer=d.get("layer", ""),
        )


@dataclass(frozen=True)
class ReplacementSpec:
    """Specifies an event to intercept, a condition, and an alternative IR.

    When the watched event is about to occur and the condition is met,
    the original event is suppressed and the alternative_ir is executed
    instead.

    Attributes:
        watch_event: The GameEvent name to intercept (e.g., "deal_damage").
        condition: When to intercept — evaluated against event context.
        alternative_ir: The IR subtree to execute instead of the original.
        target: Which objects this replacement applies to.
    """

    watch_event: str = ""
    condition: ConditionSpec = field(default_factory=lambda: ConditionSpec(cond_type="always"))
    alternative_ir: dict = field(default_factory=dict)
    target: TargetSpec = field(default_factory=TargetSpec)

    def to_dict(self) -> dict[str, Any]:
        return {
            "watch_event": self.watch_event,
            "condition": self.condition.to_dict() if isinstance(self.condition, ConditionSpec) else self.condition,
            "alternative_ir": self.alternative_ir,
            "target": self.target.to_dict() if isinstance(self.target, TargetSpec) else self.target,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ReplacementSpec:
        cond = d.get("condition", {"cond_type": "always"})
        if isinstance(cond, dict):
            cond = ConditionSpec.from_dict(cond)
        target = d.get("target", {})
        if isinstance(target, dict):
            target = TargetSpec.from_dict(target)
        return ReplacementSpec(
            watch_event=d.get("watch_event", ""),
            condition=cond,
            alternative_ir=d.get("alternative_ir", {}),
            target=target,
        )


@dataclass(frozen=True)
class ChoiceOptionSpec:
    """One selectable option within a player_choice node.

    Attributes:
        label: Human-readable label for the frontend (e.g., "Deal 3 damage").
        effect: The IR subtree to execute if this option is selected.
    """

    label: str = ""
    effect: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "effect": self.effect}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ChoiceOptionSpec:
        return ChoiceOptionSpec(label=d.get("label", ""), effect=d.get("effect", {}))


def modifier(
    stat: str,
    amount: int,
    target: TargetSpec | dict | None = None,
    duration: str = "continuous",
    exclude_self: bool = False,
    layer: str = "",
) -> dict:
    """Create a ModifierNode IR dict.

    Represents a continuous aura/buff like "Your other units have +1 Might."
    The ``layer`` field is auto-classified from ``stat`` when empty
    (see :func:`classify_modifier_layer`).
    """
    node: dict[str, Any] = {
        "type": MODIFIER,
        "stat": stat,
        "amount": amount,
        "duration": duration,
        "exclude_self": exclude_self,
    }
    # Auto-classify layer when not explicitly provided
    if layer:
        node["layer"] = layer
    else:
        node["layer"] = classify_modifier_layer(node).value
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node


def replacement(
    watch_event: str,
    condition: ConditionSpec | dict | None = None,
    alternative_ir: dict | None = None,
    target: TargetSpec | dict | None = None,
) -> dict:
    """Create a ReplacementNode IR dict.

    Represents event interception: "If this unit would take damage, prevent it."
    """
    node: dict[str, Any] = {
        "type": REPLACEMENT,
        "watch_event": watch_event,
    }
    if condition is not None:
        node["condition"] = condition.to_dict() if isinstance(condition, ConditionSpec) else condition
    if alternative_ir is not None:
        node["alternative_ir"] = alternative_ir
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node


def player_choice(
    prompt: str,
    options: list[dict] | None = None,
    target: TargetSpec | dict | None = None,
    min_choices: int = 1,
    max_choices: int = 1,
) -> dict:
    """Create a PlayerChoice IR dict.

    Represents a point where the engine must halt and ask the player
    to pick an option or select a target before resuming.

    Args:
        prompt: Description shown to the player (e.g., "Choose a unit to destroy").
        options: List of ChoiceOptionSpec dicts (for modal choices).
        target: TargetSpec for target-selection choices (pick from matching).
        min_choices: Minimum number of selections required.
        max_choices: Maximum number of selections allowed.
    """
    node: dict[str, Any] = {
        "type": PLAYER_CHOICE,
        "prompt": prompt,
        "min_choices": min_choices,
        "max_choices": max_choices,
    }
    if options is not None:
        node["options"] = options
    if target is not None:
        node["target"] = target.to_dict() if isinstance(target, TargetSpec) else target
    return node


# ---------------------------------------------------------------------------
# Delayed trigger IR node (rules 382-385)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DelayedTriggerSpec:
    """Specifies a delayed ability that fires at a future time or event.

    Rules 382-385: Delayed abilities are created by other abilities/spells
    and execute when their condition and/or specified time occurs, regardless
    of whether the creating source is still on the board.

    Attributes:
        trigger_event: The GameEvent name that will fire this trigger
                       (e.g., "damage_dealt", "turn_end").
        condition: When to fire -- evaluated against event context.
        effect_ir: The IR subtree to execute when the trigger fires.
        duration: How long the delayed trigger persists --
                  "turn" (until end of turn), "once" (fires once then expires),
                  "permanent" (persists until removed).
        max_fires: Maximum times this delayed trigger can fire (0 = unlimited).
    """

    trigger_event: str = ""
    condition: ConditionSpec = field(default_factory=lambda: ConditionSpec(cond_type="always"))
    effect_ir: dict = field(default_factory=dict)
    duration: str = "turn"
    max_fires: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_event": self.trigger_event,
            "condition": self.condition.to_dict() if isinstance(self.condition, ConditionSpec) else self.condition,
            "effect_ir": self.effect_ir,
            "duration": self.duration,
            "max_fires": self.max_fires,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DelayedTriggerSpec:
        cond = d.get("condition", {"cond_type": "always"})
        if isinstance(cond, dict):
            cond = ConditionSpec.from_dict(cond)
        return DelayedTriggerSpec(
            trigger_event=d.get("trigger_event", ""),
            condition=cond,
            effect_ir=d.get("effect_ir", {}),
            duration=d.get("duration", "turn"),
            max_fires=d.get("max_fires", 1),
        )


def delayed_trigger(
    trigger_event: str,
    effect_ir: dict,
    condition: ConditionSpec | dict | None = None,
    duration: str = "turn",
    max_fires: int = 1,
) -> dict:
    """Create a DelayedTrigger IR dict (rules 382-385).

    Represents an ability that is created now but fires later when a
    specified event occurs. The delayed trigger persists independently
    of its source -- even if the creating card leaves play, the delayed
    trigger still resolves at the specified time.

    Example: "The next time this unit takes damage this turn, kill it."
      delayed_trigger("damage_dealt", kill(SELF_TARGET), duration="turn", max_fires=1)
    """
    node: dict[str, Any] = {
        "type": DELAYED_TRIGGER,
        "trigger_event": trigger_event,
        "effect_ir": effect_ir,
        "duration": duration,
        "max_fires": max_fires,
    }
    if condition is not None:
        node["condition"] = condition.to_dict() if isinstance(condition, ConditionSpec) else condition
    return node
