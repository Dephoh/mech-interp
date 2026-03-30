"""Vertex AI Translation Pipeline — translate dead abilities into Effect IR.

Reads card_definitions.json, finds abilities where effect_ir is null but text
exists, calls Gemini 2.5 Pro via the google-genai SDK with structured output
enforcement, and saves the translated IR back into the file.

Usage:
    python vertex_pipeline.py [--dry-run] [--limit N]

Authentication: uses Application Default Credentials (gcloud auth application-default login).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vertex_pipeline")

DATA_PATH = Path(__file__).parent / "data" / "card_definitions.json"


# ---------------------------------------------------------------------------
# Pydantic schema — flat (non-recursive) for google-genai SDK compatibility.
#
# The google-genai SDK inlines $ref without cycle detection, so any
# self-referential Pydantic model causes "maximum recursion depth exceeded".
# We work around this by typing nested effect fields as dict[str, Any].
# The system prompt + few-shot examples enforce valid nested structure.
# ---------------------------------------------------------------------------


class FilterSpecModel(BaseModel):
    field: str = Field(description="Field to check: keyword, might, tag, card_type, domain, name, is_exhausted")
    op: str = Field(default="eq", description="Comparison: eq, neq, lte, gte, has, not_has")
    value: Any = Field(default=None)


class TargetSpecModel(BaseModel):
    """Target specification for effect resolution."""

    obj_type: str = Field(
        default="unit",
        description="Object type: unit, gear, spell, rune, card, player, permanent",
    )
    scope: str = Field(
        default="any",
        description="Ownership: friendly, enemy, any, self",
    )
    zone: str = Field(
        default="any",
        description="Zone: base, battlefield, chain, hand, trash, any",
    )
    location: str = Field(
        default="any",
        description="Location: here (same battlefield), any, or specific id",
    )
    count: int | str = Field(
        default=1,
        description="Number of targets: positive int, -1 for all, or the string 'all'",
    )
    chooser: str = Field(
        default="controller",
        description="Who picks: controller, opponent, auto",
    )
    filters: list[FilterSpecModel] = Field(default_factory=list)
    exclude_self: bool = Field(
        default=False,
        description="If true, the source card is excluded from matching",
    )


class ConditionSpecModel(BaseModel):
    cond_type: str = Field(
        description="Condition type: legion, mighty, xp_gte, has_keyword, unit_count_gte, "
                    "controls_battlefield, has_buff, is_attacker, is_defender, "
                    "card_played_this_turn, hand_count_lte, always, never",
    )
    params: dict[str, Any] = Field(default_factory=dict)


class DestinationSpecModel(BaseModel):
    zone: str = Field(default="base", description="Target zone: base, battlefield, hand, trash, deck_top, deck_bottom, banishment")
    location: str = Field(default="owner", description="Location: owner, here, or specific id")


class EffectNodeModel(BaseModel):
    """A single Effect IR node.

    Nested effect fields (steps, then, else, effect, options, alternative_ir)
    are typed as dict/list[dict] to avoid recursive $ref that the google-genai
    SDK cannot handle. The system prompt enforces valid nesting.
    """

    type: str = Field(
        description="Node type. Primitives: deal_damage, draw_cards, give_might, buff, stun, "
                    "heal, kill, move, ready, exhaust, discard, banish, counter, return_to_hand, "
                    "return_to_deck, recycle, play_token, add_energy, add_power, channel_rune, "
                    "attach, detach, score_points, gain_xp, spend_xp, look_at_top, give_keyword, "
                    "restrict. Composition: sequence, conditional, for_each, choose_one, optional, "
                    "repeat_effect. Rule-breakers: modifier, replacement, player_choice.",
    )

    # -- Primitive fields --
    amount: int | str | None = Field(default=None, description="Numeric amount or 'all'")
    target: TargetSpecModel | None = Field(default=None, description="Target specification")
    count: int | None = Field(default=None, description="Count (e.g., draw count)")
    player: str | None = Field(default=None, description="Player: controller, opponent, owner")
    duration: str | None = Field(default=None, description="Duration: turn, permanent, continuous")
    keyword: str | None = Field(default=None, description="Keyword name for give_keyword")
    destination: DestinationSpecModel | None = Field(default=None, description="For move effects")
    position: str | None = Field(default=None, description="top or bottom for return_to_deck")
    name: str | None = Field(default=None, description="Token name for play_token")
    might: int | None = Field(default=None, description="Token might for play_token")
    temporary: bool | None = Field(default=None, description="Token temporary flag")
    ready_on_enter: bool | None = Field(default=None, description="Token ready on enter")
    domain: str | None = Field(default=None, description="Domain for add_power")
    may_recycle: bool | None = Field(default=None, description="For look_at_top")
    keywords: list[dict[str, Any]] | None = Field(default=None, description="Keywords for play_token")
    restriction: str | None = Field(default=None, description="Restriction type")
    scope: str | None = Field(default=None, description="Restriction scope")
    gear: TargetSpecModel | None = Field(default=None, description="For attach")
    unit: TargetSpecModel | None = Field(default=None, description="For attach")

    # -- Composition fields (flat: nested nodes are plain dicts) --
    steps: list[dict[str, Any]] | None = Field(default=None, description="Steps for sequence node — each element is an Effect IR node dict")
    condition: ConditionSpecModel | None = Field(default=None, description="Condition for conditional node")
    then: dict[str, Any] | None = Field(default=None, description="Then-branch Effect IR node for conditional")
    else_: dict[str, Any] | None = Field(default=None, alias="else", description="Else-branch Effect IR node for conditional")
    targets: TargetSpecModel | None = Field(default=None, description="Target spec for for_each node")
    effect: dict[str, Any] | None = Field(default=None, description="Inner Effect IR node for for_each, optional, repeat_effect")
    options: list[dict[str, Any]] | None = Field(default=None, description="Option Effect IR nodes for choose_one")

    # -- Modifier fields --
    stat: str | None = Field(default=None, description="Stat to modify: might, shield, assault")
    exclude_self: bool | None = Field(default=None, description="Exclude source from modifier")

    # -- Replacement fields --
    watch_event: str | None = Field(default=None, description="Primitive type to intercept (e.g., deal_damage)")
    alternative_ir: dict[str, Any] | None = Field(default=None, description="Alternative Effect IR node to execute instead")

    # -- Player choice fields --
    prompt: str | None = Field(default=None, description="Prompt shown to the player")
    min_choices: int | None = Field(default=None)
    max_choices: int | None = Field(default=None)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert at translating card game ability text into structured Effect IR (Intermediate Representation) JSON for the Riftbound digital card game engine.

## RULES
- Output ONLY valid Effect IR JSON matching the provided schema.
- For multi-step abilities, use {"type": "sequence", "steps": [...]}.
- For conditional abilities ("If X, then Y"), use {"type": "conditional", "condition": {...}, "then": {...}}.
- For "each" / "all" targeting, use {"type": "for_each", "targets": {...}, "effect": {...}}.
- For "you may" optional effects, use {"type": "optional", "effect": {...}}.
- For "choose one" effects, use {"type": "choose_one", "options": [...]}.
- For continuous auras ("Your units have +1 Might"), use {"type": "modifier", "stat": "might", "amount": 1, "target": {...}}.
- For replacement effects ("If this would..., instead..."), use {"type": "replacement", "watch_event": "...", "alternative_ir": {...}}.
- For player choice during resolution, use {"type": "player_choice", "prompt": "...", "options": [...]}.
- "count": -1 means "all matching targets".
- Target scopes: "friendly" = your stuff, "enemy" = opponent's, "self" = this card, "any" = either player.
- SCOPE RULE — choose the scope that matches the text EXACTLY:
  · "a friendly unit" / "your unit" / "another unit you control" / "a unit you control" → scope: "friendly"
  · "an enemy unit" / "opponent's unit" → scope: "enemy"
  · "a unit" / "a unit at a battlefield" (NO ownership qualifier) → scope: "any"
  · "me" / "this card" / "I" → scope: "self"
  If the text does NOT say "friendly", "your", or "you control", the scope MUST NOT be "friendly". Default to "any" when no ownership is stated.
- Keywords: accelerate, action, ambush, assault, backline, deathknell, deflect, equip, ganking, hidden, hunt, legion, level, mighty, predict, quick_draw, reaction, repeat, shield, tank, temporary, unique, vision, weaponmaster.
- Domains: fury, calm, mind, body, chaos, order, any.
- [S] in card text means the Might stat (Shield icon). +1 [S] = +1 Might this turn.
- When text says "here", set location to "here" in target spec.
- Dynamic amounts: when an amount depends on a unit's Might, use the string "source_might" (the card with the ability) or "target_might" (the target being affected). Example: "deal damage equal to my Might" → amount: "source_might".

## NODE TYPE REFERENCE
Valid types and key fields (? = optional). Nested effect fields MUST be JSON objects with a "type" key, NEVER strings.

Primitives:
  deal_damage(amount, target?) · draw_cards(count?) · give_might(amount, target?, duration?)
  buff(target?) · stun(target?) · heal(amount?, target?) · kill(target?)
  move(target?, destination:{zone,location?}) · ready(target?) · exhaust(target?)
  discard(count?, player?) · banish(target?) · counter(target?)
  return_to_hand(target?) · return_to_deck(target?, position?) · recycle(target?)
  play_token(name?, might?, temporary?, ready_on_enter?, count?, keywords?)
  add_energy(amount?, player?) · add_power(domain, amount?, player?) · channel_rune(count?)
  attach(gear:target, unit:target) · detach(target?) · score_points(amount?, player?)
  gain_xp(amount?) · spend_xp(amount?) · look_at_top(count?, may_recycle?)
  give_keyword(keyword, target?, duration?) · restrict(restriction?, scope?, duration?)
  grant_activated_ability(target?, ability:{cost?, effect:effect_node, params?})
  share_abilities(target?, sources:[target_spec], ability_filter?:{field, op?, value?})

Composition (nested fields MUST be effect node objects):
  sequence     → steps: [effect_node, ...]
  conditional  → condition: {cond_type, params?}, then: effect_node, else?: effect_node
  for_each     → targets: target_spec, effect: effect_node
  choose_one   → options: [effect_node, ...]
  optional     → effect: effect_node
  repeat_effect → effect: effect_node

Rule-breakers:
  modifier      → stat("might"|"shield"|"assault"), amount, target?, duration?, exclude_self?
  replacement   → watch_event(str), alternative_ir?: effect_node, condition?, target?
  player_choice → prompt(str), options?: [{label, effect: effect_node}], target?, min_choices?, max_choices?

target_spec: {obj_type?, scope?, zone?, location?, count?, chooser?, filters?: [{field, op?, value?}], exclude_self?}
condition_spec: {cond_type, params?}

## CRITICAL CONSTRAINTS
1. ONLY use the node types listed above. NEVER invent types like "array", "challenge", or "trigger_effect".
2. Every nested effect position (steps items, then, else, effect, options items, alternative_ir) MUST be a valid effect node object with a "type" field. NEVER use plain strings.
3. Composition nodes MUST include their required nested fields (e.g., sequence needs non-empty "steps").
4. Omit fields that are not relevant — do not include null or empty values.

## FEW-SHOT EXAMPLES

Text: "Deal 3 damage to an enemy unit."
JSON: {"type": "deal_damage", "amount": 3, "target": {"obj_type": "unit", "scope": "enemy", "count": 1}}

Text: "Draw 2 cards, then deal 1 damage to all enemy units."
JSON: {"type": "sequence", "steps": [{"type": "draw_cards", "count": 2}, {"type": "deal_damage", "amount": 1, "target": {"obj_type": "unit", "scope": "enemy", "count": -1}}]}

Text: "Your other units have +1 Might."
JSON: {"type": "modifier", "stat": "might", "amount": 1, "target": {"obj_type": "unit", "scope": "friendly", "exclude_self": true}}

Text: "When you play me, draw a card."
JSON: {"type": "draw_cards", "count": 1}

Text: "Kill an enemy unit with 2 or less Might."
JSON: {"type": "kill", "target": {"obj_type": "unit", "scope": "enemy", "count": 1, "filters": [{"field": "might", "op": "lte", "value": 2}]}}

Text: "You may give a friendly unit +2 Might this turn."
JSON: {"type": "optional", "effect": {"type": "give_might", "amount": 2, "target": {"obj_type": "unit", "scope": "friendly", "count": 1}, "duration": "turn"}}

Text: "If you have 3 or more XP, deal 4 damage to an enemy unit."
JSON: {"type": "conditional", "condition": {"cond_type": "xp_gte", "params": {"value": 3}}, "then": {"type": "deal_damage", "amount": 4, "target": {"obj_type": "unit", "scope": "enemy", "count": 1}}}

Text: "Each player discards their hand, then draws 4."
JSON: {"type": "for_each", "targets": {"obj_type": "player", "scope": "any", "count": -1}, "effect": {"type": "sequence", "steps": [{"type": "discard", "count": -1, "player": "target"}, {"type": "draw_cards", "count": 4, "player": "target"}]}}

Text: "Play a 1/1 temporary Poro token."
JSON: {"type": "play_token", "name": "Poro", "might": 1, "temporary": true, "count": 1}

Text: "Choose one: Deal 2 damage to a unit, or draw 2 cards."
JSON: {"type": "choose_one", "options": [{"type": "deal_damage", "amount": 2, "target": {"obj_type": "unit", "scope": "any", "count": 1}}, {"type": "draw_cards", "count": 2}]}

Text: "Deal 3 to a unit at a battlefield."
JSON: {"type": "deal_damage", "amount": 3, "target": {"obj_type": "unit", "scope": "any", "count": 1, "location": "any"}}

Text: "Kill a unit at a battlefield."
JSON: {"type": "kill", "target": {"obj_type": "unit", "scope": "any", "count": 1, "location": "any"}}

Text: "Stun a unit."
JSON: {"type": "stun", "target": {"obj_type": "unit", "scope": "any", "count": 1}}
"""


def build_user_prompt(card_name: str, ability_text: str, card_context: str) -> str:
    return (
        f"Card: {card_name}\n"
        f"Context: {card_context}\n"
        f"Ability text: {ability_text}\n\n"
        f"Translate this ability into Effect IR JSON."
    )


def get_card_context(card: dict) -> str:
    """Build a short context string about the card for the LLM."""
    parts = [card.get("card_type", "unknown")]
    if card.get("domains"):
        parts.append(f"domains={card['domains']}")
    if card.get("base_might"):
        parts.append(f"might={card['base_might']}")
    if card.get("cost_energy"):
        parts.append(f"cost={card['cost_energy']}")
    kws = [kw.get("keyword", "") for kw in card.get("keywords", [])]
    if kws:
        parts.append(f"keywords={kws}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# API schema — discriminated-union sent via response_json_schema so Gemini
# gets real structural guidance for nested effects (recursive $refs OK here
# because all recursive fields are non-required).
# ---------------------------------------------------------------------------

def build_api_schema() -> dict[str, Any]:
    """Build the batch-mode JSON schema for response_json_schema."""
    return {
        "$defs": {
            "target_spec": {
                "type": "object",
                "properties": {
                    "obj_type": {"type": "string"},
                    "scope": {"type": "string"},
                    "zone": {"type": "string"},
                    "location": {"type": "string"},
                    "count": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                    "chooser": {"type": "string"},
                    "filters": {"type": "array", "items": {"$ref": "#/$defs/filter_spec"}},
                    "exclude_self": {"type": "boolean"},
                },
            },
            "filter_spec": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "op": {"type": "string"},
                    "value": {},
                },
                "required": ["field"],
            },
            "condition_spec": {
                "type": "object",
                "properties": {
                    "cond_type": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["cond_type"],
            },
            "destination_spec": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string"},
                    "location": {"type": "string"},
                },
            },
            "effect_node": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    # Primitive fields
                    "amount": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                    "target": {"$ref": "#/$defs/target_spec"},
                    "count": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                    "player": {"type": "string"},
                    "duration": {"type": "string"},
                    "keyword": {"type": "string"},
                    "destination": {"$ref": "#/$defs/destination_spec"},
                    "position": {"type": "string"},
                    "name": {"type": "string"},
                    "might": {"type": "integer"},
                    "temporary": {"type": "boolean"},
                    "ready_on_enter": {"type": "boolean"},
                    "domain": {"type": "string"},
                    "may_recycle": {"type": "boolean"},
                    "keywords": {"type": "array", "items": {"type": "object"}},
                    "restriction": {"type": "string"},
                    "scope": {"type": "string"},
                    "gear": {"$ref": "#/$defs/target_spec"},
                    "unit": {"$ref": "#/$defs/target_spec"},
                    # Composition — recursive refs, all non-required
                    "steps": {"type": "array", "items": {"$ref": "#/$defs/effect_node"}},
                    "condition": {"$ref": "#/$defs/condition_spec"},
                    "then": {"$ref": "#/$defs/effect_node"},
                    "else": {"$ref": "#/$defs/effect_node"},
                    "targets": {"$ref": "#/$defs/target_spec"},
                    "effect": {"$ref": "#/$defs/effect_node"},
                    "options": {"type": "array", "items": {"$ref": "#/$defs/effect_node"}},
                    # Modifier / replacement / player-choice
                    "stat": {"type": "string"},
                    "exclude_self": {"type": "boolean"},
                    "watch_event": {"type": "string"},
                    "alternative_ir": {"$ref": "#/$defs/effect_node"},
                    "prompt": {"type": "string"},
                    "min_choices": {"type": "integer"},
                    "max_choices": {"type": "integer"},
                },
                "required": ["type"],
            },
            "batch_item": {
                "type": "object",
                "properties": {
                    "card_id": {"type": "string"},
                    "effect_ir": {"$ref": "#/$defs/effect_node"},
                },
                "required": ["card_id", "effect_ir"],
            },
        },
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {"$ref": "#/$defs/batch_item"},
            },
        },
        "required": ["results"],
    }


BATCH_SCHEMA = build_api_schema()


def build_batch_prompt(abilities: list[tuple[str, str, str, str]]) -> str:
    """Build a multi-card prompt.

    abilities: list of (ability_id, card_name, ability_text, card_context)
    """
    lines = [
        "Translate each card ability below into Effect IR JSON.",
        'Return {"results": [{"card_id": "...", "effect_ir": {...}}, ...]} — one entry per card, in order.',
        "",
    ]
    for i, (ab_id, card_name, ab_text, ctx) in enumerate(abilities, 1):
        lines.append(f"{i}. card_id: \"{ab_id}\"")
        lines.append(f"   Card: {card_name} ({ctx})")
        lines.append(f"   Ability: {ab_text}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-processing: clean the Pydantic model output into minimal IR dicts
# ---------------------------------------------------------------------------

def clean_ir_node(node: dict[str, Any]) -> dict[str, Any]:
    """Remove None/default fields from a parsed IR dict for compact storage."""
    cleaned: dict[str, Any] = {}
    for k, v in node.items():
        if v is None:
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        if isinstance(v, dict):
            v = clean_ir_node(v)
        elif isinstance(v, list):
            v = [clean_ir_node(item) if isinstance(item, dict) else item for item in v]
        # Convert count="all" to count=-1 for engine compatibility
        if k == "count" and v == "all":
            v = -1
        cleaned[k] = v
    return cleaned


# ---------------------------------------------------------------------------
# Validation — catch hallucinated types, strings-as-effects, missing fields
# ---------------------------------------------------------------------------

_NODE_REQUIRED: dict[str, set[str]] = {
    # Primitives
    "deal_damage": set(), "draw_cards": set(), "give_might": {"amount"},
    "buff": set(), "stun": set(), "heal": set(), "kill": set(),
    "move": set(), "ready": set(), "exhaust": set(), "discard": set(),
    "banish": set(), "counter": set(), "return_to_hand": set(),
    "return_to_deck": set(), "recycle": set(), "play_token": set(),
    "add_energy": set(), "add_power": {"domain"}, "channel_rune": set(),
    "attach": set(), "detach": set(), "score_points": set(),
    "gain_xp": set(), "spend_xp": set(), "look_at_top": set(),
    "give_keyword": {"keyword"}, "restrict": set(),
    "grant_activated_ability": set(), "share_abilities": set(),
    # Composition
    "sequence": {"steps"}, "conditional": {"condition", "then"},
    "for_each": {"targets", "effect"}, "choose_one": {"options"},
    "optional": {"effect"}, "repeat_effect": {"effect"},
    # Rule-breakers
    "modifier": {"stat", "amount"}, "replacement": {"watch_event"},
    "player_choice": {"prompt"},
}

_NESTED_SINGLE = {"then", "else", "effect", "alternative_ir"}


def validate_ir_node(node: Any, path: str = "root") -> list[str]:
    """Validate an IR node recursively. Returns list of error strings (empty = valid)."""
    errors: list[str] = []

    if not isinstance(node, dict):
        errors.append(f"{path}: expected object, got {type(node).__name__}")
        return errors

    ntype = node.get("type")
    if not ntype:
        errors.append(f"{path}: missing 'type' field")
        return errors

    if ntype not in _NODE_REQUIRED:
        errors.append(f"{path}: unknown type '{ntype}'. Valid types: {', '.join(sorted(_NODE_REQUIRED))}")
        return errors

    # Check required fields (also reject empty lists — e.g. sequence with steps=[])
    for field in _NODE_REQUIRED[ntype]:
        val = node.get(field)
        if val is None or (isinstance(val, list) and len(val) == 0):
            errors.append(f"{path}: type '{ntype}' requires non-empty field '{field}'")

    # Validate nested single-effect fields
    for field in _NESTED_SINGLE:
        val = node.get(field)
        if val is None:
            continue
        if isinstance(val, str):
            errors.append(f"{path}.{field}: must be an effect node object, got string: '{val[:60]}'")
        elif isinstance(val, dict):
            errors.extend(validate_ir_node(val, f"{path}.{field}"))

    # Validate steps (array of effect nodes)
    if "steps" in node and node["steps"] is not None:
        if not isinstance(node["steps"], list):
            errors.append(f"{path}.steps: expected array, got {type(node['steps']).__name__}")
        else:
            for i, item in enumerate(node["steps"]):
                if isinstance(item, str):
                    errors.append(f"{path}.steps[{i}]: must be an effect node object, got string")
                elif isinstance(item, dict):
                    errors.extend(validate_ir_node(item, f"{path}.steps[{i}]"))

    # Validate options — structure depends on parent type
    if "options" in node and node["options"] is not None and isinstance(node["options"], list):
        for i, item in enumerate(node["options"]):
            if ntype == "choose_one":
                # options are direct effect nodes
                if isinstance(item, str):
                    errors.append(f"{path}.options[{i}]: must be an effect node object, got string")
                elif isinstance(item, dict):
                    errors.extend(validate_ir_node(item, f"{path}.options[{i}]"))
            elif ntype == "player_choice":
                # options are {label, effect} wrappers
                if isinstance(item, dict) and "effect" in item and isinstance(item["effect"], dict):
                    errors.extend(validate_ir_node(item["effect"], f"{path}.options[{i}].effect"))

    return errors


# Scope words that indicate the target belongs to the caster
_FRIENDLY_INDICATORS = re.compile(
    r"\b(friendly|your|you control|another .* you control)\b", re.IGNORECASE,
)


def validate_scope_from_text(node: Any, ability_text: str, path: str = "root") -> list[str]:
    """Check IR target scopes against ability text. Returns error strings."""
    errors: list[str] = []
    if not isinstance(node, dict):
        return errors

    target = node.get("target")
    if isinstance(target, dict) and target.get("scope") == "friendly":
        # Extract just the clause relevant to this node's action
        ntype = node.get("type", "")
        if ntype in ("deal_damage", "kill", "stun", "banish", "give_might",
                      "heal", "ready", "exhaust", "buff", "move",
                      "return_to_hand", "return_to_deck", "recycle"):
            if not _FRIENDLY_INDICATORS.search(ability_text):
                errors.append(
                    f"{path}: scope is 'friendly' but ability text has no friendly/your qualifier — should be 'any'"
                )

    # Recurse into nested effects
    for field in ("then", "else", "effect", "alternative_ir"):
        val = node.get(field)
        if isinstance(val, dict):
            errors.extend(validate_scope_from_text(val, ability_text, f"{path}.{field}"))
    for field in ("steps", "options"):
        val = node.get(field)
        if isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    errors.extend(validate_scope_from_text(item, ability_text, f"{path}.{field}[{i}]"))

    return errors


# ---------------------------------------------------------------------------
# Translation with validation + retry
# ---------------------------------------------------------------------------

def translate_ability(
    client: genai.Client,
    card_name: str,
    ab_text: str,
    context: str,
    max_retries: int = 2,
) -> tuple[dict[str, Any] | None, str | None]:
    """Translate one ability to Effect IR. Returns (cleaned_ir, None) or (None, error_msg)."""
    prompt = build_user_prompt(card_name, ab_text, context)
    last_errors = ""

    for attempt in range(1 + max_retries):
        try:
            contents = prompt
            if attempt > 0:
                contents = (
                    prompt
                    + f"\n\nYour previous output was invalid:\n{last_errors}\n"
                    + "Fix these issues and output corrected JSON."
                )
                time.sleep(2)  # rate-limit between retries

            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            raw_text = response.text
            if not raw_text:
                return None, "empty response"

            parsed = json.loads(raw_text)
            errs = validate_ir_node(parsed)
            errs.extend(validate_scope_from_text(parsed, ab_text))

            if not errs:
                return clean_ir_node(parsed), None

            last_errors = "; ".join(errs)
            logger.warning("  Validation failed (attempt %d/%d): %s", attempt + 1, 1 + max_retries, last_errors)

        except Exception as e:
            return None, str(e)

    return None, f"validation failed after {1 + max_retries} attempts: {last_errors}"


# ---------------------------------------------------------------------------
# Batch translation (async)
# ---------------------------------------------------------------------------

async def translate_batch_async(
    client_aio,
    abilities: list[tuple[str, str, str, str]],
    semaphore: asyncio.Semaphore,
) -> list[tuple[str, dict[str, Any] | None, str | None]]:
    """Translate a batch of abilities in one API call.

    abilities: list of (ability_id, card_name, ability_text, card_context)
    Returns: list of (ability_id, cleaned_ir | None, error | None)
    """
    async with semaphore:
        prompt = build_batch_prompt(abilities)
        ab_ids = [a[0] for a in abilities]

        try:
            response = await client_aio.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            raw = response.text
            if not raw:
                return [(aid, None, "empty response") for aid in ab_ids]

            parsed = json.loads(raw)
            results_list = parsed.get("results", [])

            # Build lookup by card_id
            by_id: dict[str, dict] = {}
            for item in results_list:
                cid = item.get("card_id", "")
                by_id[cid] = item.get("effect_ir")

            # Build text lookup for scope validation
            text_by_id = {a[0]: a[2] for a in abilities}

            out: list[tuple[str, dict | None, str | None]] = []
            for aid in ab_ids:
                ir = by_id.get(aid)
                if ir is None:
                    out.append((aid, None, f"missing from batch response"))
                    continue
                errs = validate_ir_node(ir)
                errs.extend(validate_scope_from_text(ir, text_by_id.get(aid, "")))
                if errs:
                    out.append((aid, None, "; ".join(errs)))
                else:
                    out.append((aid, clean_ir_node(ir), None))
            return out

        except Exception as e:
            return [(aid, None, str(e)) for aid in ab_ids]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline_async(
    dry_run: bool = False,
    limit: int | None = None,
    revalidate: bool = False,
    batch_size: int = 10,
    concurrency: int = 5,
) -> None:
    logger.info("Loading card definitions from %s", DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        cards: list[dict] = json.load(f)

    # Optionally null out existing invalid IR so it gets re-translated
    if revalidate:
        invalidated = 0
        for card in cards:
            for ab in card.get("abilities", []):
                ir = ab.get("effect_ir")
                ab_text = ab.get("text", "")
                if ir is not None and (validate_ir_node(ir) or validate_scope_from_text(ir, ab_text)):
                    logger.info("  Invalidating %s — %s", card.get("name"), ab_text[:50])
                    ab["effect_ir"] = None
                    invalidated += 1
        logger.info("Revalidation: invalidated %d entries with bad IR", invalidated)

    # Collect dead abilities: (card_idx, ab_idx, ability_id, card_name, text, context)
    dead: list[tuple[int, int, str, str, str, str]] = []
    for ci, card in enumerate(cards):
        for ai, ab in enumerate(card.get("abilities", [])):
            if ab.get("effect_ir") is None and ab.get("text", "").strip():
                dead.append((
                    ci, ai,
                    ab.get("ability_id", f"{card.get('card_id', 'unk')}_{ai}"),
                    card.get("name", "Unknown"),
                    ab["text"],
                    get_card_context(card),
                ))

    logger.info("Found %d dead abilities to translate", len(dead))
    if limit:
        dead = dead[:limit]
        logger.info("Limiting to %d abilities (--limit)", limit)

    if not dead:
        logger.info("Nothing to do.")
        return

    # Initialize Vertex AI client
    client = genai.Client(
        vertexai=True,
        project="riftbound-491700",
        location="us-central1",
    )

    # Build index: ability_id → (card_idx, ab_idx) for writing results back
    id_to_idx: dict[str, tuple[int, int]] = {d[2]: (d[0], d[1]) for d in dead}

    # Chunk into batches
    batches: list[list[tuple[str, str, str, str]]] = []
    for i in range(0, len(dead), batch_size):
        chunk = dead[i : i + batch_size]
        # Each batch item: (ability_id, card_name, text, context)
        batches.append([(d[2], d[3], d[4], d[5]) for d in chunk])

    logger.info(
        "Processing %d abilities in %d batches (size=%d, concurrency=%d)",
        len(dead), len(batches), batch_size, concurrency,
    )

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        translate_batch_async(client.aio, batch, semaphore)
        for batch in batches
    ]

    translated = 0
    errors = 0
    failed_singles: list[tuple[str, str, str, str]] = []  # retry individually

    batch_results = await asyncio.gather(*tasks)

    for batch_idx, results in enumerate(batch_results):
        for ability_id, ir, err in results:
            ci, ai = id_to_idx[ability_id]
            card_name = cards[ci].get("name", "Unknown")
            if err:
                logger.warning("  Batch fail %s: %s — queuing for single retry", card_name, err[:100])
                # Find original tuple for single-card retry
                for d in dead:
                    if d[2] == ability_id:
                        failed_singles.append((d[2], d[3], d[4], d[5]))
                        break
            else:
                if not dry_run:
                    cards[ci]["abilities"][ai]["effect_ir"] = ir
                translated += 1
                logger.info("  OK: %s — %s", card_name, json.dumps(ir, indent=None)[:100])

    # Retry failures individually (sync, with validation + retry)
    if failed_singles:
        logger.info("Retrying %d failed abilities individually...", len(failed_singles))
        for ab_id, card_name, ab_text, ctx in failed_singles:
            ci, ai = id_to_idx[ab_id]
            ir, err = translate_ability(client, card_name, ab_text, ctx)
            if err:
                logger.error("  FAILED (single) %s: %s", card_name, err[:100])
                errors += 1
            else:
                if not dry_run:
                    cards[ci]["abilities"][ai]["effect_ir"] = ir
                translated += 1
                logger.info("  OK (single): %s — %s", card_name, json.dumps(ir, indent=None)[:100])
            time.sleep(2)

    logger.info("Done. Translated: %d, Errors: %d", translated, errors)

    if not dry_run and translated > 0:
        logger.info("Saving card_definitions.json...")
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2, ensure_ascii=False)
        logger.info("Saved.")
    elif dry_run:
        logger.info("Dry run — no changes written.")


def run_pipeline(**kwargs) -> None:
    """Sync wrapper around the async pipeline."""
    asyncio.run(run_pipeline_async(**kwargs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate dead abilities to Effect IR via Vertex AI")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes to file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of abilities to translate")
    parser.add_argument("--revalidate", action="store_true", help="Re-translate existing entries that fail validation")
    parser.add_argument("--batch-size", type=int, default=10, help="Cards per API call (default 10)")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent API calls (default 5)")
    args = parser.parse_args()
    run_pipeline(
        dry_run=args.dry_run,
        limit=args.limit,
        revalidate=args.revalidate,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
