"""Card data pipeline: converts CMS cards.json to engine CardDefinitions.

Usage:
    python -m app.engine.card_pipeline <input_json> <output_json>

Reads the CMS card database (cards/cards.json) and produces a single JSON file
of engine-native CardDefinition dicts with structured Effect IR trees.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Symbol token mapping (richText -> clean text)
# ---------------------------------------------------------------------------

SYMBOL_MAP = {
    ":rb_exhaust:": "[T]",
    ":rb_might:": "[S]",
    ":rb_rune_rainbow:": "[A]",
    ":rb_rune_fury:": "[R]",
    ":rb_rune_calm:": "[G]",
    ":rb_rune_mind:": "[B]",
    ":rb_rune_body:": "[O]",
    ":rb_rune_chaos:": "[P]",
    ":rb_rune_order:": "[Y]",
}
# :rb_energy_N: -> [N]
ENERGY_RE = re.compile(r":rb_energy_(\d+):")

DOMAIN_SYMBOL_TO_ID = {
    "[R]": "fury", "[G]": "calm", "[B]": "mind",
    "[O]": "body", "[P]": "chaos", "[Y]": "order",
    "[A]": "any",
}

DOMAIN_ID_TO_SYMBOL = {v: k for k, v in DOMAIN_SYMBOL_TO_ID.items()}


# ---------------------------------------------------------------------------
# Keyword detection
# ---------------------------------------------------------------------------

# Keywords with optional numeric values: [Assault 2], [Shield 3], [Hunt 2], etc.
VALUED_KEYWORDS = {
    "Assault", "Shield", "Deflect", "Hunt", "Level", "Predict",
}

# Keywords without values
SIMPLE_KEYWORDS = {
    "Reaction", "Action", "Tank", "Ganking", "Hidden", "Accelerate",
    "Temporary", "Vision", "Deathknell", "Legion", "Mighty",
    "Weaponmaster", "Quick-Draw", "Ambush", "Backline", "Unique",
}

KEYWORD_NAME_TO_ID = {
    "Reaction": "reaction", "Action": "action", "Tank": "tank",
    "Ganking": "ganking", "Hidden": "hidden", "Accelerate": "accelerate",
    "Temporary": "temporary", "Vision": "vision", "Deathknell": "deathknell",
    "Legion": "legion", "Mighty": "mighty", "Weaponmaster": "weaponmaster",
    "Quick-Draw": "quick_draw", "Ambush": "ambush", "Backline": "backline",
    "Unique": "unique", "Assault": "assault", "Shield": "shield",
    "Deflect": "deflect", "Hunt": "hunt", "Level": "level",
    "Predict": "predict",
}

# Regex to find keywords in text: [Keyword] or [Keyword N]
KEYWORD_RE = re.compile(
    r"\[(" + "|".join(re.escape(k) for k in sorted(
        list(SIMPLE_KEYWORDS) + list(VALUED_KEYWORDS), key=len, reverse=True
    )) + r")(?:\s+(\d+))?\]"
)

# Reminder text in parentheses after keywords
REMINDER_RE = re.compile(r"\s*\([^)]*\)")

# Equip cost pattern: [Equip COST] or [Equip [N]] or [Equip [N][C]]
EQUIP_RE = re.compile(r"\[Equip\s+([^\]]+)\]")

# Repeat cost pattern: [Repeat COST]
REPEAT_RE = re.compile(r"\[Repeat\s+([^\]]+)\]")


# ---------------------------------------------------------------------------
# Trigger pattern detection
# ---------------------------------------------------------------------------

TRIGGER_PATTERNS = [
    (r"When you play me,?\s*", "on_play"),
    (r"When you play this,?\s*", "on_play"),
    (r"When I enter the board,?\s*", "on_play"),
    (r"When I conquer or hold,?\s*", "on_conquer_or_hold"),
    (r"When I conquer,?\s*", "on_conquer"),
    (r"When I hold,?\s*", "on_hold"),
    (r"When you conquer or hold,?\s*", "on_conquer_or_hold"),
    (r"When you conquer here,?\s*", "on_conquer"),
    (r"When you conquer,?\s*", "on_conquer"),
    (r"When you hold here,?\s*", "on_hold"),
    (r"When you hold,?\s*", "on_hold"),
    (r"When I attack or defend,?\s*", "on_attack_or_defend"),
    (r"When I attack,?\s*", "on_attack"),
    (r"When I defend,?\s*", "on_defend"),
    (r"When I die,?\s*", "on_death"),
    (r"When I move to a battlefield,?\s*", "on_move_to_bf"),
    (r"When I move,?\s*", "on_move"),
    (r"When a friendly unit dies,?\s*", "on_friendly_death"),
    (r"When an enemy unit dies,?\s*", "on_enemy_death"),
    (r"When one or more enemy units die,?\s*", "on_enemy_death"),
    (r"When you play a spell,?\s*", "on_spell_played"),
    (r"When you play a unit,?\s*", "on_unit_played"),
    (r"When a player plays a spell,?\s*", "on_any_spell_played"),
    (r"When you attach an Equipment to me,?\s*", "on_equip"),
    (r"When you or an ally (?:conquer|hold),?\s*", "on_conquer_or_hold"),
    (r"When you win a combat,?\s*", "on_combat_win"),
    (r"When you recycle a rune,?\s*", "on_recycle_rune"),
    (r"At the (?:start|beginning) of your turn,?\s*", "on_turn_start"),
    (r"At the end of your turn,?\s*", "on_turn_end"),
    (r"At the start of each turn,?\s*", "on_turn_start"),
]

TRIGGER_COMPILED = [(re.compile(pat, re.IGNORECASE), trigger) for pat, trigger in TRIGGER_PATTERNS]


# ---------------------------------------------------------------------------
# Effect text -> IR parsing
# ---------------------------------------------------------------------------

def parse_effect_text(text: str, card_name: str) -> dict | None:
    """Parse cleaned effect text into an Effect IR node.

    Returns None if the text cannot be parsed.
    """
    text = text.strip()
    if not text:
        return None

    # Try to parse as a sequence of sentences
    sentences = _split_sentences(text)
    if len(sentences) > 1:
        nodes = []
        for s in sentences:
            node = _parse_single_effect(s.strip())
            if node:
                nodes.append(node)
        if nodes:
            return nodes[0] if len(nodes) == 1 else {"type": "sequence", "steps": nodes}
        return None

    return _parse_single_effect(text)


def _split_sentences(text: str) -> list[str]:
    """Split text into effect sentences, respecting parentheses."""
    # Split on periods followed by space and uppercase, or newlines
    parts = re.split(r'(?<=[.!])\s+(?=[A-Z\[])|(?:\n)+', text)
    return [p.strip() for p in parts if p.strip()]


def _parse_single_effect(text: str) -> dict | None:
    """Parse a single effect sentence into an IR node."""
    text = text.strip().rstrip(".")

    # "you may X" -> optional wrapper
    may_match = re.match(r"[Yy]ou may\s+(.*)", text)
    if may_match:
        inner = _parse_single_effect(may_match.group(1))
        if inner:
            return {"type": "optional", "effect": inner}
        return None

    # "If CONDITION, EFFECT" -> conditional
    if_match = re.match(r"[Ii]f\s+(.+?),\s+(.*)", text)
    if if_match:
        cond_text = if_match.group(1)
        effect_text = if_match.group(2)
        condition = _parse_condition(cond_text)
        effect = _parse_single_effect(effect_text)
        if condition and effect:
            return {"type": "conditional", "condition": condition, "then": effect}

    # --- Concrete effect patterns ---

    # "Deal N to TARGET" / "Deal N damage to TARGET"
    deal_match = re.match(
        r"[Dd]eal\s+(\d+)\s+(?:damage\s+)?to\s+(.*)", text
    )
    if deal_match:
        amount = int(deal_match.group(1))
        target = _parse_target(deal_match.group(2))
        return {"type": "deal_damage", "amount": amount, "target": target}

    # "Draw N" / "Draw a card"
    draw_match = re.match(r"[Dd]raw\s+(\d+|a card)", text)
    if draw_match:
        count_str = draw_match.group(1)
        count = 1 if count_str == "a card" else int(count_str)
        return {"type": "draw_cards", "count": count, "player": "controller"}

    # "Give TARGET +N [S|M] this turn" / "give me +2 [S] this turn"
    give_match = re.match(
        r"[Gg]ive\s+(.+?)\s+\+?(\d+)\s+\[(?:S|M)\]\s*(?:this turn)?", text
    )
    if give_match:
        target = _parse_target(give_match.group(1))
        amount = int(give_match.group(2))
        return {"type": "give_might", "amount": amount, "target": target, "duration": "turn"}

    # "Give TARGET -N [S|M] this turn" (negative might)
    neg_give_match = re.match(
        r"[Gg]ive\s+(.+?)\s+-(\d+)\s+\[(?:S|M)\]\s*(?:this turn)?", text
    )
    if neg_give_match:
        target = _parse_target(neg_give_match.group(1))
        amount = -int(neg_give_match.group(2))
        return {"type": "give_might", "amount": amount, "target": target, "duration": "turn"}

    # "Buff TARGET" / "Buff me"
    buff_match = re.match(r"[Bb]uff\s+(.*)", text)
    if buff_match:
        target = _parse_target(buff_match.group(1))
        return {"type": "buff", "target": target}

    # "Stun TARGET"
    stun_match = re.match(r"[Ss]tun\s+(.*)", text)
    if stun_match:
        target = _parse_target(stun_match.group(1))
        return {"type": "stun", "target": target}

    # "Kill TARGET"
    kill_match = re.match(r"[Kk]ill\s+(.*)", text)
    if kill_match:
        target = _parse_target(kill_match.group(1))
        return {"type": "kill", "target": target}

    # "Heal TARGET" / "Heal N from TARGET"
    heal_match = re.match(r"[Hh]eal\s+(?:(\d+)\s+(?:damage\s+)?(?:from\s+)?)?(.+)", text)
    if heal_match:
        amount = int(heal_match.group(1)) if heal_match.group(1) else "all"
        target = _parse_target(heal_match.group(2))
        return {"type": "heal", "amount": amount, "target": target}

    # "Counter a spell"
    if re.match(r"[Cc]ounter\s+a\s+spell", text):
        return {"type": "counter", "target": {"obj_type": "spell", "zone": "chain"}}

    # "Return TARGET to TARGET_OWNER's hand"
    return_match = re.match(r"[Rr]eturn\s+(.+?)\s+to\s+(?:its|their)\s+owner'?s?\s+hand", text)
    if return_match:
        target = _parse_target(return_match.group(1))
        return {"type": "return_to_hand", "target": target}

    # "Return TARGET to hand" (simpler)
    return_match2 = re.match(r"[Rr]eturn\s+(.+?)\s+to\s+(?:your\s+)?hand", text)
    if return_match2:
        target = _parse_target(return_match2.group(1))
        return {"type": "return_to_hand", "target": target}

    # "Return all units with N [S|M] or less to their owners' hands"
    return_all_match = re.match(
        r"[Rr]eturn\s+all\s+units?\s+with\s+(\d+)\s+\[(?:S|M)\]\s+or\s+less\s+to\s+their\s+owners['']?\s*hands?",
        text,
    )
    if return_all_match:
        threshold = int(return_all_match.group(1))
        return {
            "type": "return_to_hand",
            "target": {
                "obj_type": "unit", "scope": "any", "count": -1,
                "filters": [{"field": "might", "op": "lte", "value": threshold}],
            },
        }

    # "Ready TARGET"
    ready_match = re.match(r"[Rr]eady\s+(.*)", text)
    if ready_match:
        target = _parse_target(ready_match.group(1))
        return {"type": "ready", "target": target}

    # "Exhaust TARGET"
    exhaust_match = re.match(r"[Ee]xhaust\s+(.*)", text)
    if exhaust_match:
        target = _parse_target(exhaust_match.group(1))
        return {"type": "exhaust", "target": target}

    # "Play a [ready] N [S|M] NAME unit token [with [Temporary]]"
    token_match = re.match(
        r"[Pp]lay\s+a\s+(?:(ready)\s+)?(\d+)\s+\[(?:S|M)\]\s+([\w\s]+?)\s*unit\s+token(?:\s+(?:with\s+)?\[Temporary\])?(?:\s+(?:in|to)\s+(?:your\s+base|here))?",
        text,
    )
    if token_match:
        ready = token_match.group(1) is not None
        might = int(token_match.group(2))
        name = token_match.group(3).strip()
        temporary = "[Temporary]" in text
        return {
            "type": "play_token", "name": name, "might": might,
            "temporary": temporary, "ready_on_enter": ready,
        }

    # "Gain N XP"
    xp_match = re.match(r"[Gg]ain\s+(\d+)\s+XP", text)
    if xp_match:
        return {"type": "gain_xp", "amount": int(xp_match.group(1))}

    # "Spend N XP"
    spend_xp_match = re.match(r"[Ss]pend\s+(\d+)\s+XP", text)
    if spend_xp_match:
        return {"type": "spend_xp", "amount": int(spend_xp_match.group(1))}

    # "Channel N rune[s] [exhausted]"
    channel_match = re.match(r"[Cc]hannel\s+(\d+)\s+runes?\s*(exhausted)?", text)
    if channel_match:
        node: dict[str, Any] = {"type": "channel_rune", "count": int(channel_match.group(1))}
        if channel_match.group(2):
            node["exhausted"] = True
        return node

    # "Move TARGET to TARGET_DEST"
    move_match = re.match(r"[Mm]ove\s+(.+?)\s+to\s+(?:its|your|their)\s+(base|hand)", text)
    if move_match:
        target = _parse_target(move_match.group(1))
        dest = move_match.group(2)
        return {
            "type": "move" if dest == "base" else "return_to_hand",
            "target": target,
            "destination": {"zone": dest, "location": "owner"},
        }

    # "you score N point[s]"
    score_match = re.match(r"[Yy]ou\s+score\s+(\d+)\s+points?", text)
    if score_match:
        return {"type": "score_points", "amount": int(score_match.group(1)), "player": "controller"}

    # "Discard N"
    discard_match = re.match(r"[Dd]iscard\s+(\d+)", text)
    if discard_match:
        return {"type": "discard", "count": int(discard_match.group(1)), "player": "controller"}

    # "Banish TARGET"
    banish_match = re.match(r"[Bb]anish\s+(.*)", text)
    if banish_match:
        target = _parse_target(banish_match.group(1))
        return {"type": "banish", "target": target}

    # "Recycle TARGET"
    recycle_match = re.match(r"[Rr]ecycle\s+(.*)", text)
    if recycle_match:
        target = _parse_target(recycle_match.group(1))
        return {"type": "recycle", "target": target}

    # "[Add] [N]" / "[Add] [N][C]" - resource gain shorthand
    add_match = re.match(r"\[Add\]\s*((?:\[\w+\][\s]*)+)", text)
    if add_match:
        tokens = re.findall(r"\[(\w+)\]", add_match.group(1))
        steps = []
        for tok in tokens:
            if tok.isdigit():
                steps.append({"type": "add_energy", "amount": int(tok)})
            elif tok in ("A", "R", "G", "B", "O", "P", "Y", "C"):
                domain = DOMAIN_SYMBOL_TO_ID.get(f"[{tok}]", "any")
                steps.append({"type": "add_power", "domain": domain, "amount": 1})
        if len(steps) == 1:
            return steps[0]
        if steps:
            return {"type": "sequence", "steps": steps}

    # "They discard N" / "choose a player. They discard N" / "TARGET discards N"
    they_discard = re.match(r"(?:[Tt]hey|[Tt]hat player|[Yy]our opponent)\s+discards?\s+(\d+)", text)
    if they_discard:
        return {"type": "discard", "count": int(they_discard.group(1)), "player": "opponent"}

    # "[Buff] TARGET" / "[Stun] TARGET" - bracketed keyword as verb
    bracket_buff = re.match(r"\[Buff\]\s+(.*)", text)
    if bracket_buff:
        target = _parse_target(bracket_buff.group(1))
        return {"type": "buff", "target": target}

    bracket_stun = re.match(r"\[Stun\]\s+(.*)", text)
    if bracket_stun:
        target = _parse_target(bracket_stun.group(1))
        return {"type": "stun", "target": target}

    # "Give TARGET [Keyword] [this turn]" - grant keyword
    give_kw_match = re.match(
        r"[Gg]ive\s+(.+?)\s+\[(\w+(?:-\w+)?)\](?:\s+this\s+turn)?",
        text,
    )
    if give_kw_match:
        # Check it's not a might grant (already handled above)
        kw_name = give_kw_match.group(2)
        if kw_name in ("S", "M"):
            pass  # Skip - handled by give_might patterns
        else:
            target = _parse_target(give_kw_match.group(1))
            kw_id = KEYWORD_NAME_TO_ID.get(kw_name, kw_name.lower())
            duration = "turn" if "this turn" in text.lower() else "permanent"
            return {"type": "give_keyword", "keyword": kw_id, "target": target, "duration": duration}

    # "Move an/a enemy/friendly unit [here/to this battlefield/to TARGET]"
    move_unit_match = re.match(
        r"[Mm]ove\s+(.+?)\s+(?:to\s+(?:here|this\s+battlefield|my\s+(?:location|battlefield)))",
        text,
    )
    if move_unit_match:
        target = _parse_target(move_unit_match.group(1))
        return {"type": "move", "target": target, "destination": {"zone": "battlefield", "location": "here"}}

    # "Move an/a enemy/friendly unit" (no destination = generic move)
    move_generic = re.match(r"[Mm]ove\s+(an?\s+(?:enemy|friendly)\s+(?:unit|gear))", text)
    if move_generic:
        target = _parse_target(move_generic.group(1))
        return {"type": "move", "target": target}

    # "Recall TARGET" / "Recall me" = return to base
    recall_match = re.match(r"[Rr]ecall\s+(.*)", text)
    if recall_match:
        target = _parse_target(recall_match.group(1))
        return {"type": "move", "target": target, "destination": {"zone": "base", "location": "owner"}}

    # "Return me to my owner's hand" / "Return me to hand"
    return_me = re.match(r"[Rr]eturn\s+me\s+to\s+(?:my\s+)?(?:owner'?s?\s+)?hand", text)
    if return_me:
        return {"type": "return_to_hand", "target": {"obj_type": "unit", "scope": "self", "count": 1}}

    # "Play N/word M [S] NAME unit tokens [here|to your base]"
    tokens_match = re.match(
        r"[Pp]lay\s+(?:(\w+)\s+)?(\d+)\s+\[(?:S|M)\]\s+([\w\s]+?)\s*unit\s+tokens?(?:\s+(?:with\s+)?\[Temporary\])?(?:\s+(?:in|to)\s+(your\s+base|here))?",
        text,
    )
    if tokens_match:
        count_word = tokens_match.group(1)
        might = int(tokens_match.group(2))
        name = tokens_match.group(3).strip()
        dest = tokens_match.group(4) or ""
        temporary = "[Temporary]" in text
        count = _word_to_number(count_word) if count_word else 1
        node: dict[str, Any] = {
            "type": "play_token", "name": name, "might": might,
            "temporary": temporary, "count": count,
        }
        if "here" in dest:
            node["destination"] = "here"
        return node

    # "Play two/three Gold gear tokens [exhausted]"
    gear_tokens_match = re.match(
        r"[Pp]lay\s+(\w+)\s+Gold\s+gear\s+tokens?(?:\s+exhausted)?",
        text,
    )
    if gear_tokens_match:
        count = _word_to_number(gear_tokens_match.group(1))
        return {"type": "play_token", "name": "Gold", "token_type": "gear", "might": 0,
                "count": count, "exhausted": "exhausted" in text.lower()}

    # "Ready N runes" / "Ready up to N runes"
    ready_runes = re.match(r"[Rr]eady\s+(?:up\s+to\s+)?(\d+)\s+runes?", text)
    if ready_runes:
        return {"type": "ready", "target": {"obj_type": "rune", "scope": "friendly", "count": int(ready_runes.group(1))}}

    # "Play a Gold gear token [exhausted]"
    gold_token = re.match(r"[Pp]lay\s+a\s+Gold\s+gear\s+token(?:\s+exhausted)?", text)
    if gold_token:
        return {"type": "play_token", "name": "Gold", "token_type": "gear", "might": 0, "exhausted": "exhausted" in text.lower()}

    # "N to TARGET" shorthand for "Deal N to TARGET"
    short_damage = re.match(r"(\d+)\s+to\s+(.*)", text)
    if short_damage:
        amount = int(short_damage.group(1))
        target = _parse_target(short_damage.group(2))
        return {"type": "deal_damage", "amount": amount, "target": target}

    # "Opponents can't play cards this turn"
    if re.match(r"[Oo]pponents?\s+can'?t\s+play\s+cards?\s+this\s+turn", text):
        return {"type": "restrict", "restriction": "cant_play", "scope": "opponents", "duration": "turn"}

    # "I enter ready" / "I enter the board ready"
    if re.match(r"I\s+enter(?:\s+the\s+board)?\s+ready", text, re.I):
        return {"type": "ready", "target": {"obj_type": "unit", "scope": "self", "count": 1}}

    # "I have +N [S]" - static might modifier
    static_might = re.match(r"I\s+have\s+\+(\d+)\s+\[S\]", text, re.I)
    if static_might:
        return {"type": "give_might", "amount": int(static_might.group(1)), "target": {"obj_type": "unit", "scope": "self", "count": 1}, "duration": "permanent"}

    return None  # Unparseable


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _word_to_number(word: str) -> int:
    """Convert a word number to int, e.g. 'three' -> 3."""
    if word is None:
        return 1
    word = word.strip().lower()
    if word.isdigit():
        return int(word)
    return _WORD_NUMBERS.get(word, 1)


def _parse_target(text: str) -> dict:
    """Parse a target description into a TargetSpec dict."""
    text = text.strip().rstrip(".,;")

    # "me" / "I" / "this" / "it" / "myself"
    if text.lower() in ("me", "i", "this", "this unit", "it", "myself", "this gear"):
        return {"obj_type": "unit", "scope": "self", "count": 1}

    # "each player" / "all players"
    if text.lower() in ("each player", "all players", "each opponent"):
        scope = "enemy" if "opponent" in text.lower() else "any"
        return {"obj_type": "player", "scope": scope, "count": -1}

    # "another friendly unit" / "another unit"
    another_match = re.match(r"another\s+(?:(friendly|enemy)\s+)?unit(?:\s+here)?", text, re.I)
    if another_match:
        scope = another_match.group(1).lower() if another_match.group(1) else "friendly"
        location = "here" if "here" in text.lower() else "any"
        return {"obj_type": "unit", "scope": scope, "count": 1, "location": location, "filters": [{"field": "not_self", "op": "eq", "value": True}]}

    # "your other units here"
    other_match = re.match(r"(?:your|my)\s+other\s+units?\s*(?:here)?", text, re.I)
    if other_match:
        location = "here" if "here" in text.lower() else "any"
        return {"obj_type": "unit", "scope": "friendly", "count": -1, "location": location, "filters": [{"field": "not_self", "op": "eq", "value": True}]}

    # "all units" / "all friendly units" / "all enemy units"
    all_match = re.match(r"all\s+(friendly\s+|enemy\s+)?units?(?:\s+here)?", text, re.I)
    if all_match:
        scope = "any"
        if all_match.group(1):
            scope = all_match.group(1).strip().lower()
        location = "here" if "here" in text.lower() else "any"
        return {"obj_type": "unit", "scope": scope, "count": -1, "location": location}

    # "a friendly unit [here|at a battlefield]"
    friendly_match = re.match(
        r"(?:a|an|one of (?:your|their))\s+(?:friendly\s+)?unit(?:\s+(?:you control\s+)?here)?",
        text, re.I,
    )
    if friendly_match:
        location = "here" if "here" in text.lower() else "any"
        return {"obj_type": "unit", "scope": "friendly", "count": 1, "location": location}

    # "an enemy unit [here|at a battlefield]"
    enemy_match = re.match(r"(?:a|an)\s+enemy\s+unit(?:\s+here)?", text, re.I)
    if enemy_match:
        location = "here" if "here" in text.lower() else "any"
        return {"obj_type": "unit", "scope": "enemy", "count": 1, "location": location}

    # "a unit [here]"
    unit_match = re.match(r"(?:a|an)\s+unit(?:\s+here)?", text, re.I)
    if unit_match:
        location = "here" if "here" in text.lower() else "any"
        return {"obj_type": "unit", "scope": "any", "count": 1, "location": location}

    # "a gear" / "a friendly gear" / "an Equipment"
    gear_match = re.match(r"(?:a|an)\s+(?:(friendly|enemy)\s+)?(?:gear|Equipment)", text, re.I)
    if gear_match:
        scope = gear_match.group(1).lower() if gear_match.group(1) else "any"
        return {"obj_type": "gear", "scope": scope, "count": 1}

    # "a spell"
    if re.match(r"(?:a|an)\s+spell", text, re.I):
        return {"obj_type": "spell", "zone": "chain", "count": 1}

    # "a rune" / "N runes"
    rune_match = re.match(r"(?:(\d+|a)\s+)?runes?", text, re.I)
    if rune_match:
        count_str = rune_match.group(1) or "1"
        count = 1 if count_str == "a" else int(count_str)
        return {"obj_type": "rune", "scope": "friendly", "count": count}

    # "one of their gear" / "one of your gear"
    their_gear = re.match(r"one\s+of\s+(?:their|your)\s+gear", text, re.I)
    if their_gear:
        scope = "friendly" if "your" in text.lower() else "any"
        return {"obj_type": "gear", "scope": scope, "count": 1}

    # "a unit here with less/more Might than me"
    might_filter = re.match(
        r"(?:a|an)\s+(?:(friendly|enemy)\s+)?unit\s+(?:here\s+)?with\s+(less|more)\s+Might\s+than\s+me",
        text, re.I,
    )
    if might_filter:
        scope = might_filter.group(1).lower() if might_filter.group(1) else "any"
        op = "lt" if might_filter.group(2).lower() == "less" else "gt"
        return {
            "obj_type": "unit", "scope": scope, "count": 1,
            "location": "here",
            "filters": [{"field": "might", "op": op, "value": "source_might"}],
        }

    # "an enemy unit attacking here"
    if re.match(r"an\s+enemy\s+unit\s+attacking\s+here", text, re.I):
        return {"obj_type": "unit", "scope": "enemy", "count": 1, "location": "here",
                "filters": [{"field": "combat_role", "op": "eq", "value": "attacker"}]}

    # Default: any unit
    return {"obj_type": "unit", "scope": "any", "count": 1}


def _parse_condition(text: str) -> dict | None:
    """Parse a condition clause into a ConditionSpec dict."""
    text = text.strip()

    if re.match(r"you (?:have\s+)?played\s+another\s+card", text, re.I):
        return {"cond_type": "legion"}

    if re.match(r".*mighty", text, re.I) or re.match(r".*(?:might|Might)\s*>=?\s*5", text, re.I):
        return {"cond_type": "mighty"}

    xp_match = re.match(r"you\s+have\s+(\d+)\+?\s+XP", text, re.I)
    if xp_match:
        return {"cond_type": "xp_gte", "params": {"threshold": int(xp_match.group(1))}}

    if re.match(r"you paid the additional cost", text, re.I):
        return {"cond_type": "additional_cost_paid"}

    if re.match(r"you do", text, re.I):
        return {"cond_type": "previous_effect_succeeded"}

    return {"cond_type": "always"}  # fallback: always true


# ---------------------------------------------------------------------------
# Activated ability cost parsing
# ---------------------------------------------------------------------------

def parse_activated_cost(text: str) -> tuple[dict | None, str]:
    """Parse an activated ability cost prefix like '[4], [T]: EFFECT'.

    Returns (cost_dict, remaining_effect_text).
    """
    # Pattern: [N], [T]: or [N][C], [T]: or [T]: or Recycle this:
    cost_match = re.match(
        r"^((?:\[\w+\][\s,]*)+):\s*(.*)",
        text,
    )
    if not cost_match:
        # Try "Recycle this:" or "Recycle a THING:"
        recycle_match = re.match(r"^Recycle\s+(?:this|me|a\s+\w+(?:\s+\w+)?)\s*(?:from\s+\w+\s*)?:\s*(.*)", text)
        if recycle_match:
            return ({"recycle_source": True}, recycle_match.group(1))
        return (None, text)

    cost_str = cost_match.group(1)
    effect_text = cost_match.group(2)

    cost: dict[str, Any] = {}
    # Parse cost components
    tokens = re.findall(r"\[(\w+)\]", cost_str)
    for tok in tokens:
        if tok == "T":
            cost["exhaust_source"] = True
        elif tok.isdigit():
            cost["energy"] = cost.get("energy", 0) + int(tok)
        elif tok in ("A",):
            cost.setdefault("power", {})["any"] = cost.get("power", {}).get("any", 0) + 1
        elif tok in ("C",):
            cost["power_of_domain"] = True  # resolved at play time
        elif tok in ("R", "G", "B", "O", "P", "Y"):
            domain = DOMAIN_SYMBOL_TO_ID.get(f"[{tok}]", "any")
            cost.setdefault("power", {})[domain] = cost.get("power", {}).get(domain, 0) + 1

    return (cost if cost else None, effect_text)


# ---------------------------------------------------------------------------
# CMS card -> engine CardDefinition dict
# ---------------------------------------------------------------------------

def convert_card(cms_card: dict) -> dict:
    """Convert a single CMS card to engine-native CardDefinition dict."""
    # Basic fields
    card_id = cms_card["id"]
    name = cms_card["name"]

    card_type_list = cms_card.get("cardType", {}).get("type", [])
    card_type = card_type_list[0]["id"] if card_type_list else "unit"

    # Domains
    domain_values = cms_card.get("domain", {}).get("values", [])
    domains = [d["id"] for d in domain_values if d.get("id") != "colorless"]

    # Stats
    cost_energy = 0
    energy_field = cms_card.get("energy", {})
    if energy_field:
        val = energy_field.get("value", {})
        if isinstance(val, dict):
            cost_energy = val.get("id", 0)
        elif isinstance(val, (int, float)):
            cost_energy = int(val)
    if isinstance(cost_energy, str):
        cost_energy = int(cost_energy) if cost_energy.isdigit() else 0

    # Power cost: single value mapped to card's first domain
    cost_power = {}
    power_field = cms_card.get("power", {})
    if power_field:
        val = power_field.get("value", {})
        power_amount = 0
        if isinstance(val, dict):
            power_amount = val.get("id", 0)
        elif isinstance(val, (int, float)):
            power_amount = int(val)
        if isinstance(power_amount, str):
            power_amount = int(power_amount) if power_amount.isdigit() else 0
        if power_amount > 0 and domains:
            cost_power[domains[0]] = power_amount

    # Might
    base_might = 0
    might_field = cms_card.get("might", {})
    if might_field:
        val = might_field.get("value", {})
        if isinstance(val, dict):
            base_might = val.get("id", 0)
        elif isinstance(val, (int, float)):
            base_might = int(val)
    if isinstance(base_might, str):
        base_might = int(base_might) if base_might.isdigit() else 0

    # Might bonus (gear)
    might_bonus = 0
    mb_field = cms_card.get("mightBonus", {})
    if mb_field:
        val = mb_field.get("value", {})
        if isinstance(val, dict):
            mb_id = val.get("id", 0)
            if isinstance(mb_id, str):
                mb_id = int(mb_id) if mb_id.lstrip("+-").isdigit() else 0
            might_bonus = int(mb_id)
        elif isinstance(val, (int, float)):
            might_bonus = int(val)

    # Tags
    tags_obj = cms_card.get("tags", {})
    tags = tags_obj.get("tags", []) if isinstance(tags_obj, dict) else []

    # Art
    art_url = cms_card.get("cardImage", {}).get("url", "")

    # Get ability text - prefer accessibilityText (cleaner)
    accessibility_text = cms_card.get("cardImage", {}).get("accessibilityText", "")
    rich_text_body = cms_card.get("text", {}).get("richText", {}).get("body", "")
    # Effect text (for gear with Effect Text field)
    effect_rich = cms_card.get("effect", {})
    effect_text_html = ""
    if effect_rich:
        effect_text_html = effect_rich.get("richText", {}).get("body", "")

    # Clean the accessibility text
    ability_text = _clean_accessibility_text(accessibility_text, name, card_type)
    effect_text = _clean_rich_text(effect_text_html) if effect_text_html else ""

    # Parse keywords from text
    keywords, text_after_keywords = _extract_keywords(ability_text)

    # Parse abilities
    abilities = _parse_abilities(text_after_keywords, effect_text, card_id, name, card_type, keywords)

    # Build result
    result: dict[str, Any] = {
        "card_id": card_id,
        "name": name,
        "card_type": card_type,
        "cost_energy": cost_energy,
        "base_might": base_might,
        "text": ability_text,
        "art_path": art_url,
    }

    if domains:
        result["domains"] = domains
    if cost_power:
        result["cost_power"] = cost_power
    if keywords:
        result["keywords"] = [{"keyword": k["keyword"], "value": k.get("value", 0)} for k in keywords]
    if abilities:
        result["abilities"] = abilities
    if tags:
        result["tags"] = tags
    if might_bonus:
        result["might_bonus"] = might_bonus
    if effect_text:
        result["effect_text"] = effect_text

    return result


def _clean_accessibility_text(text: str, name: str, card_type: str) -> str:
    """Strip preamble from accessibility text."""
    # Remove "Riftbound TYPE: NAME." prefix
    prefix_re = re.compile(
        r"^Riftbound\s+\w+:\s+" + re.escape(name) + r"\.\s*",
        re.IGNORECASE,
    )
    text = prefix_re.sub("", text)
    return text.strip()


def _clean_rich_text(html: str) -> str:
    """Convert richText HTML to clean text with symbol substitution."""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Replace symbol tokens
    for token, replacement in SYMBOL_MAP.items():
        text = text.replace(token, replacement)
    text = ENERGY_RE.sub(lambda m: f"[{m.group(1)}]", text)
    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_keywords(text: str) -> tuple[list[dict], str]:
    """Extract keyword instances and return (keywords, remaining_text)."""
    keywords: list[dict] = []
    remaining = text

    # Find all keyword matches
    for match in KEYWORD_RE.finditer(text):
        kw_name = match.group(1)
        kw_value = int(match.group(2)) if match.group(2) else 0
        kw_id = KEYWORD_NAME_TO_ID.get(kw_name)
        if kw_id:
            keywords.append({"keyword": kw_id, "value": kw_value})

    # Special: check for Equip in text
    equip_match = EQUIP_RE.search(text)
    if equip_match:
        keywords.append({"keyword": "equip", "value": 0})

    # Special: check for Repeat in text
    repeat_match = REPEAT_RE.search(text)
    if repeat_match:
        keywords.append({"keyword": "repeat", "value": 0})

    # Remove keyword brackets and reminder text from the remaining text
    remaining = KEYWORD_RE.sub("", remaining)
    remaining = EQUIP_RE.sub("", remaining)
    remaining = REPEAT_RE.sub("", remaining)
    remaining = REMINDER_RE.sub("", remaining)
    remaining = re.sub(r"\s+", " ", remaining).strip()

    # Deduplicate
    seen = set()
    unique_kws = []
    for kw in keywords:
        key = (kw["keyword"], kw["value"])
        if key not in seen:
            seen.add(key)
            unique_kws.append(kw)

    return unique_kws, remaining


def _parse_abilities(
    text: str, effect_text: str, card_id: str, card_name: str,
    card_type: str, keywords: list[dict],
) -> list[dict]:
    """Parse ability text into a list of AbilityDefinition dicts."""
    abilities: list[dict] = []

    if not text and not effect_text:
        return abilities

    # Split into individual abilities (by newlines or double-sentence breaks)
    ability_blocks = _split_ability_blocks(text)

    for i, block in enumerate(ability_blocks):
        block = block.strip()
        if not block:
            continue

        ability_id = f"{card_id}_ab_{i}"
        ability: dict[str, Any] = {
            "ability_id": ability_id,
            "text": block,
        }

        # Check for activated ability (cost: effect)
        cost, effect_body = parse_activated_cost(block)
        if cost is not None:
            ability["ability_type"] = "activated"
            ability["cost"] = cost
            # Determine timing from keywords
            if any(k["keyword"] == "reaction" for k in keywords):
                ability["timing"] = "reaction"
            else:
                ability["timing"] = "default"
            ir = parse_effect_text(effect_body, card_name)
            if ir:
                ability["effect_ir"] = ir
            abilities.append(ability)
            continue

        # Check for triggered ability
        trigger = None
        trigger_remaining = block
        for pattern, trigger_type in TRIGGER_COMPILED:
            m = pattern.match(block)
            if m:
                trigger = trigger_type
                trigger_remaining = block[m.end():].strip()
                break

        if trigger:
            ability["ability_type"] = "triggered"
            ability["trigger_condition"] = trigger
            ir = parse_effect_text(trigger_remaining, card_name)
            if ir:
                ability["effect_ir"] = ir
            abilities.append(ability)
            continue

        # Check for [>] Level ability prefix
        level_match = re.match(r"\[>\](?:\[>\])*\s*(.*)", block)
        if level_match:
            level_text = level_match.group(1).strip()
            ability["ability_type"] = "passive"
            ability["level_ability"] = True
            ir = parse_effect_text(level_text, card_name)
            if ir:
                ability["effect_ir"] = ir
            abilities.append(ability)
            continue

        # Check for passive ("While...", static text without trigger/cost)
        if block.lower().startswith("while ") or block.lower().startswith("your "):
            ability["ability_type"] = "passive"
            ability["trigger_condition"] = None
            # Try to parse inline effects from passive text
            ir = parse_effect_text(block, card_name)
            if ir:
                ability["effect_ir"] = ir
            abilities.append(ability)
            continue

        # For spells: the entire text IS the effect
        if card_type == "spell" and block:
            ability["ability_type"] = "activated"
            # Determine timing
            if any(k["keyword"] == "reaction" for k in keywords):
                ability["timing"] = "reaction"
            elif any(k["keyword"] == "action" for k in keywords):
                ability["timing"] = "action"
            else:
                ability["timing"] = "default"

            ir = parse_effect_text(block, card_name)
            if ir:
                ability["effect_ir"] = ir

            # Determine target requirements from the IR
            _annotate_targets(ability, ir)
            abilities.append(ability)
            continue

        # Fallback: try parsing as effect, else classify as passive
        if block and len(block) > 5:
            ir = parse_effect_text(block, card_name)
            if ir:
                ability["ability_type"] = "activated"
                ability["effect_ir"] = ir
            else:
                ability["ability_type"] = "passive"
            abilities.append(ability)

    return abilities


def _split_ability_blocks(text: str) -> list[str]:
    """Split ability text into separate ability blocks."""
    # Split on newlines first
    blocks = text.split("\n")
    result = []
    for block in blocks:
        block = block.strip()
        if block:
            result.append(block)
    if not result and text.strip():
        result = [text.strip()]
    return result


def _annotate_targets(ability: dict, ir: dict | None) -> None:
    """Set targets_required and target_type on ability based on IR analysis."""
    if not ir:
        return

    node_type = ir.get("type", "")
    target = ir.get("target", {})

    if node_type in ("deal_damage", "give_might", "stun", "kill", "buff",
                     "heal", "return_to_hand", "banish"):
        scope = target.get("scope", "any")
        obj_type = target.get("obj_type", "unit")
        count = target.get("count", 1)
        if count == -1:
            ability["targets_required"] = 0  # auto-all, no choice needed
        elif count >= 1:
            ability["targets_required"] = count
            if scope == "friendly":
                ability["target_type"] = f"friendly_{obj_type}"
            elif scope == "enemy":
                ability["target_type"] = f"enemy_{obj_type}"
            else:
                ability["target_type"] = obj_type

    elif node_type == "counter":
        ability["targets_required"] = 1
        ability["target_type"] = "spell_on_chain"

    elif node_type == "sequence":
        # Check first step for target needs
        steps = ir.get("steps", [])
        if steps:
            _annotate_targets(ability, steps[0])


# ---------------------------------------------------------------------------
# Rune special-casing
# ---------------------------------------------------------------------------

def make_basic_rune(card_id: str, name: str, domain: str, art_url: str = "") -> dict:
    """Create a basic rune CardDefinition with standard abilities."""
    return {
        "card_id": card_id,
        "name": name,
        "card_type": "rune",
        "domains": [domain],
        "text": f"[T]: [Reaction] - Add [1]. Recycle this: [Reaction] - Add [{DOMAIN_ID_TO_SYMBOL.get(domain, '[A]')}].",
        "art_path": art_url,
        "keywords": [],
        "abilities": [
            {
                "ability_id": f"{card_id}_exhaust",
                "ability_type": "activated",
                "cost": {"exhaust_source": True},
                "effect_script": "rune_add_energy",
                "timing": "reaction",
                "text": "Add 1 Energy",
            },
            {
                "ability_id": f"{card_id}_recycle",
                "ability_type": "activated",
                "cost": {},
                "effect_script": "rune_recycle_power",
                "timing": "reaction",
                "text": f"Recycle: Add 1 {domain.title()} Power",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

SET_PRIORITY = {"SFD": 3, "UNL": 2, "OGN": 1, "OGS": 0}


def deduplicate_cards(cards: list[dict]) -> list[dict]:
    """Deduplicate cards with same name, preferring newest set."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for c in cards:
        by_name[c["name"]].append(c)

    result = []
    for name, versions in by_name.items():
        if len(versions) == 1:
            result.append(versions[0])
        else:
            # Pick the version from the highest-priority set
            best = max(versions, key=lambda c: SET_PRIORITY.get(
                c.get("_set", ""), 0
            ))
            result.append(best)

    return result


# ---------------------------------------------------------------------------
# Pipeline main
# ---------------------------------------------------------------------------

def run_pipeline(input_path: str, output_path: str) -> dict:
    """Run the full card pipeline. Returns coverage stats."""
    with open(input_path, encoding="utf-8") as f:
        cms_cards = json.load(f)

    print(f"Loaded {len(cms_cards)} CMS cards")

    # Convert each card
    converted = []
    rune_names_seen: set[str] = set()

    for cms_card in cms_cards:
        card_type_list = cms_card.get("cardType", {}).get("type", [])
        card_type = card_type_list[0]["id"] if card_type_list else "unit"

        # Track set for dedup
        set_id = cms_card.get("set", {}).get("value", {}).get("id", "")

        if card_type == "rune":
            # Special-case runes with standard abilities
            name = cms_card["name"]
            if name in rune_names_seen:
                continue
            rune_names_seen.add(name)
            domain_values = cms_card.get("domain", {}).get("values", [])
            domain = domain_values[0]["id"] if domain_values else "fury"
            art_url = cms_card.get("cardImage", {}).get("url", "")
            card = make_basic_rune(cms_card["id"], name, domain, art_url)
        else:
            card = convert_card(cms_card)

        card["_set"] = set_id
        converted.append(card)

    # Deduplicate
    deduped = deduplicate_cards(converted)
    print(f"After dedup: {len(deduped)} unique cards")

    # Clean up internal fields
    for c in deduped:
        c.pop("_set", None)

    # Coverage stats
    stats = _compute_stats(deduped)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(deduped)} cards to {output_path}")
    print(f"Coverage: {stats['with_ir']} cards have effect_ir, "
          f"{stats['without_ir']} have no IR (keywords/passive only or unparsed)")
    print(f"By type: {stats['by_type']}")

    return stats


def _compute_stats(cards: list[dict]) -> dict:
    """Compute pipeline coverage statistics."""
    with_ir = 0
    without_ir = 0
    by_type: dict[str, int] = defaultdict(int)

    for c in cards:
        by_type[c.get("card_type", "?")] += 1
        has_ir = any(
            ab.get("effect_ir") for ab in c.get("abilities", [])
        )
        has_script = any(
            ab.get("effect_script") for ab in c.get("abilities", [])
        )
        if has_ir or has_script:
            with_ir += 1
        else:
            without_ir += 1

    return {
        "total": len(cards),
        "with_ir": with_ir,
        "without_ir": without_ir,
        "by_type": dict(by_type),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python -m app.engine.card_pipeline <input.json> <output.json>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    stats = run_pipeline(input_file, output_file)
    print(f"\nPipeline complete. Stats: {json.dumps(stats, indent=2)}")
