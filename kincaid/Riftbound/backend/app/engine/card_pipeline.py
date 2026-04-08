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
    (r"When I move from a (?:battlefield|location),?\s*", "on_move_from_bf"),
    (r"When I move,?\s*", "on_move"),
    (r"When I win a combat,?\s*", "on_combat_win"),
    (r"When I'm played and when I conquer,?\s*", "on_play_or_conquer"),
    (r"When a friendly unit dies,?\s*", "on_friendly_death"),
    (r"When another friendly unit dies,?\s*", "on_friendly_death"),
    (r"When another non-\w+ unit you control dies,?\s*", "on_friendly_death"),
    (r"When an enemy unit dies,?\s*", "on_enemy_death"),
    (r"When one or more enemy units die,?\s*", "on_enemy_death"),
    (r"When you play a spell,?\s*", "on_spell_played"),
    (r"When you play a unit,?\s*", "on_unit_played"),
    (r"When you play another unit,?\s*", "on_unit_played"),
    (r"When you play a gear,?\s*", "on_gear_played"),
    (r"When you play a card from (?:face down|\w+),?\s*", "on_play_from_facedown"),
    (r"When you play a card on an opponent's turn,?\s*", "on_play_on_opponent_turn"),
    (r"When a player plays a spell,?\s*", "on_any_spell_played"),
    (r"When a player plays a unit here,?\s*", "on_any_unit_played_here"),
    (r"When you attach an Equipment to me,?\s*", "on_equip"),
    (r"When you or an ally (?:conquer|hold),?\s*", "on_conquer_or_hold"),
    (r"When you win a combat,?\s*", "on_combat_win"),
    (r"When you recycle a rune,?\s*", "on_recycle_rune"),
    (r"When you recycle one or more cards,?\s*", "on_recycle"),
    (r"When you defend here,?\s*", "on_defend_here"),
    (r"When you defend at a battlefield,?\s*", "on_defend"),
    (r"When you stun (?:an|one or more) enemy units?,?\s*", "on_stun_enemy"),
    (r"When you \[Stun\] an enemy unit at a battlefield,?\s*", "on_stun_enemy"),
    (r"When you buff (?:a friendly unit|me),?\s*", "on_buff"),
    (r"When you ready a friendly unit,?\s*", "on_ready_unit"),
    (r"When you choose (?:a friendly unit|me|or ready me),?\s*", "on_choose_friendly"),
    (r"When you draw your second card each turn,?\s*", "on_second_draw"),
    (r"When you play your second card in a turn,?\s*", "on_second_play"),
    (r"When you discard (?:me|one or more cards),?\s*", "on_discard"),
    (r"When you kill a (?:stunned )?(?:enemy )?unit(?:\s+with a spell)?,?\s*", "on_kill"),
    (r"When you use an activated ability of a gear,?\s*", "on_gear_ability"),
    (r"When you hide a card,?\s*", "on_hide"),
    (r"When a unit (?:here )?(?:moves|is returned) from here,?\s*", "on_unit_leave_here"),
    (r"When a unit moves from here,?\s*", "on_unit_leave_here"),
    (r"When a buffed friendly unit dies,?\s*", "on_buffed_friendly_death"),
    (r"When a friendly unit attacks or defends alone,?\s*", "on_attack_or_defend_alone"),
    (r"When a showdown begins here,?\s*", "on_showdown_here"),
    (r"When an opponent (?:scores|plays a unit|moves to a battlefield),?\s*", "on_opponent_action"),
    (r"When one of your units becomes \w*,?\s*", "on_unit_status_change"),
    (r"When any unit takes damage this turn,?\s*", "on_any_damage"),
    (r"When this is played,?\s+(?:discarded,?\s+)?(?:or killed,?\s*)?", "on_play_or_discard"),
    (r"When this leaves the board,?\s*", "on_leave_board"),
    (r"When you play me (?:or (?:when I hold|another \w+|at the start)),?\s*", "on_play"),
    (r"When you play me to a battlefield,?\s*", "on_play"),
    (r"When you play me from face down(?: on your turn)?,?\s*", "on_play_from_facedown"),
    (r"When you play this from face down,?\s*", "on_play_from_facedown"),
    (r"When you conquer here with one or more units,?\s*", "on_conquer"),
    (r"When I conquer (?:a battlefield|after an attack),?\s*", "on_conquer"),
    (r"At the (?:start|beginning) of your (?:turn|Beginning Phase),?\s*", "on_turn_start"),
    (r"At the end of your turn,?\s*", "on_turn_end"),
    (r"At the start of each (?:turn|player's (?:first )?Beginning Phase),?\s*", "on_turn_start"),
    (r"At (?:the )?start of your Beginning Phase,?\s*", "on_turn_start"),
    (r"At the start of each player's Beginning Phase,?\s*", "on_each_turn_start"),
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

    # "you may pay [N] to EFFECT" (must be checked before generic "you may")
    may_pay_early = re.match(r"[Yy]ou\s+may\s+pay\s+((?:\[\w+\][\s]*)+)\s+to\s+(.*)", text)
    if may_pay_early:
        cost_tokens = re.findall(r"\[(\w+)\]", may_pay_early.group(1))
        inner = _parse_single_effect(may_pay_early.group(2))
        if inner:
            cost_dict: dict[str, Any] = {}
            for tok in cost_tokens:
                if tok == "C":
                    cost_dict["power_of_domain"] = True
                elif tok.isdigit():
                    cost_dict["energy"] = cost_dict.get("energy", 0) + int(tok)
                elif tok in ("A", "R", "G", "B", "O", "P", "Y"):
                    domain = DOMAIN_SYMBOL_TO_ID.get(f"[{tok}]", "any")
                    cost_dict.setdefault("power", {})[domain] = cost_dict.get("power", {}).get(domain, 0) + 1
            return {"type": "optional", "cost": cost_dict, "effect": inner}

    # "you may move me there/to that battlefield" -> optional move self
    may_move_me = re.match(r"[Yy]ou\s+may\s+move\s+me\s+(?:there|to\s+(?:that|this)\s+battlefield)", text)
    if may_move_me:
        return {"type": "optional", "effect": {"type": "move", "target": {"obj_type": "unit", "scope": "self", "count": 1}, "destination": {"zone": "battlefield", "location": "target"}}}

    # "you may X" -> optional wrapper (generic - catches remaining)
    may_match = re.match(r"[Yy]ou may\s+(.*)", text)
    if may_match:
        inner = _parse_single_effect(may_match.group(1))
        if inner:
            return {"type": "optional", "effect": inner}
        # Don't return None here - allow falling through to other patterns

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

    # "Give TARGET [Keyword N] [this turn]" - grant keyword (with optional value)
    give_kw_match = re.match(
        r"[Gg]ive\s+(.+?)\s+\[(\w+(?:-\w+)?)(?:\s+(\d+))?\](?:\s+this\s+turn)?",
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
            kw_value = int(give_kw_match.group(3)) if give_kw_match.group(3) else 0
            duration = "turn" if "this turn" in text.lower() else "permanent"
            node: dict[str, Any] = {"type": "give_keyword", "keyword": kw_id, "target": target, "duration": duration}
            if kw_value:
                node["value"] = kw_value
            return node

    # "Give TARGET [Temporary]" - keyword grant
    give_temp = re.match(r"[Gg]ive\s+(.+?)\s+\[Temporary\]", text)
    if give_temp:
        target = _parse_target(give_temp.group(1))
        return {"type": "give_keyword", "keyword": "temporary", "target": target, "duration": "permanent"}

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

    # --- NEW: expanded effect patterns for BE-06 ---

    # "Deal damage equal to my/its Might to TARGET"
    deal_eq_might = re.match(
        r"[Dd]eal\s+damage\s+equal\s+to\s+(?:my|its|their)\s+(?:Might|\[S\])\s+to\s+(.*)",
        text,
    )
    if deal_eq_might:
        target = _parse_target(deal_eq_might.group(1))
        return {"type": "deal_damage", "amount": "source_might", "target": target}

    # "deal damage equal to my [S|M] to TARGET" (with symbol reference)
    deal_eq_sym = re.match(
        r"deal\s+damage\s+equal\s+to\s+(?:my|its)\s+\[(?:S|M)\]\s+to\s+(.*)",
        text, re.I,
    )
    if deal_eq_sym:
        target = _parse_target(deal_eq_sym.group(1))
        return {"type": "deal_damage", "amount": "source_might", "target": target}

    # "deal N to TARGET" (lowercase, shorthand after trigger)
    deal_lower = re.match(r"deal\s+(\d+)\s+(?:damage\s+)?to\s+(.*)", text)
    if deal_lower:
        amount = int(deal_lower.group(1))
        target = _parse_target(deal_lower.group(2))
        return {"type": "deal_damage", "amount": amount, "target": target}

    # "Each player kills one of their units/gear"
    each_kills = re.match(
        r"[Ee]ach\s+player\s+kills?\s+one\s+of\s+their\s+(units?|gear)",
        text,
    )
    if each_kills:
        obj_type = "gear" if "gear" in each_kills.group(1) else "unit"
        return {"type": "kill", "target": {"obj_type": obj_type, "scope": "each_player", "count": 1}}

    # "Each player discards N / discards their hand, then draws N"
    each_discard = re.match(
        r"[Ee]ach\s+player\s+discards?\s+(?:their\s+hand|(\d+))",
        text,
    )
    if each_discard:
        if each_discard.group(1):
            return {"type": "discard", "count": int(each_discard.group(1)), "player": "each"}
        else:
            steps_ed: list[dict[str, Any]] = [{"type": "discard", "count": -1, "player": "each"}]
            draw_m = re.search(r"then\s+draws?\s+(\d+)", text)
            if draw_m:
                steps_ed.append({"type": "draw_cards", "count": int(draw_m.group(1)), "player": "each"})
            return {"type": "sequence", "steps": steps_ed} if len(steps_ed) > 1 else steps_ed[0]

    # "Each player channels N rune[s] exhausted"
    each_channel = re.match(r"[Ee]ach\s+player\s+channels?\s+(\d+)\s+runes?\s*(?:exhausted)?", text)
    if each_channel:
        return {"type": "channel_rune", "count": int(each_channel.group(1)),
                "player": "each", "exhausted": "exhausted" in text.lower()}

    # "Deal N damage split among any number of enemy units here"
    deal_split = re.match(
        r"[Dd]eal\s+(\d+)\s+(?:damage\s+)?split\s+among\s+(?:any\s+number\s+of\s+)?(.+)",
        text,
    )
    if deal_split:
        target = _parse_target(deal_split.group(2))
        return {"type": "split_damage", "amount": int(deal_split.group(1)), "target": target}

    # "Play two/three WORD unit tokens [here]"
    play_generic_tokens = re.match(
        r"[Pp]lay\s+(\w+)\s+(\w+)\s+unit\s+tokens?\s*(?:here)?",
        text,
    )
    if play_generic_tokens:
        count = _word_to_number(play_generic_tokens.group(1))
        name = play_generic_tokens.group(2).strip()
        dest = "here" if "here" in text.lower() else None
        node_pgt: dict[str, Any] = {"type": "play_token", "name": name, "might": 0, "count": count}
        if dest:
            node_pgt["destination"] = dest
        return node_pgt

    # "move an enemy unit [here] to its base" / "move an enemy unit to base"
    move_enemy_base = re.match(
        r"[Mm]ove\s+an\s+enemy\s+unit\s+(?:here\s+)?to\s+(?:its\s+)?base",
        text,
    )
    if move_enemy_base:
        return {"type": "move", "target": {"obj_type": "unit", "scope": "enemy", "count": 1},
                "destination": {"zone": "base", "location": "owner"}}

    # "move an enemy unit to here" / "move an enemy unit to my/this battlefield"
    move_enemy_here = re.match(
        r"[Mm]ove\s+an\s+enemy\s+unit\s+(?:at\s+a\s+(?:different\s+)?location\s+)?to\s+(?:here|my\s+battlefield|this\s+battlefield)",
        text,
    )
    if move_enemy_here:
        return {"type": "move", "target": {"obj_type": "unit", "scope": "enemy", "count": 1},
                "destination": {"zone": "battlefield", "location": "here"}}

    # "recycle one of your runes"
    recycle_rune = re.match(r"(?:you\s+must\s+)?[Rr]ecycle\s+one\s+of\s+your\s+runes?", text)
    if recycle_rune:
        return {"type": "recycle", "target": {"obj_type": "rune", "scope": "friendly", "count": 1}}

    # "ready your legend" / "ready your runes"
    ready_your = re.match(r"[Rr]eady\s+(?:your|my)\s+(\w+)", text)
    if ready_your:
        obj = ready_your.group(1).lower()
        if obj in ("legend", "legends"):
            return {"type": "ready", "target": {"obj_type": "legend", "scope": "friendly", "count": 1}}
        elif obj in ("rune", "runes"):
            return {"type": "ready", "target": {"obj_type": "rune", "scope": "friendly", "count": -1}}

    # "Return all units and gear to their owners' hands"
    if re.match(r"[Rr]eturn\s+all\s+units?\s+(?:and\s+gear\s+)?to\s+their\s+owners['\u2019]?\s*hands?", text):
        return {"type": "return_to_hand", "target": {"obj_type": "unit", "scope": "any", "count": -1}}

    # "Return a friendly unit and an enemy unit to their owners' hands"
    return_both = re.match(
        r"[Rr]eturn\s+(?:a|another)\s+friendly\s+unit\s+and\s+an\s+enemy\s+unit\s+to\s+their\s+owners['\u2019]?\s*hands?",
        text,
    )
    if return_both:
        return {"type": "sequence", "steps": [
            {"type": "return_to_hand", "target": {"obj_type": "unit", "scope": "friendly", "count": 1}},
            {"type": "return_to_hand", "target": {"obj_type": "unit", "scope": "enemy", "count": 1}},
        ]}

    # "Other friendly units [here] have +N [S]" - static aura
    other_units_might = re.match(
        r"[Oo]ther\s+friendly\s+units?\s*(?:here\s+)?have\s+\+(\d+)\s+\[S\]",
        text,
    )
    if other_units_might:
        location = "here" if "here" in text.lower() else "any"
        return {
            "type": "aura", "effect": "give_might",
            "amount": int(other_units_might.group(1)),
            "target": {"obj_type": "unit", "scope": "friendly", "count": -1, "location": location,
                        "filters": [{"field": "not_self", "op": "eq", "value": True}]},
            "duration": "permanent",
        }

    # "Units here have +N [S]"
    units_here_might = re.match(r"[Uu]nits?\s+here\s+have\s+\+(\d+)\s+\[S\]", text)
    if units_here_might:
        return {
            "type": "aura", "effect": "give_might",
            "amount": int(units_here_might.group(1)),
            "target": {"obj_type": "unit", "scope": "any", "count": -1, "location": "here"},
            "duration": "permanent",
        }

    # "Your token units have +N [S]" / "Your X have +N [S]"
    your_units_might = re.match(
        r"[Yy]our\s+(?:(\w+(?:\s+\w+)?)\s+)?(?:units?\s+)?have\s+\+(\d+)\s+\[S\]",
        text,
    )
    if your_units_might:
        tag = your_units_might.group(1)
        amount = int(your_units_might.group(2))
        target: dict[str, Any] = {"obj_type": "unit", "scope": "friendly", "count": -1}
        if tag:
            target["filters"] = [{"field": "tag", "op": "eq", "value": tag.strip().lower()}]
        return {"type": "aura", "effect": "give_might", "amount": amount, "target": target, "duration": "permanent"}

    # "Your spells and abilities deal N Bonus Damage"
    bonus_dmg = re.match(
        r"[Yy]our\s+spells?\s+(?:and\s+abilities?\s+)?deal\s+(\d+)\s+Bonus\s+Damage",
        text,
    )
    if bonus_dmg:
        return {"type": "aura", "effect": "bonus_damage", "amount": int(bonus_dmg.group(1)), "scope": "controller"}

    # "Spells and abilities deal N Bonus Damage to units here"
    bonus_dmg2 = re.match(
        r"[Ss]pells?\s+and\s+abilities?\s+deal\s+(\d+)\s+Bonus\s+Damage",
        text,
    )
    if bonus_dmg2:
        return {"type": "aura", "effect": "bonus_damage", "amount": int(bonus_dmg2.group(1)), "scope": "all"}

    # "I cost [N] less [for each ...]"
    cost_less = re.match(r"I\s+cost\s+\[(\d+)\]\s+less(?:\s+for\s+each\s+(.+?))?(?:\s*,\s*to\s+a\s+minimum\s+of\s+\[(\d+)\])?$", text, re.I)
    if cost_less:
        node_out: dict[str, Any] = {"type": "cost_reduction", "amount": int(cost_less.group(1))}
        if cost_less.group(2):
            node_out["scaling"] = cost_less.group(2).strip()
        if cost_less.group(3):
            node_out["minimum"] = int(cost_less.group(3))
        return node_out

    # "I cost [N][C] less [for each ...]"
    cost_less_domain = re.match(
        r"I\s+cost\s+\[(\d+)\]\[([CARGBOPY])\]\s+less(?:\s+for\s+each\s+(.+))?",
        text, re.I,
    )
    if cost_less_domain:
        return {
            "type": "cost_reduction",
            "amount": int(cost_less_domain.group(1)),
            "domain": True,
            "scaling": cost_less_domain.group(3).strip() if cost_less_domain.group(3) else None,
        }

    # "I can't be chosen by enemy spells and abilities"
    if re.match(r"I\s+can['\u2019]?t\s+be\s+chosen\s+by\s+enemy\s+spells?\s+and\s+abilities?", text, re.I):
        return {"type": "give_keyword", "keyword": "hexproof", "target": {"obj_type": "unit", "scope": "self", "count": 1}, "duration": "permanent"}

    # "I can't move to base"
    if re.match(r"I\s+can['\u2019]?t\s+move\s+to\s+base", text, re.I):
        return {"type": "restrict", "restriction": "cant_move_to_base", "scope": "self", "duration": "permanent"}

    # "I can't be readied"
    if re.match(r"I\s+can['\u2019]?t\s+be\s+readied", text, re.I):
        return {"type": "restrict", "restriction": "cant_ready", "scope": "self", "duration": "permanent"}

    # "I don't deal combat damage"
    if re.match(r"I\s+don['\u2019]?t\s+deal\s+combat\s+damage", text, re.I):
        return {"type": "restrict", "restriction": "no_combat_damage", "scope": "self", "duration": "permanent"}

    # "I can be played to LOCATION"
    can_play_to = re.match(r"I\s+can\s+be\s+played\s+to\s+(.*)", text, re.I)
    if can_play_to:
        return {"type": "play_restriction", "allows": can_play_to.group(1).strip().rstrip(".")}

    # "You may play me to an open/occupied [enemy] battlefield"
    may_play_to = re.match(r"[Yy]ou\s+may\s+play\s+me\s+to\s+an?\s+(open|occupied)\s+(?:(enemy)\s+)?battlefield", text)
    if may_play_to:
        return {"type": "play_restriction", "allows": f"{may_play_to.group(1)}_{may_play_to.group(2) or 'any'}_battlefield"}

    # "Other friendly units enter ready"
    if re.match(r"[Oo]ther\s+friendly\s+units?\s+enter\s+ready", text):
        return {"type": "aura", "effect": "enter_ready", "target": {"obj_type": "unit", "scope": "friendly", "count": -1,
                "filters": [{"field": "not_self", "op": "eq", "value": True}]}}

    # "Your tokens enter ready"
    if re.match(r"[Yy]our\s+tokens?\s+enter\s+ready", text):
        return {"type": "aura", "effect": "enter_ready", "target": {"obj_type": "unit", "scope": "friendly", "count": -1,
                "filters": [{"field": "token", "op": "eq", "value": True}]}}

    # "I get +N [S] for each ..."
    scaled_might = re.match(r"I\s+(?:get|have)\s+\+(\d+)\s+\[S\]\s+for\s+each\s+(.*)", text, re.I)
    if scaled_might:
        return {
            "type": "give_might", "amount": int(scaled_might.group(1)),
            "scaling": scaled_might.group(2).strip().rstrip("."),
            "target": {"obj_type": "unit", "scope": "self", "count": 1},
            "duration": "permanent",
        }

    # "My Might is increased by ..."
    my_might_inc = re.match(r"[Mm]y\s+[Mm]ight\s+is\s+increased\s+by\s+(.*)", text)
    if my_might_inc:
        return {
            "type": "give_might", "amount": "variable",
            "scaling": my_might_inc.group(1).strip().rstrip("."),
            "target": {"obj_type": "unit", "scope": "self", "count": 1},
            "duration": "permanent",
        }

    # "Gain control of TARGET"
    gain_control = re.match(r"[Gg]ain\s+control\s+of\s+(.*)", text)
    if gain_control:
        target = _parse_target(gain_control.group(1))
        return {"type": "gain_control", "target": target}

    # "Take control of TARGET"
    take_control = re.match(r"[Tt]ake\s+control\s+of\s+(.*)", text)
    if take_control:
        target = _parse_target(take_control.group(1))
        return {"type": "gain_control", "target": target}

    # "Double a friendly unit's Might this turn"
    double_might = re.match(r"[Dd]ouble\s+(?:a\s+(?:friendly\s+)?unit['\u2019]?s?\s+|my\s+)Might\s+this\s+(?:turn|combat)", text)
    if double_might:
        scope = "self" if "my" in text.lower() else "friendly"
        return {"type": "give_might", "amount": "double", "target": {"obj_type": "unit", "scope": scope, "count": 1}, "duration": "turn"}

    # "Play a unit from your trash, ignoring its Energy cost"
    play_from_trash = re.match(
        r"[Pp]lay\s+a\s+unit\s+from\s+(?:your\s+)?trash(?:,?\s+ignoring\s+its\s+(?:Energy\s+)?cost)?",
        text,
    )
    if play_from_trash:
        return {"type": "play_from_trash", "obj_type": "unit", "ignore_cost": True}

    # "Play a unit ... from your trash"
    play_unit_trash2 = re.match(
        r"[Pp]lay\s+a\s+(?:unit|card)\s+.*?from\s+(?:your\s+)?trash",
        text,
    )
    if play_unit_trash2:
        return {"type": "play_from_trash", "obj_type": "unit", "ignore_cost": "ignoring" in text.lower()}

    # "Choose an enemy unit at a battlefield. Its owner places it on the top or bottom of their Main Deck"
    tuck = re.match(
        r"[Cc]hoose\s+an\s+enemy\s+unit\s+at\s+a\s+battlefield\.\s+Its\s+owner\s+places\s+it\s+on\s+the\s+(?:top\s+or\s+bottom|bottom)\s+of\s+their\s+Main\s+Deck",
        text,
    )
    if tuck:
        return {"type": "tuck", "target": {"obj_type": "unit", "scope": "enemy", "count": 1, "location": "battlefield"}}

    # "Choose an enemy unit at a battlefield. Take control of it and recall it"
    steal_recall = re.match(
        r"[Cc]hoose\s+an\s+enemy\s+unit\s+at\s+a\s+battlefield\.\s+Take\s+control\s+of\s+it\s+and\s+recall\s+it",
        text,
    )
    if steal_recall:
        return {"type": "sequence", "steps": [
            {"type": "gain_control", "target": {"obj_type": "unit", "scope": "enemy", "count": 1, "location": "battlefield"}},
            {"type": "move", "target": {"obj_type": "unit", "scope": "self", "count": 1}, "destination": {"zone": "base", "location": "owner"}},
        ]}

    # "Choose a friendly unit and an enemy unit. They deal damage equal to their Mights to each other"
    fight_match = re.match(
        r"[Cc]hoose\s+(?:a\s+friendly\s+unit\s+(?:anywhere\s+)?and\s+an\s+enemy\s+unit|two\s+units)\.\s+They\s+deal\s+damage\s+equal\s+to\s+their\s+Mights?\s+to\s+each\s+other",
        text,
    )
    if fight_match:
        return {"type": "fight", "target_a": {"obj_type": "unit", "scope": "friendly", "count": 1},
                "target_b": {"obj_type": "unit", "scope": "enemy", "count": 1}}

    # "Choose an opponent. They reveal their hand. Choose a card from it"
    hand_rip = re.match(
        r"[Cc]hoose\s+an\s+opponent\.\s+They\s+reveal\s+their\s+hand\.\s+(?:You\s+may\s+)?[Cc]hoose\s+a\s+(?:non-unit\s+)?card\s+from\s+it",
        text,
    )
    if hand_rip:
        if "discard" in text.lower():
            return {"type": "sequence", "steps": [
                {"type": "reveal_hand", "player": "opponent"},
                {"type": "discard", "count": 1, "player": "opponent", "chosen_by": "controller"},
            ]}
        if "recycle" in text.lower():
            return {"type": "sequence", "steps": [
                {"type": "reveal_hand", "player": "opponent"},
                {"type": "recycle", "target": {"obj_type": "card", "scope": "opponent_hand", "count": 1}},
            ]}

    # "Choose an opponent. They play a N [S] NAME unit token"
    opp_token = re.match(
        r"[Cc]hoose\s+an\s+opponent\.\s+They\s+play\s+a\s+(\d+)\s+\[S\]\s+(\w+)\s+unit\s+token",
        text,
    )
    if opp_token:
        return {"type": "play_token", "name": opp_token.group(2), "might": int(opp_token.group(1)),
                "controller": "opponent"}

    # "Choose a friendly unit. The next time it would die this turn"
    save_unit = re.match(
        r"[Cc]hoose\s+a\s+friendly\s+unit\.\s+The\s+next\s+time\s+it\s+would\s+die\s+this\s+turn",
        text,
    )
    if save_unit:
        return {"type": "shield_from_death", "target": {"obj_type": "unit", "scope": "friendly", "count": 1}, "duration": "turn"}

    # "Choose a unit. Double all damage that would be dealt to it this turn"
    double_dmg = re.match(r"[Cc]hoose\s+a\s+unit\.\s+Double\s+all\s+damage", text)
    if double_dmg:
        return {"type": "modify_damage", "multiplier": 2, "target": {"obj_type": "unit", "scope": "any", "count": 1}, "duration": "turn"}

    # "Choose a friendly unit. It deals damage equal to its Might split among enemy units"
    split_dmg = re.match(
        r"[Cc]hoose\s+a\s+friendly\s+unit\.\s+It\s+deals\s+damage\s+equal\s+to\s+its\s+Might\s+split\s+among\s+enemy\s+units",
        text,
    )
    if split_dmg:
        return {"type": "split_damage", "amount": "source_might",
                "target": {"obj_type": "unit", "scope": "enemy", "count": -1, "location": "battlefield"}}

    # "Choose a friendly unit at a battlefield. Counter an enemy spell or ability"
    counter_spell = re.match(
        r"[Cc]hoose\s+a\s+friendly\s+unit\s+at\s+a\s+battlefield\.\s+Counter\s+an?\s+enemy\s+spell\s+or\s+ability",
        text,
    )
    if counter_spell:
        return {"type": "counter", "target": {"obj_type": "spell", "scope": "enemy", "zone": "chain"}}

    # "Choose a friendly unit in your base. Deal damage equal to its Might to all enemy units at a battlefield"
    base_slam = re.match(
        r"[Cc]hoose\s+a\s+friendly\s+unit\s+in\s+your\s+base\.\s+Deal\s+damage\s+equal\s+to\s+its\s+Might\s+to\s+all\s+enemy\s+units?\s+at\s+a\s+battlefield",
        text,
    )
    if base_slam:
        return {"type": "sequence", "steps": [
            {"type": "deal_damage", "amount": "source_might", "target": {"obj_type": "unit", "scope": "enemy", "count": -1, "location": "battlefield"}},
            {"type": "move", "target": {"obj_type": "unit", "scope": "friendly", "count": 1}, "destination": {"zone": "battlefield", "location": "target"}},
        ]}

    # "Choose an equipped friendly unit. It deals damage equal to its Might to an enemy unit"
    equip_strike = re.match(
        r"[Cc]hoose\s+an\s+equipped\s+friendly\s+unit\.\s+It\s+deals\s+damage\s+equal\s+to\s+its\s+Might\s+to\s+an\s+enemy\s+unit",
        text,
    )
    if equip_strike:
        return {"type": "sequence", "steps": [
            {"type": "deal_damage", "amount": "source_might", "target": {"obj_type": "unit", "scope": "enemy", "count": 1}},
            {"type": "detach", "target": {"obj_type": "gear", "scope": "source", "count": 1}},
        ]}

    # "Move up to N friendly units to base"
    move_to_base = re.match(r"[Mm]ove\s+(?:up\s+to\s+)?(\d+)\s+friendly\s+units?\s+to\s+base", text)
    if move_to_base:
        return {"type": "move", "target": {"obj_type": "unit", "scope": "friendly", "count": int(move_to_base.group(1))},
                "destination": {"zone": "base", "location": "owner"}}

    # "Prevent all spell and ability damage this turn"
    if re.match(r"[Pp]revent\s+all\s+(?:spell\s+and\s+ability\s+)?damage\s+this\s+turn", text):
        return {"type": "prevent_damage", "scope": "all", "duration": "turn"}

    # "Swap the Might of two units at the same battlefield this turn"
    if re.match(r"[Ss]wap\s+the\s+Might\s+of\s+two\s+units", text):
        return {"type": "swap_might", "target": {"obj_type": "unit", "scope": "any", "count": 2}, "duration": "turn"}

    # "Increase the points needed to win the game by N"
    increase_points = re.match(r"[Ii]ncrease\s+the\s+points\s+needed\s+to\s+win.*?by\s+(\d+)", text)
    if increase_points:
        return {"type": "modify_win_condition", "points_increase": int(increase_points.group(1))}

    # "Units can't move from here to base" / "Units can't be played here"
    units_restrict = re.match(r"[Uu]nits?\s+can['\u2019]?t\s+(.*)", text)
    if units_restrict:
        return {"type": "restrict", "restriction": units_restrict.group(1).strip().rstrip("."), "scope": "units", "duration": "permanent"}

    # "Units can move here from anywhere"
    units_allow = re.match(r"[Uu]nits?\s+can\s+move\s+here\s+from\s+anywhere", text)
    if units_allow:
        return {"type": "allow", "permission": "move_here_from_anywhere", "scope": "units"}

    # "you may pay [N] to EFFECT" (inline cost in effect)
    may_pay = re.match(r"you\s+may\s+pay\s+\[(\d+)\](?:\[(\w)\])?\s+to\s+(.*)", text, re.I)
    if may_pay:
        inner = _parse_single_effect(may_pay.group(3))
        if inner:
            cost_node: dict[str, Any] = {"energy": int(may_pay.group(1))} if may_pay.group(1) else {}
            if may_pay.group(2):
                domain = DOMAIN_SYMBOL_TO_ID.get(f"[{may_pay.group(2)}]", "any")
                cost_node.setdefault("power", {})[domain] = 1
            return {"type": "optional", "cost": cost_node, "effect": inner}

    # "you may pay [C] to EFFECT"
    may_pay_c = re.match(r"you\s+may\s+pay\s+\[C\]\s+to\s+(.*)", text, re.I)
    if may_pay_c:
        inner = _parse_single_effect(may_pay_c.group(1))
        if inner:
            return {"type": "optional", "cost": {"power_of_domain": True}, "effect": inner}

    # "give me +N [S] this turn"
    give_me_might = re.match(r"give\s+me\s+\+?(\d+)\s+\[S\]\s+this\s+turn", text, re.I)
    if give_me_might:
        return {"type": "give_might", "amount": int(give_me_might.group(1)),
                "target": {"obj_type": "unit", "scope": "self", "count": 1}, "duration": "turn"}

    # "give me -N [S] this turn"
    give_me_neg = re.match(r"give\s+me\s+-(\d+)\s+\[S\]\s+this\s+turn", text, re.I)
    if give_me_neg:
        return {"type": "give_might", "amount": -int(give_me_neg.group(1)),
                "target": {"obj_type": "unit", "scope": "self", "count": 1}, "duration": "turn"}

    # "Discard N, then draw N"
    discard_draw = re.match(r"[Dd]iscard\s+(\d+),?\s+then\s+draw\s+(\d+)", text)
    if discard_draw:
        return {"type": "sequence", "steps": [
            {"type": "discard", "count": int(discard_draw.group(1)), "player": "controller"},
            {"type": "draw_cards", "count": int(discard_draw.group(2)), "player": "controller"},
        ]}

    # "Channel N runes exhausted [and draw N]"
    channel_exh = re.match(r"[Cc]hannel\s+(\d+)\s+runes?\s+exhausted(?:\s+and\s+draw\s+(\d+))?", text)
    if channel_exh:
        steps_ch: list[dict[str, Any]] = [{"type": "channel_rune", "count": int(channel_exh.group(1)), "exhausted": True}]
        if channel_exh.group(2):
            steps_ch.append({"type": "draw_cards", "count": int(channel_exh.group(2)), "player": "controller"})
        return steps_ch[0] if len(steps_ch) == 1 else {"type": "sequence", "steps": steps_ch}

    # "[Add] [C]" where C is a domain letter
    add_domain = re.match(r"\[Add\]\s*\[([CARGBOPY])\]", text)
    if add_domain:
        tok = add_domain.group(1)
        if tok == "C":
            return {"type": "add_power", "domain": "card_domain", "amount": 1}
        domain = DOMAIN_SYMBOL_TO_ID.get(f"[{tok}]", "any")
        return {"type": "add_power", "domain": domain, "amount": 1}

    # "[Add] [N]" with numeric energy
    add_energy_match = re.match(r"\[Add\]\s*\[(\d+)\]", text)
    if add_energy_match:
        return {"type": "add_energy", "amount": int(add_energy_match.group(1))}

    # "play a N [S] NAME unit token" (catches lowercase and variations missed above)
    token_generic = re.match(
        r"[Pp]lay\s+a\s+(?:ready\s+)?(\d+)\s+\[S\]\s+([\w\s]+?)\s*unit\s+token(?:\s+with\s+\[?\w+\]?)?(?:\s+(?:into|in|to)\s+(your\s+base|here))?",
        text,
    )
    if token_generic:
        might = int(token_generic.group(1))
        name = token_generic.group(2).strip()
        dest = token_generic.group(3) or ""
        node_tok: dict[str, Any] = {
            "type": "play_token", "name": name, "might": might,
            "temporary": "[Temporary]" in text or "Temporary" in text,
            "ready_on_enter": "ready" in text.lower().split("[")[0],
        }
        if "here" in dest:
            node_tok["destination"] = "here"
        return node_tok

    # "play N N [S] NAME unit tokens" (multiple tokens, lowercase)
    tokens_multi = re.match(
        r"[Pp]lay\s+(\w+)\s+(\d+)\s+\[S\]\s+([\w\s]+?)\s*unit\s+tokens?(?:\s+with)?",
        text,
    )
    if tokens_multi:
        count = _word_to_number(tokens_multi.group(1))
        might = int(tokens_multi.group(2))
        name = tokens_multi.group(3).strip()
        return {"type": "play_token", "name": name, "might": might, "count": count,
                "temporary": "Temporary" in text}

    # "Recycle me to EFFECT" (self-recycle as cost)
    recycle_me = re.match(r"[Rr]ecycle\s+me\s+to\s+(.*)", text)
    if recycle_me:
        inner = _parse_single_effect(recycle_me.group(1))
        if inner:
            return {"type": "optional", "cost": {"recycle_self": True}, "effect": inner}

    # "give a unit +N [S] this turn" (lowercase)
    give_unit_lower = re.match(r"give\s+(.+?)\s+([+-]?\d+)\s+\[S\]\s+this\s+turn", text, re.I)
    if give_unit_lower:
        target = _parse_target(give_unit_lower.group(1))
        return {"type": "give_might", "amount": int(give_unit_lower.group(2)),
                "target": target, "duration": "turn"}

    # "score N point[s]" (lowercase after trigger)
    score_lower = re.match(r"(?:you\s+)?score\s+(\d+)\s+points?", text, re.I)
    if score_lower:
        return {"type": "score_points", "amount": int(score_lower.group(1)), "player": "controller"}

    # "look at the top N cards of your Main Deck"
    look_at = re.match(r"[Ll]ook\s+at\s+the\s+top\s+(\d+)\s+cards?\s+of\s+your\s+Main\s+Deck", text)
    if look_at:
        return {"type": "look_at_top", "count": int(look_at.group(1))}

    # "reveal the top card of your Main Deck"
    reveal_top = re.match(r"[Rr]eveal\s+(?:the\s+)?top\s+(?:card|rune)\s+of\s+your", text)
    if reveal_top:
        return {"type": "reveal_top", "count": 1}

    # "ready me [and give me +N [S] this turn]"
    ready_me = re.match(r"[Rr]eady\s+me(?:\s+and\s+give\s+me\s+\+?(\d+)\s+\[S\]\s+this\s+turn)?", text)
    if ready_me:
        ready_node: dict[str, Any] = {"type": "ready", "target": {"obj_type": "unit", "scope": "self", "count": 1}}
        if ready_me.group(1):
            return {"type": "sequence", "steps": [
                ready_node,
                {"type": "give_might", "amount": int(ready_me.group(1)),
                 "target": {"obj_type": "unit", "scope": "self", "count": 1}, "duration": "turn"},
            ]}
        return ready_node

    # "buff me"
    if re.match(r"buff\s+me", text, re.I):
        return {"type": "buff", "target": {"obj_type": "unit", "scope": "self", "count": 1}}

    # "draw N and channel N rune[s] exhausted"
    draw_and_channel = re.match(r"[Dd]raw\s+(\d+)\s+and\s+channel\s+(\d+)\s+runes?\s+exhausted", text)
    if draw_and_channel:
        return {"type": "sequence", "steps": [
            {"type": "draw_cards", "count": int(draw_and_channel.group(1)), "player": "controller"},
            {"type": "channel_rune", "count": int(draw_and_channel.group(2)), "exhausted": True},
        ]}

    # "Counter a spell unless its controller pays [N]"
    counter_unless = re.match(r"[Cc]ounter\s+a\s+spell\s+unless\s+its\s+controller\s+pays\s+\[(\d+)\]", text)
    if counter_unless:
        return {"type": "counter_unless_pay", "amount": int(counter_unless.group(1)),
                "target": {"obj_type": "spell", "zone": "chain"}}

    # "Counter an enemy spell or ability"
    counter_enemy = re.match(r"[Cc]ounter\s+an\s+enemy\s+spell\s+or\s+ability", text)
    if counter_enemy:
        return {"type": "counter", "target": {"obj_type": "spell", "scope": "enemy", "zone": "chain"}}

    # "put the top N cards of your Main Deck into your trash"
    mill = re.match(r"put\s+the\s+top\s+(\d+)\s+cards?\s+of\s+your\s+Main\s+Deck\s+into\s+your\s+trash", text, re.I)
    if mill:
        return {"type": "mill", "count": int(mill.group(1)), "player": "controller"}

    # "When any unit takes damage this turn, kill it"
    if re.match(r"[Ww]hen\s+any\s+unit\s+takes\s+damage\s+this\s+turn,?\s+kill\s+it", text):
        return {"type": "aura", "effect": "damage_kills", "duration": "turn"}

    # "Starting with the next player, each player may ..."
    starting_next = re.match(r"[Ss]tarting\s+with\s+the\s+next\s+player,?\s+each\s+player\s+may\s+(.*)", text)
    if starting_next:
        inner = _parse_single_effect(starting_next.group(1))
        if inner:
            return {"type": "optional", "effect": inner, "scope": "each_player_from_next"}

    # --- Additional patterns for IR coverage expansion ---

    # "While I'm at a battlefield, opponents can't gain points"
    if re.match(r"[Ww]hile\s+I['\u2019]?m\s+at\s+a\s+battlefield,?\s+opponents?\s+can['\u2019]?t\s+gain\s+points?", text):
        return {"type": "restrict", "restriction": "cant_gain_points", "scope": "opponents", "duration": "while_on_board"}

    # "While I'm attacking or defending alone, I have +N [S]"
    while_alone = re.match(
        r"[Ww]hile\s+I['\u2019]?m\s+(?:attacking\s+or\s+defending|in\s+combat)\s+alone,?\s+I\s+have\s+\+(\d+)\s+\[S\]",
        text,
    )
    if while_alone:
        return {"type": "conditional", "condition": {"cond_type": "in_combat_alone"},
                "then": {"type": "give_might", "amount": int(while_alone.group(1)),
                         "target": {"obj_type": "unit", "scope": "self", "count": 1}, "duration": "combat"}}

    # "While a friendly unit defends/attacks alone, it gets +N [S]"
    while_unit_alone = re.match(
        r"[Ww]hile\s+a\s+friendly\s+unit\s+(?:defends|attacks|defends\s+or\s+attacks|attacks\s+or\s+defends)\s+alone,?\s+it\s+(?:gets|has)\s+\+(\d+)\s+\[S\]",
        text,
    )
    if while_unit_alone:
        return {"type": "aura", "effect": "give_might",
                "amount": int(while_unit_alone.group(1)),
                "condition": {"cond_type": "in_combat_alone"},
                "target": {"obj_type": "unit", "scope": "friendly", "count": -1}, "duration": "combat"}

    # "While I'm buffed, I have an additional +N [S]"
    while_buffed = re.match(r"[Ww]hile\s+I['\u2019]?m\s+buffed,?\s+I\s+have\s+(?:an\s+additional\s+)?\+(\d+)\s+\[S\]", text)
    if while_buffed:
        return {"type": "conditional", "condition": {"cond_type": "has_buff"},
                "then": {"type": "give_might", "amount": int(while_buffed.group(1)),
                         "target": {"obj_type": "unit", "scope": "self", "count": 1}, "duration": "permanent"}}

    # "While you have another unit here, I have +N [S]"
    while_another_unit = re.match(
        r"[Ww]hile\s+you\s+have\s+another\s+unit\s+here,?\s+I\s+have\s+\+(\d+)\s+\[S\]",
        text,
    )
    if while_another_unit:
        return {"type": "conditional", "condition": {"cond_type": "has_ally_here"},
                "then": {"type": "give_might", "amount": int(while_another_unit.group(1)),
                         "target": {"obj_type": "unit", "scope": "self", "count": 1}, "duration": "permanent"}}

    # "While you have N+ runes, I have +N [S]"
    while_runes = re.match(r"[Ww]hile\s+you\s+have\s+(\d+)\+?\s+runes?,?\s+I\s+have\s+\+(\d+)\s+\[S\]", text)
    if while_runes:
        return {"type": "conditional",
                "condition": {"cond_type": "rune_count_gte", "params": {"threshold": int(while_runes.group(1))}},
                "then": {"type": "give_might", "amount": int(while_runes.group(2)),
                         "target": {"obj_type": "unit", "scope": "self", "count": 1}, "duration": "permanent"}}

    # "While I'm in a showdown, AURA_EFFECT"
    while_showdown = re.match(r"[Ww]hile\s+I['\u2019]?m\s+in\s+a\s+showdown,?\s+(.*)", text)
    if while_showdown:
        inner = _parse_single_effect(while_showdown.group(1))
        if inner:
            return {"type": "conditional", "condition": {"cond_type": "in_showdown"}, "then": inner}
        return {"type": "aura", "effect": "showdown_aura", "text": while_showdown.group(1).strip(), "duration": "showdown"}

    # "While I'm in combat, EFFECT"
    while_combat = re.match(r"[Ww]hile\s+I['\u2019]?m\s+in\s+combat,?\s+(.*)", text)
    if while_combat:
        inner = _parse_single_effect(while_combat.group(1))
        if inner:
            return {"type": "conditional", "condition": {"cond_type": "in_combat"}, "then": inner}
        return {"type": "aura", "effect": "combat_aura", "text": while_combat.group(1).strip(), "duration": "combat"}

    # "The first time EVENT each turn, EFFECT" (also: "during your Beginning Phase each turn")
    first_time = re.match(
        r"[Tt]he\s+first\s+time\s+(.*?)\s+(?:during\s+your\s+\w+\s+Phase\s+)?each\s+turn,?\s+(.*)",
        text,
    )
    if first_time:
        event_text = first_time.group(1).strip()
        effect_text_ft = first_time.group(2).strip()
        effect = _parse_single_effect(effect_text_ft)
        if not effect:
            # Try "they draw N" etc
            draw_m = re.match(r"(?:they|that\s+player)\s+draw\s+(\d+)", effect_text_ft, re.I)
            if draw_m:
                effect = {"type": "draw_cards", "count": int(draw_m.group(1)), "player": "trigger_player"}
            # Try "each opponent must kill one of their units"
            each_opp_kill = re.match(r"each\s+opponent\s+(?:must\s+)?kills?\s+one\s+of\s+their\s+(\w+)", effect_text_ft, re.I)
            if each_opp_kill:
                obj = "unit" if "unit" in each_opp_kill.group(1) else each_opp_kill.group(1)
                effect = {"type": "kill", "target": {"obj_type": obj, "scope": "each_opponent", "count": 1}}
            # Try "they may move another unit..."
            move_m = re.match(r"(?:they|that\s+player)\s+may\s+(.*)", effect_text_ft, re.I)
            if move_m and not effect:
                inner = _parse_single_effect(move_m.group(1))
                if inner:
                    effect = {"type": "optional", "effect": inner}
        # classify the event
        event_type = "generic_event"
        if "dies" in event_text or "die" in event_text:
            event_type = "on_friendly_death" if "friendly" in event_text else "on_death"
        elif "plays" in event_text or "play" in event_text:
            event_type = "on_unit_played"
        elif "chooses" in event_text or "choose" in event_text:
            event_type = "on_choose"
        # Check for phase constraint
        phase = None
        phase_m = re.search(r"during\s+your\s+(\w+)\s+Phase", text)
        if phase_m:
            phase = phase_m.group(1).lower()
        if effect:
            node_ft: dict[str, Any] = {"type": "triggered_effect", "trigger_event": event_type, "once_per_turn": True,
                    "event_description": event_text, "effect_ir": effect}
            if phase:
                node_ft["phase"] = phase
            return node_ft

    # "When EVENT for the first time each turn, EFFECT" (alternate ordering)
    when_first_time = re.match(
        r"[Ww]hen\s+(.*?)\s+for\s+the\s+first\s+time\s+each\s+turn,?\s+(.*)",
        text,
    )
    if when_first_time:
        event_text = when_first_time.group(1).strip()
        effect_text_ft = when_first_time.group(2).strip()
        effect = _parse_single_effect(effect_text_ft)
        if not effect:
            draw_m = re.match(r"(?:they|that\s+player)\s+draw\s+(\d+)", effect_text_ft, re.I)
            if draw_m:
                effect = {"type": "draw_cards", "count": int(draw_m.group(1)), "player": "trigger_player"}
        event_type = "generic_event"
        if "chooses" in event_text or "choose" in event_text:
            event_type = "on_choose"
        elif "plays" in event_text or "play" in event_text:
            event_type = "on_unit_played"
        if effect:
            return {"type": "triggered_effect", "trigger_event": event_type, "once_per_turn": True,
                    "event_description": event_text, "effect_ir": effect}

    # "The Nth time EVENT in a turn, EFFECT"
    nth_time = re.match(r"[Tt]he\s+(\w+)\s+time\s+(.*?)\s+in\s+a\s+turn,?\s+(.*)", text)
    if nth_time:
        count = _word_to_number(nth_time.group(1))
        event_text_n = nth_time.group(2).strip()
        effect_text_n = nth_time.group(3).strip()
        effect = _parse_single_effect(effect_text_n)
        if effect:
            return {"type": "triggered_effect", "trigger_event": "nth_occurrence",
                    "occurrence_count": count, "event_description": event_text_n, "effect_ir": effect}

    # "Choose one — - OPTION1. - OPTION2" (with dashes)
    choose_one_m = re.match(r"[Cc]hoose\s+one\s*[\u2014\u2013\-]+\s*[-\u2022]\s*(.*)", text)
    if choose_one_m:
        options_text = choose_one_m.group(1)
        # Split on "- " or "* " or "\u2022 "
        option_parts = re.split(r'\s*[-\u2022]\s+', options_text)
        options = []
        for opt in option_parts:
            opt = opt.strip().rstrip(".")
            if opt:
                parsed = _parse_single_effect(opt)
                if parsed:
                    options.append({"label": opt, "effect": parsed})
                else:
                    options.append({"label": opt, "effect": {"type": "noop", "text": opt}})
        if options:
            return {"type": "choose_one", "options": [o["effect"] for o in options]}

    # "Spend my buff: EFFECT" (activated with buff cost)
    spend_buff = re.match(r"[Ss]pend\s+(?:my|its)\s+buff:\s*(.*)", text)
    if spend_buff:
        inner = _parse_single_effect(spend_buff.group(1))
        if inner:
            return {"type": "optional", "cost": {"spend_buff": True}, "effect": inner}
        # Check for choose_one with * delimiters
        choose_parts = re.split(r'\s*\*\s+', spend_buff.group(1))
        choose_parts = [p.strip().rstrip(".") for p in choose_parts if p.strip()]
        if len(choose_parts) > 1:
            options = []
            for opt in choose_parts:
                parsed = _parse_single_effect(opt)
                if parsed:
                    options.append(parsed)
            if options:
                return {"type": "optional", "cost": {"spend_buff": True},
                        "effect": {"type": "choose_one", "options": options}}

    # "We deal damage equal to our Mights to each other"
    we_fight = re.match(
        r"(?:choose\s+an?\s+(?:enemy|friendly)\s+unit.*?\.\s*)?[Ww]e\s+deal\s+damage\s+equal\s+to\s+(?:our|their)\s+(?:Might|Mights?)s?\s+to\s+each\s+other",
        text,
    )
    if we_fight:
        return {"type": "fight", "target_a": {"obj_type": "unit", "scope": "self", "count": 1},
                "target_b": {"obj_type": "unit", "scope": "enemy", "count": 1}}

    # "choose an enemy unit at a battlefield. We deal damage..."
    choose_fight = re.match(
        r"[Cc]hoose\s+an\s+enemy\s+unit.*?\.\s+[Ww]e\s+deal\s+damage\s+equal\s+to\s+(?:our|their)\s+(?:Might|Mights?)s?\s+to\s+each\s+other",
        text,
    )
    if choose_fight:
        return {"type": "fight", "target_a": {"obj_type": "unit", "scope": "self", "count": 1},
                "target_b": {"obj_type": "unit", "scope": "enemy", "count": 1}}

    # "deal damage equal to my Might to an enemy unit in a base"
    deal_eq_base = re.match(
        r"deal\s+damage\s+equal\s+to\s+(?:my|its)\s+(?:Might|\[S\])\s+to\s+an\s+enemy\s+unit\s+in\s+a\s+base",
        text, re.I,
    )
    if deal_eq_base:
        return {"type": "deal_damage", "amount": "source_might",
                "target": {"obj_type": "unit", "scope": "enemy", "count": 1, "location": "base"}}

    # "I cost [N] less to play from LOCATION"
    cost_less_from = re.match(r"I\s+cost\s+\[(\d+)\]\s+less\s+to\s+play\s+from\s+(.*)", text, re.I)
    if cost_less_from:
        return {"type": "cost_reduction", "amount": int(cost_less_from.group(1)),
                "from_zone": cost_less_from.group(2).strip().rstrip(".")}

    # "you may play me from your trash for [N][C]"
    play_from_trash_self = re.match(
        r"[Yy]ou\s+may\s+play\s+me\s+from\s+(?:your\s+)?trash(?:\s+for\s+(?:\[\w+\]\s*)+)?",
        text,
    )
    if play_from_trash_self:
        return {"type": "play_restriction", "allows": "play_from_trash"}

    # "Move any number of enemy units ... to a single location"
    move_any_number = re.match(
        r"[Mm]ove\s+any\s+number\s+of\s+(enemy|friendly)\s+units?.*?to\s+a\s+single\s+location",
        text,
    )
    if move_any_number:
        scope = move_any_number.group(1).lower()
        return {"type": "move", "target": {"obj_type": "unit", "scope": scope, "count": -1},
                "destination": {"zone": "battlefield", "location": "chosen"}}

    # "that player gains/scores N point[s]" / "each player gains N point[s]"
    player_gains_pts = re.match(
        r"(?:that\s+player|each\s+player|they)\s+(?:gains?|scores?)\s+(\d+)\s+points?",
        text, re.I,
    )
    if player_gains_pts:
        return {"type": "score_points", "amount": int(player_gains_pts.group(1)),
                "player": "trigger_player"}

    # "you may pay [N] to [Buff] it/TARGET" (pay to buff)
    may_pay_buff = re.match(r"(?:they|that\s+player|you)\s+may\s+pay\s+\[(\d+)\]\s+to\s+\[Buff\]\s+(\w+)", text, re.I)
    if may_pay_buff:
        target = _parse_target(may_pay_buff.group(2))
        return {"type": "optional", "cost": {"energy": int(may_pay_buff.group(1))},
                "effect": {"type": "buff", "target": target}}

    # "you may pay [N] to draw N"
    may_pay_draw = re.match(r"(?:they|that\s+player|you)\s+may\s+pay\s+\[(\d+)\]\s+to\s+draw\s+(\d+)", text, re.I)
    if may_pay_draw:
        return {"type": "optional", "cost": {"energy": int(may_pay_draw.group(1))},
                "effect": {"type": "draw_cards", "count": int(may_pay_draw.group(2)), "player": "controller"}}

    # "you may pay [N] to channel N rune[s] exhausted"
    may_pay_channel = re.match(
        r"(?:they|that\s+player|you)\s+may\s+pay\s+\[(\d+)\]\s+to\s+channel\s+(\d+)\s+runes?\s+exhausted",
        text, re.I,
    )
    if may_pay_channel:
        return {"type": "optional", "cost": {"energy": int(may_pay_channel.group(1))},
                "effect": {"type": "channel_rune", "count": int(may_pay_channel.group(2)), "exhausted": True}}

    # "your non-token units cost [N] more to play this turn"
    cost_more = re.match(
        r"[Yy]our\s+(?:non-token\s+)?units?\s+cost\s+\[(\d+)\]\s+more\s+to\s+play\s+this\s+turn",
        text,
    )
    if cost_more:
        return {"type": "restrict", "restriction": "cost_increase", "amount": int(cost_more.group(1)),
                "scope": "friendly_units", "duration": "turn"}

    # "give me +N [S] this turn and give me [Keyword]" / "give a unit X and Y this turn" with multiple keywords
    give_kw_and_kw = re.match(
        r"[Gg]ive\s+(.+?)\s+\[(\w+(?:-\w+)?)\]\s+and\s+\[(\w+(?:-\w+)?)\]\s+this\s+turn",
        text,
    )
    if give_kw_and_kw:
        target = _parse_target(give_kw_and_kw.group(1))
        kw1 = KEYWORD_NAME_TO_ID.get(give_kw_and_kw.group(2), give_kw_and_kw.group(2).lower())
        kw2 = KEYWORD_NAME_TO_ID.get(give_kw_and_kw.group(3), give_kw_and_kw.group(3).lower())
        return {"type": "sequence", "steps": [
            {"type": "give_keyword", "keyword": kw1, "target": target, "duration": "turn"},
            {"type": "give_keyword", "keyword": kw2, "target": target, "duration": "turn"},
        ]}

    # "The next unit you play this turn enters ready"
    next_enters_ready = re.match(r"[Tt]he\s+next\s+unit\s+you\s+play\s+this\s+turn\s+enters\s+ready", text)
    if next_enters_ready:
        return {"type": "delayed_trigger", "trigger_event": "on_unit_played", "duration": "turn",
                "max_fires": 1, "effect_ir": {"type": "ready", "target": {"obj_type": "unit", "scope": "played", "count": 1}}}

    # "you may spend a buff to EFFECT"
    spend_buff_to = re.match(r"[Yy]ou\s+may\s+spend\s+a\s+buff\s+to\s+(.*)", text)
    if spend_buff_to:
        inner = _parse_single_effect(spend_buff_to.group(1))
        if inner:
            return {"type": "optional", "cost": {"spend_buff": True}, "effect": inner}

    # "buff me and ready me"
    buff_and_ready = re.match(r"buff\s+me\s+and\s+ready\s+me", text, re.I)
    if buff_and_ready:
        return {"type": "sequence", "steps": [
            {"type": "buff", "target": {"obj_type": "unit", "scope": "self", "count": 1}},
            {"type": "ready", "target": {"obj_type": "unit", "scope": "self", "count": 1}},
        ]}

    # "heal it, exhaust it, and recall it [instead]"
    heal_exhaust_recall = re.match(r"heal\s+it,?\s+exhaust\s+it,?\s+and\s+recall\s+it(?:\s+instead)?", text, re.I)
    if heal_exhaust_recall:
        return {"type": "sequence", "steps": [
            {"type": "heal", "amount": "all", "target": {"obj_type": "unit", "scope": "chosen", "count": 1}},
            {"type": "exhaust", "target": {"obj_type": "unit", "scope": "chosen", "count": 1}},
            {"type": "move", "target": {"obj_type": "unit", "scope": "chosen", "count": 1},
             "destination": {"zone": "base", "location": "owner"}},
        ]}

    # "look at the top N cards of your Main Deck. You may reveal TYPE and draw it. Then recycle the rest"
    look_reveal_draw = re.match(
        r"[Ll]ook\s+at\s+the\s+top\s+(\d+)\s+cards?\s+of\s+your\s+Main\s+Deck\.\s+[Yy]ou\s+may\s+reveal\s+a\s+(\w+)\s+from\s+among\s+them\s+and\s+draw\s+it\.\s+Then\s+recycle\s+the\s+rest",
        text,
    )
    if look_reveal_draw:
        return {"type": "sequence", "steps": [
            {"type": "look_at_top", "count": int(look_reveal_draw.group(1))},
            {"type": "optional", "effect": {"type": "draw_cards", "count": 1, "player": "controller",
             "filter": {"card_type": look_reveal_draw.group(2).lower()}}},
            {"type": "recycle", "target": {"obj_type": "card", "scope": "looked_at", "count": -1}},
        ]}

    # "gain N XP for each THING"
    xp_for_each = re.match(r"[Gg]ain\s+(\d+)\s+XP\s+for\s+each\s+(.*)", text)
    if xp_for_each:
        return {"type": "for_each",
                "targets": {"obj_type": "unit", "scope": "friendly", "count": -1},
                "effect": {"type": "gain_xp", "amount": int(xp_for_each.group(1))},
                "scaling": xp_for_each.group(2).strip().rstrip(".")}

    # "[Stun] it. They can't move it this turn"
    stun_and_restrict = re.match(r"\[Stun\]\s+it\.\s+They\s+can['\u2019]?t\s+move\s+it\s+this\s+turn", text)
    if stun_and_restrict:
        return {"type": "sequence", "steps": [
            {"type": "stun", "target": {"obj_type": "unit", "scope": "triggering", "count": 1}},
            {"type": "restrict", "restriction": "cant_move", "scope": "target", "duration": "turn"},
        ]}

    # "Enemy units here with less Might than me don't deal combat damage"
    enemy_no_combat = re.match(
        r"[Ee]nemy\s+units?\s+here\s+with\s+less\s+[Mm]ight\s+than\s+me\s+don['\u2019]?t\s+deal\s+combat\s+damage",
        text,
    )
    if enemy_no_combat:
        return {"type": "aura", "effect": "no_combat_damage",
                "target": {"obj_type": "unit", "scope": "enemy", "count": -1, "location": "here",
                           "filters": [{"field": "might", "op": "lt", "value": "source_might"}]},
                "duration": "permanent"}

    return None  # Unparseable


_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
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

    # "an attacking unit"
    if re.match(r"an?\s+attacking\s+unit", text, re.I):
        return {"obj_type": "unit", "scope": "any", "count": 1,
                "filters": [{"field": "combat_role", "op": "eq", "value": "attacker"}]}

    # "a defending unit"
    if re.match(r"an?\s+defending\s+unit", text, re.I):
        return {"obj_type": "unit", "scope": "any", "count": 1,
                "filters": [{"field": "combat_role", "op": "eq", "value": "defender"}]}

    # "an enemy unit at a battlefield" / "a unit at a battlefield"
    unit_at_bf = re.match(r"(?:a|an)\s+(?:(friendly|enemy)\s+)?unit\s+at\s+a\s+battlefield", text, re.I)
    if unit_at_bf:
        scope = unit_at_bf.group(1).lower() if unit_at_bf.group(1) else "any"
        return {"obj_type": "unit", "scope": scope, "count": 1, "location": "battlefield"}

    # "a unit in a base" / "an enemy unit in a base"
    unit_in_base = re.match(r"(?:a|an)\s+(?:(friendly|enemy)\s+)?unit\s+in\s+a\s+base", text, re.I)
    if unit_in_base:
        scope = unit_in_base.group(1).lower() if unit_in_base.group(1) else "any"
        return {"obj_type": "unit", "scope": scope, "count": 1, "location": "base"}

    # "all enemy units at a battlefield"
    all_at_bf = re.match(r"all\s+(?:(friendly|enemy)\s+)?units?\s+at\s+(?:a\s+|my\s+)?battlefield", text, re.I)
    if all_at_bf:
        scope = all_at_bf.group(1).lower() if all_at_bf.group(1) else "any"
        return {"obj_type": "unit", "scope": scope, "count": -1, "location": "battlefield"}

    # "a friendly unit anywhere" / "an enemy unit anywhere"
    unit_anywhere = re.match(r"(?:a|an)\s+(?:(friendly|enemy)\s+)?unit\s+anywhere", text, re.I)
    if unit_anywhere:
        scope = unit_anywhere.group(1).lower() if unit_anywhere.group(1) else "any"
        return {"obj_type": "unit", "scope": scope, "count": 1, "location": "any"}

    # "your runes" / "my runes"
    if re.match(r"(?:your|my)\s+runes?", text, re.I):
        return {"obj_type": "rune", "scope": "friendly", "count": -1}

    # "your legend"
    if re.match(r"(?:your|my)\s+legend", text, re.I):
        return {"obj_type": "legend", "scope": "friendly", "count": 1}

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
    # Also handles em-dash separator: [T]: — EFFECT or [T]:, — EFFECT
    cost_match = re.match(
        r"^((?:\[\w+\][\s,]*)+):\s*[,\s]*[\u2014\u2013\-]*\s*(.*)",
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
    abilities = _parse_abilities(text_after_keywords, effect_text, card_id, name, card_type, keywords, domains)

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
    card_type: str, keywords: list[dict], domains: list[str] | None = None,
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

        # Check for [Equip] ability on gear (rule 744)
        equip_match = re.match(r"\[Equip\]\s*(.*)", block)
        if equip_match and card_type == "gear":
            equip_rest = equip_match.group(1).strip()
            # Remove leading dash/emdash and reminder text in parentheses
            equip_rest = re.sub(r"^[—\-]\s*", "", equip_rest)
            equip_rest = re.sub(r"\s*\([^)]*\)\s*$", "", equip_rest).strip()
            # Parse cost tokens from e.g. "[C]", "[1][C]", "[C], Kill a friendly unit"
            cost: dict[str, Any] = {}
            cost_tokens = re.findall(r"\[(\w+)\]", equip_rest)
            for tok in cost_tokens:
                if tok == "T":
                    cost["exhaust_source"] = True
                elif tok.isdigit():
                    cost["energy"] = cost.get("energy", 0) + int(tok)
                elif tok == "C":
                    cost["power_of_domain"] = True
                elif tok == "A":
                    cost.setdefault("power", {})["any"] = cost.get("power", {}).get("any", 0) + 1
                elif tok in ("R", "G", "B", "O", "P", "Y"):
                    domain = DOMAIN_SYMBOL_TO_ID.get(f"[{tok}]", "any")
                    cost.setdefault("power", {})[domain] = cost.get("power", {}).get(domain, 0) + 1
            # Resolve [C] → actual domain power cost
            if cost.pop("power_of_domain", False) and domains:
                cost.setdefault("power", {})[domains[0]] = cost.get("power", {}).get(domains[0], 0) + 1
            ability["ability_type"] = "activated"
            ability["cost"] = cost
            ability["timing"] = "default"
            ability["targets_required"] = 1
            ability["target_type"] = "friendly_unit"
            ability["effect_ir"] = {"type": "attach"}
            abilities.append(ability)
            continue

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
        level_match = re.match(r"\[>\](?:\[>\]|\[>>\])*\s*(.*)", block)
        if level_match:
            level_text = level_match.group(1).strip()
            ability["ability_type"] = "passive"
            ability["level_ability"] = True
            # Level text may itself be an activated ability (e.g. "[T]: [Add] [1]")
            l_cost, l_effect = parse_activated_cost(level_text)
            if l_cost is not None:
                ability["ability_type"] = "activated"
                ability["cost"] = l_cost
                ability["timing"] = "default"
                ir = parse_effect_text(l_effect, card_name)
            else:
                # Level text may be a trigger
                l_trigger = None
                l_remaining = level_text
                for l_pat, l_trig in TRIGGER_COMPILED:
                    l_m = l_pat.match(level_text)
                    if l_m:
                        l_trigger = l_trig
                        l_remaining = level_text[l_m.end():].strip()
                        break
                if l_trigger:
                    ability["ability_type"] = "triggered"
                    ability["trigger_condition"] = l_trigger
                    ir = parse_effect_text(l_remaining, card_name)
                else:
                    ir = parse_effect_text(level_text, card_name)
            if ir:
                ability["effect_ir"] = ir
            abilities.append(ability)
            continue

        # Check for em-dash prefix (deathknell abilities): "— EFFECT"
        emdash_match = re.match(r"[\u2014\u2013\-]+\s*(.*)", block)
        if emdash_match:
            emdash_text = emdash_match.group(1).strip()
            ability["ability_type"] = "triggered"
            ability["trigger_condition"] = "on_death"  # deathknell default
            # The em-dash text might itself contain a trigger
            dk_trigger = None
            dk_remaining = emdash_text
            for dk_pat, dk_trig in TRIGGER_COMPILED:
                dk_m = dk_pat.match(emdash_text)
                if dk_m:
                    dk_trigger = dk_trig
                    dk_remaining = emdash_text[dk_m.end():].strip()
                    break
            if dk_trigger:
                ability["trigger_condition"] = dk_trigger
            ir = parse_effect_text(dk_remaining, card_name)
            if ir:
                ability["effect_ir"] = ir
            abilities.append(ability)
            continue

        # Check for [Repeat] spell pattern: [Repeat] [N] Effect or [Repeat] [N][C] Effect
        repeat_match = re.match(
            r"\[Repeat\]\s*(?:[\u2014\u2013\-]+\s*)?(?:\[[\w]+\]\s*(?:/\s*\[[\w]+\]\s*)*)*\s*(.*)",
            block,
        )
        if repeat_match:
            repeat_effect = repeat_match.group(1).strip()
            # Strip any remaining leading em-dash or "Discard N" cost text before effect
            repeat_effect = re.sub(r"^[\u2014\u2013\-]+\s*", "", repeat_effect).strip()
            ability["ability_type"] = "activated"
            ability["repeat"] = True
            if any(k["keyword"] == "reaction" for k in keywords):
                ability["timing"] = "reaction"
            elif any(k["keyword"] == "action" for k in keywords):
                ability["timing"] = "action"
            else:
                ability["timing"] = "default"
            ir = parse_effect_text(repeat_effect, card_name)
            if ir:
                ability["effect_ir"] = ir
            _annotate_targets(ability, ir)
            abilities.append(ability)
            continue

        # Check for passive ("While...", static text without trigger/cost)
        # Expanded to cover more passive patterns
        passive_starts = (
            "while ", "your ", "other ", "my ", "friendly ", "enemy ",
            "units ", "once ", "the first ", "opponents ",
            "i cost ", "i can't ", "i can be ", "i don't ", "i must ",
            "i have ", "i get ", "each equipment",
        )
        if any(block.lower().startswith(p) for p in passive_starts):
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
                "effect_ir": {"type": "add_energy", "amount": 1},
                "timing": "reaction",
                "text": "Add 1 Energy",
            },
            {
                "ability_id": f"{card_id}_recycle",
                "ability_type": "activated",
                "cost": {},
                "effect_ir": {"type": "add_power", "amount": 1, "domain": domain},
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
        if has_ir:
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
