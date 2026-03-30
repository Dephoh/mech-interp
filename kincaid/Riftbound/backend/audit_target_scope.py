"""
Audit: detect cards where effect_ir target scope doesn't match the card's ability text.

Mismatch rules:
  1. Text says "a unit" (no friend/enemy qualifier) but scope is "friendly"
     -- should be "any" for unqualified targets
  2. Text says "an enemy unit" but scope is "friendly"
  3. Text says "a friendly unit" / "unit you control" but scope is "enemy"
  4. Text says "Deal X to a unit" (damage, no qualifier) but scope is "friendly"
     -- damage spells that say "a unit" can hit anything

Key nuance:
  - "your trash", "your runes", "your base", "your Mechs" etc. -> friendly (correct)
  - "a unit from your trash" -> the pronoun "your" modifies "trash", not "unit",
    but the unit IS yours (it's in your trash), so friendly is correct
  - Self-referencing text like "When I attack" uses "I" for the card itself, not the
    target -- must not confuse trigger context with target scope
  - Multi-step abilities may target both friendly and enemy -- analyze per-target
"""

import json
import re
from pathlib import Path
from typing import Any

CARD_FILE = Path(__file__).parent / "data" / "card_definitions.json"


# ---------------------------------------------------------------------------
# IR tree walking
# ---------------------------------------------------------------------------

def walk_ir(node: Any, path: str = "root"):
    """Yield (path, node) for every dict node in an IR tree."""
    if isinstance(node, dict):
        yield (path, node)
        for key, value in node.items():
            if isinstance(value, dict):
                yield from walk_ir(value, f"{path}.{key}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    yield from walk_ir(item, f"{path}.{key}[{i}]")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from walk_ir(item, f"{path}[{i}]")


def collect_targets(ir: dict) -> list[tuple[str, str, str, dict]]:
    """
    Return list of (ir_path, ir_node_type, scope, target_dict)
    for every node that has a target with a scope.
    """
    results = []
    for ir_path, node in walk_ir(ir):
        if "target" in node and isinstance(node["target"], dict):
            target = node["target"]
            if "scope" in target:
                ir_type = node.get("type", "unknown")
                results.append((ir_path, ir_type, target["scope"], target))
    return results


# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------

# Phrases that mean "your own stuff" -- friendly scope is CORRECT here
FRIENDLY_CONTEXT_RE = re.compile(
    r"(?:from\s+)?your\s+(?:trash|hand|deck|base|main\s+deck|rune|runes|"
    r"supply|side|domain|domains|battlefield|battlefields|legend|token|tokens|"
    r"mech|mechs|sand\s+soldier|sand\s+soldiers)",
    re.IGNORECASE,
)

# "unit you/they control" -- friendly context
CONTROL_RE = re.compile(
    r"(?:unit|units|gear)\s+(?:you|they)\s+control",
    re.IGNORECASE,
)

# Possessive "your X have" / "your X enter" -- modifiers on own units
YOUR_UNITS_MODIFIER_RE = re.compile(
    r"your\s+(?:\w+\s+)?(?:units?|tokens?|mechs?|sand\s+soldiers?)\s+(?:have|get|gain|enter|cost)",
    re.IGNORECASE,
)

# Explicit "a friendly unit" / "another friendly unit" / "other friendly units"
EXPLICIT_FRIENDLY_UNIT_RE = re.compile(
    r"(?:a|an|another|other)\s+friendly\s+(?:units?|gear|spells?)",
    re.IGNORECASE,
)

# Self-reference: "I", "me", "my" -- only relevant for the unit card itself
SELF_REF_RE = re.compile(
    r"\b(?:(?:buff|ready|play|move|return)\s+)?(?:me|myself)\b|"
    r"\bwhen\s+I\b|\bif\s+I\b|\bgive\s+me\b|\bI\s+(?:have|get|gain|cost|enter|attack|defend|hold|conquer|die|leave)\b",
    re.IGNORECASE,
)

# Enemy qualifiers
ENEMY_UNIT_RE = re.compile(
    r"(?:an?\s+)?enemy\s+(?:unit|gear|spell|units|creatures?)",
    re.IGNORECASE,
)
OPPONENT_RE = re.compile(
    r"(?:an?\s+)?opponent'?s?\s+(?:unit|gear|card|spell|hand)",
    re.IGNORECASE,
)
# "choose an opponent. They ..." -> the next action targets enemy stuff
CHOOSE_OPPONENT_RE = re.compile(
    r"choose\s+an\s+opponent",
    re.IGNORECASE,
)

# Unqualified "a unit" patterns (no qualifier)
# These match: "a unit", "a unit at a battlefield", "a unit here"
# Must NOT be preceded by enemy/friendly/your/their
UNQUALIFIED_UNIT_RE = re.compile(
    r"(?<!\w)(?:kill|stun|return|move|bounce|give|buff|ready|choose)\s+"
    r"(?:a|an|another)\s+unit\b"
    r"(?!\s+(?:you|they|it)\s+control)"
    r"(?!\s+(?:from\s+)?(?:your|their))",
    re.IGNORECASE,
)

# Damage to unqualified unit
DAMAGE_UNQUALIFIED_RE = re.compile(
    r"deal\s+\d+\s+(?:(?:bonus\s+)?damage\s+)?to\s+(?:a\s+)?unit\b"
    r"(?!\s+(?:you|they)\s+control)",
    re.IGNORECASE,
)

# "each player" patterns
EACH_PLAYER_RE = re.compile(
    r"each\s+player|starting\s+with\s+the\s+next\s+player",
    re.IGNORECASE,
)


def text_implies_friendly(text: str) -> bool:
    """Return True if the ability text clearly implies friendly-only targeting."""
    t = text.lower()
    if FRIENDLY_CONTEXT_RE.search(t):
        return True
    if CONTROL_RE.search(t):
        return True
    if YOUR_UNITS_MODIFIER_RE.search(t):
        return True
    if EXPLICIT_FRIENDLY_UNIT_RE.search(t):
        return True
    return False


def text_implies_enemy(text: str) -> bool:
    """Return True if the ability text implies enemy targeting."""
    t = text.lower()
    return bool(ENEMY_UNIT_RE.search(t) or OPPONENT_RE.search(t))


def text_has_choose_opponent(text: str) -> bool:
    return bool(CHOOSE_OPPONENT_RE.search(text))


def text_has_unqualified_unit_action(text: str) -> bool:
    """Check for 'kill/stun/give/etc a unit' without friendly/enemy qualifier."""
    return bool(UNQUALIFIED_UNIT_RE.search(text))


def text_has_damage_to_any_unit(text: str) -> bool:
    """Check for 'deal X to a unit' without friendly/enemy qualifier."""
    t = text.lower()
    for m in DAMAGE_UNQUALIFIED_RE.finditer(t):
        # Make sure 'enemy' or 'friendly' doesn't appear right before the match
        prefix_start = max(0, m.start() - 15)
        prefix = t[prefix_start:m.start()]
        if "enemy" not in prefix and "friendly" not in prefix:
            return True
    return False


def text_has_each_player(text: str) -> bool:
    return bool(EACH_PLAYER_RE.search(text))


# ---------------------------------------------------------------------------
# Categorize a single (ability_text, ir_target) pair
# ---------------------------------------------------------------------------

# IR node types that are inherently positive (you'd use on your own stuff)
POSITIVE_EFFECTS = {
    "give_might", "buff", "ready", "heal", "draw", "play_token",
    "give_keyword", "give_ability", "modify_cost", "return_to_hand",
    "play", "enter_ready",
}

# IR node types that are inherently negative (you'd use on enemy stuff)
NEGATIVE_EFFECTS = {
    "deal_damage", "damage", "kill", "destroy", "stun", "debuff",
    "give_negative_might", "discard", "banish", "exhaust",
    "return_to_hand",  # can be negative too (bounce)
}

# IR types that are ambiguous (could target either)
AMBIGUOUS_EFFECTS = {
    "move", "return_to_hand", "choose", "sequence", "optional",
    "conditional", "repeat",
}


def check_mismatch(
    card_name: str,
    card_id: str,
    card_type: str,
    ab_text: str,
    ir_path: str,
    ir_type: str,
    scope: str,
    target: dict,
) -> dict | None:
    """
    Check if a single target scope is wrong given the ability text.
    Returns a mismatch dict or None.
    """
    # Skip scopes that are inherently valid
    if scope in ("self", "each_player", "all", "none", "controller", "any"):
        return None

    # Determine what the text implies
    implies_friendly = text_implies_friendly(ab_text)
    implies_enemy = text_implies_enemy(ab_text)
    has_unqualified = text_has_unqualified_unit_action(ab_text)
    has_damage_any = text_has_damage_to_any_unit(ab_text)
    has_each_player = text_has_each_player(ab_text)
    has_choose_opp = text_has_choose_opponent(ab_text)

    # --- Filter out known-correct cases ---

    # If text explicitly says "your trash", "your runes" etc, friendly is fine
    if scope == "friendly" and implies_friendly and not has_unqualified and not has_damage_any:
        return None

    # "each player" -> scope should ideally be "each_player" but friendly
    # can be acceptable if the IR structures it per-player
    if has_each_player:
        return None

    # "choose an opponent. They ..." -> enemy scope on the next part is correct
    if scope == "enemy" and has_choose_opp:
        return None

    # Target is in a specific zone belonging to the controller (trash, hand, deck)
    target_zone = target.get("zone", "")
    if scope == "friendly" and target_zone in ("trash", "hand", "deck", "main_deck"):
        # "a unit from your trash" -> friendly is correct
        return None

    # Target has filters like is_token, is_mech -> often "your tokens" etc.
    target_filters = target.get("filters", [])
    if scope == "friendly" and target_filters and implies_friendly:
        return None

    # --- Check actual mismatches ---

    reason = None
    severity = "WARNING"

    # If text mentions BOTH friendly and enemy targets, both scopes are valid
    # (multi-target abilities like "Return a friendly unit and an enemy unit")
    if implies_friendly and implies_enemy:
        return None

    # RULE 2: text says "enemy" but scope is "friendly"
    if implies_enemy and not implies_friendly and scope == "friendly":
        reason = "Text says ENEMY but scope is 'friendly'"
        severity = "ERROR"

    # RULE 3: text says "friendly" / "your" but scope is "enemy"
    elif implies_friendly and not implies_enemy and scope == "enemy":
        # Exception: "choose an opponent" already handled above
        reason = "Text says FRIENDLY but scope is 'enemy'"
        severity = "ERROR"

    # RULE 4: damage to unqualified "a unit" with scope "friendly"
    elif has_damage_any and scope == "friendly" and ir_type in ("deal_damage", "damage"):
        reason = "DAMAGE to unqualified 'a unit' but scope is 'friendly' (should be 'any')"
        severity = "ERROR"

    # RULE 1: unqualified "kill/stun/give a unit" with scope "friendly"
    elif has_unqualified and scope == "friendly":
        # Check if text ALSO has a friendly qualifier that might apply
        if implies_friendly:
            # Mixed: text has both "your ..." and "a unit" -- could be two targets
            # Only flag if the unqualified part aligns with this specific node
            pass
        else:
            reason = "Unqualified 'a unit' action but scope is 'friendly' (should be 'any')"
            severity = "WARNING"

    # Damage to any unit -- also flag even if not caught by unqualified check
    elif has_damage_any and scope == "friendly":
        reason = "DAMAGE to unqualified 'a unit' but scope is 'friendly' (should be 'any')"
        severity = "ERROR"

    if not reason:
        return None

    return {
        "card_name": card_name,
        "card_id": card_id,
        "card_type": card_type,
        "ability_text": ab_text,
        "ir_path": ir_path,
        "ir_type": ir_type,
        "scope": scope,
        "target": target,
        "reason": reason,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def audit():
    with open(CARD_FILE, encoding="utf-8") as f:
        cards = json.load(f)

    mismatches = []
    abilities_with_ir = 0
    total_targets = 0
    skipped_ok = 0

    for card in cards:
        for ability in card.get("abilities", []):
            ir = ability.get("effect_ir")
            if not ir:
                continue
            abilities_with_ir += 1

            ab_text = ability.get("text", "") or card.get("text", "")

            targets = collect_targets(ir)
            for ir_path, ir_type, scope, target_dict in targets:
                total_targets += 1

                result = check_mismatch(
                    card_name=card["name"],
                    card_id=card["card_id"],
                    card_type=card.get("card_type", ""),
                    ab_text=ab_text,
                    ir_path=ir_path,
                    ir_type=ir_type,
                    scope=scope,
                    target=target_dict,
                )
                if result:
                    mismatches.append(result)
                else:
                    skipped_ok += 1

    # --- Report ---
    errors = [m for m in mismatches if m["severity"] == "ERROR"]
    warnings = [m for m in mismatches if m["severity"] == "WARNING"]

    # Deduplicate by card name for the summary
    unique_cards = sorted(set(m["card_name"] for m in mismatches))

    print("=" * 80)
    print("TARGET SCOPE AUDIT REPORT")
    print("=" * 80)
    print(f"Cards loaded:           {len(cards)}")
    print(f"Abilities with IR:      {abilities_with_ir}")
    print(f"Total target+scope:     {total_targets}")
    print(f"Passed (OK):            {skipped_ok}")
    print(f"Mismatches found:       {len(mismatches)}  "
          f"({len(errors)} ERROR, {len(warnings)} WARNING)")
    print(f"Unique cards affected:  {len(unique_cards)}")
    print("=" * 80)

    # Print ERRORs first, then WARNINGs
    for severity_label, group in [("ERROR", errors), ("WARNING", warnings)]:
        if not group:
            continue
        print(f"\n{'=' * 80}")
        print(f"  {severity_label}S  ({len(group)} findings)")
        print(f"{'=' * 80}")

        for i, m in enumerate(group, 1):
            print(f"\n{'-' * 70}")
            print(f"[{m['severity']}] #{i}: {m['card_name']}  ({m['card_id']})")
            print(f"  Reason:   {m['reason']}")
            print(f"  Text:     {m['ability_text'][:130]}")
            print(f"  IR type:  {m['ir_type']}")
            print(f"  IR path:  {m['ir_path']}")
            print(f"  Scope:    {m['scope']}  (should likely be 'any')")
            print(f"  Target:   {m['target']}")

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"AFFECTED CARDS ({len(unique_cards)}):")
    print(f"{'=' * 80}")
    for name in unique_cards:
        card_entries = [m for m in mismatches if m["card_name"] == name]
        severities = set(m["severity"] for m in card_entries)
        sev = "ERROR" if "ERROR" in severities else "WARNING"
        print(f"  [{sev:7s}] {name}")

    print(f"\n{'=' * 80}")
    print(f"TOTALS: {len(mismatches)} mismatches across {len(unique_cards)} cards")
    print(f"        {len(errors)} errors, {len(warnings)} warnings")
    print(f"        {skipped_ok} targets passed checks")
    print("=" * 80)


if __name__ == "__main__":
    audit()
