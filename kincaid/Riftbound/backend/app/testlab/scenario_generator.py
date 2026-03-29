"""Auto-generate testlab scenarios from the card database."""

from __future__ import annotations

import logging
from collections import defaultdict

from ..engine.card_db import CardDB
from ..engine.card_types import CardDefinition
from ..engine.enums import CardType, Keyword
from .scenarios import ScenarioDef

logger = logging.getLogger("riftbound.testlab.gen")

BATTLEFIELD_IDS = ["unl-205-219", "unl-206-219", "ogn-275-298"]

def _get_buff_spells(card_db: dict[str, CardDefinition]) -> list[str]:
    """Find spells that buff units, useful for testing mighty threshold."""
    results: list[str] = []
    for c in card_db.values():
        if c.card_type != CardType.SPELL:
            continue
        for ab in c.abilities:
            if ab.effect_ir and ab.effect_ir.get("type") == "buff":
                results.append(c.card_id)
                break
    return results[:4]


def _get_fodder_units(card_db: dict[str, CardDefinition], count: int = 6) -> list[str]:
    """Pick cheap named units (not tokens) to use as recognizable targets."""
    units = [
        c for c in card_db.values()
        if c.card_type == CardType.UNIT
        and 1 <= c.cost_energy <= 3
        and not c.card_id.startswith("unl-t")  # exclude tokens
    ]
    units.sort(key=lambda c: (c.cost_energy, c.card_id))
    return [u.card_id for u in units[:count]]


def _get_kill_spells(card_db: dict[str, CardDefinition]) -> list[str]:
    """Find spells that deal damage or kill, useful for triggering deathknell."""
    results: list[str] = []
    for c in card_db.values():
        if c.card_type != CardType.SPELL:
            continue
        for ab in c.abilities:
            if ab.effect_ir and ab.effect_ir.get("type") in ("deal_damage", "kill"):
                results.append(c.card_id)
                break
    return results[:4]


def _has_effect_ir(card: CardDefinition) -> bool:
    return any(ab.effect_ir for ab in card.abilities)


def _primary_ir_type(card: CardDefinition) -> str | None:
    for ab in card.abilities:
        if ab.effect_ir:
            return ab.effect_ir.get("type")
    return None


def _ir_needs_enemy_targets(ir_type: str) -> bool:
    return ir_type in ("deal_damage", "kill", "stun", "bounce", "banish", "exhaust")


def _ir_needs_friendly_targets(ir_type: str) -> bool:
    return ir_type in ("buff", "heal", "draw_cards", "grant_keyword", "move")


def generate_all_scenarios() -> list[ScenarioDef]:
    """Generate all testlab scenarios from the loaded card database."""
    try:
        card_db = CardDB.all_cards()
    except Exception:
        logger.warning("CardDB not loaded, returning empty scenario list")
        return []

    if not card_db:
        return []

    scenarios: list[ScenarioDef] = []
    fodder = _get_fodder_units(card_db)
    kill_spells = _get_kill_spells(card_db)

    scenarios.extend(_generate_keyword_scenarios(card_db, fodder, kill_spells))
    scenarios.extend(_generate_ir_type_scenarios(card_db, fodder))
    scenarios.extend(_generate_card_type_scenarios(card_db, fodder))
    scenarios.extend(_generate_individual_scenarios(card_db, fodder))

    logger.info("Generated %d testlab scenarios", len(scenarios))
    return scenarios


def _find_non_action_spell(card_db: dict[str, CardDefinition]) -> str | None:
    """Find a cheap non-Action, non-Reaction spell for negative comparison tests."""
    for c in sorted(card_db.values(), key=lambda x: x.cost_energy):
        if (c.card_type == CardType.SPELL
                and not c.has_keyword(Keyword.ACTION)
                and not c.has_keyword(Keyword.REACTION)
                and c.cost_energy <= 3):
            return c.card_id
    return None


# ---------------------------------------------------------------------------
# Board-state placement rules:
#
#   VALID resting states:
#     - P1 units on BF0 only  → P1 controls BF0
#     - P2 units on BF1 only  → P2 controls BF1
#     - No units               → uncontrolled
#
#   INVALID resting state (triggers showdown):
#     - Both P1 and P2 units on the SAME battlefield → contested
#
#   Exception: ambush/reaction scenarios INTENTIONALLY create contested
#   state so the bot triggers showdowns, giving P1 reaction windows.
#
# For combat testing: P1 units start on BF0, enemy on BF1. The player
# moves their units to BF1 to trigger contested → showdown → combat.
# ---------------------------------------------------------------------------


def _generate_keyword_scenarios(
    card_db: dict[str, CardDefinition],
    fodder: list[str],
    kill_spells: list[str],
) -> list[ScenarioDef]:
    by_kw: dict[str, list[CardDefinition]] = defaultdict(list)
    for c in card_db.values():
        for ki in c.keywords:
            by_kw[ki.keyword.value].append(c)

    buff_spells = _get_buff_spells(card_db)
    scenarios: list[ScenarioDef] = []

    for kw_val, cards in sorted(by_kw.items()):
        if len(cards) < 3:
            continue

        units = [c for c in cards if c.card_type == CardType.UNIT]
        spells = [c for c in cards if c.card_type == CardType.SPELL]
        gear = [c for c in cards if c.card_type == CardType.GEAR]
        hand_ids: list[str] = []
        p1_bf: dict[int, list[str]] = {}
        p2_bf: dict[int, list[str]] = {}
        behavior = ""

        # ----- Hand-only mechanics (play from hand, observe result) -----

        if kw_val == "accelerate":
            # Just units in hand + resources. Play and choose to pay or skip.
            hand_ids = [c.card_id for c in units[:5]]
            behavior = (
                "Play these units from hand. You'll be prompted to pay the "
                "Accelerate cost (1E + 1 domain Power). Choose 'Accelerate' on "
                "some and 'Skip' on others. Verify accelerated units enter READY "
                "and skipped units enter EXHAUSTED."
            )

        elif kw_val == "action":
            # Action grants Showdown play (Rule 732). Need a contested BF
            # so the bot starts a showdown, giving P1 a window to test
            # that Action cards CAN be played during showdowns.
            # Pick Action-only spells with simple unit targets (no Hidden/Reaction).
            clean_action = [
                c for c in spells
                if not c.has_keyword(Keyword.HIDDEN)
                and not c.has_keyword(Keyword.REACTION)
                and "unit" in c.text.lower()
            ]
            hand_ids = [c.card_id for c in clean_action[:4]]
            # Add a non-Action spell so P1 can verify it gets rejected.
            non_action_spell = _find_non_action_spell(card_db)
            if non_action_spell:
                hand_ids.append(non_action_spell)
            p1_bf[0] = fodder[:2]   # \
            p2_bf[0] = fodder[2:4]  # / contested BF0 — bot starts showdown
            behavior = (
                "End your turn so the bot starts a showdown at the contested "
                "battlefield. During the showdown, play Action cards — they "
                "SHOULD work (Rule 732: Action grants showdown play on any "
                "player's turn). Then try the non-Action spell — it should be "
                "REJECTED (Rule 308.1.a: only Action/Reaction cards can be "
                "played in Showdown State). Note: Action is additive (732.2) "
                "— these cards also work during your normal Action phase."
            )

        elif kw_val == "hidden":
            # Hidden units in hand, empty battlefields to deploy facedown.
            hand_ids = [c.card_id for c in units[:5]]
            behavior = (
                "Play these units to an empty battlefield. They should enter "
                "facedown (hidden). Verify reveal timing when they would be "
                "involved in combat or targeted."
            )

        elif kw_val == "predict":
            # Predict cards in hand. No targets needed — manipulates deck.
            hand_ids = [c.card_id for c in cards[:5]]
            behavior = (
                "Play predict cards. When each resolves, you should see the top "
                "N cards of your deck (N = predict value) and choose which to "
                "recycle (move to bottom) or keep on top."
            )

        elif kw_val == "temporary":
            # Temporary units in hand. Play them, end turn, verify removal.
            hand_ids = [c.card_id for c in units[:5]]
            behavior = (
                "Play temporary units to your base or a battlefield. End your "
                "turn and pass through to your next Beginning phase. Verify "
                "temporary units that did NOT enter this turn are removed."
            )

        elif kw_val == "unique":
            # Two copies of each unique card in hand. Play first, try second.
            unique_ids: list[str] = []
            for c in cards:
                unique_ids.append(c.card_id)
                unique_ids.append(c.card_id)  # duplicate
                if len(unique_ids) >= 6:
                    break
            hand_ids = unique_ids
            behavior = (
                "Play the first copy of a unique card. Then try to play the "
                "second copy with the same card_id. The second should be "
                "REJECTED — only one copy of a Unique card can be in play."
            )

        elif kw_val == "vision":
            # Vision units in hand. Play to see top of deck.
            hand_ids = [c.card_id for c in units[:5]]
            behavior = (
                "Play vision units. When each enters play, you should see the "
                "top card of your deck revealed. You may choose to recycle it "
                "(move to bottom) or leave it on top."
            )

        elif kw_val == "legion":
            # Multiple legion cards in hand. Play a non-legion card first to
            # satisfy the "played another Main Deck card this turn" condition.
            filler = fodder[0] if fodder else None
            legion_ids = [c.card_id for c in cards[:4]]
            hand_ids = ([filler] if filler else []) + legion_ids
            behavior = (
                "Play the non-legion unit first (this satisfies the Legion "
                "condition: 'you've played another Main Deck card this turn'). "
                "Then play legion cards — their Legion bonus should now be active."
            )

        # ----- On-board mechanics (need friendly units to interact with) -----

        elif kw_val == "mighty":
            # Mighty units on BF (near threshold) + buff spells in hand.
            # Units with <5 might need buffs; units with 5+ are already mighty.
            sub5 = [c for c in units if (c.base_might or 0) < 5][:2]
            gte5 = [c for c in units if (c.base_might or 0) >= 5][:2]
            p1_bf[0] = [c.card_id for c in sub5 + gte5]
            hand_ids = buff_spells[:3]
            behavior = (
                "Some units on BF0 are already Mighty (5+ might), others are "
                "below the threshold. Cast buff spells on the sub-5 units to "
                "push them to 5+ and verify they gain Mighty status."
            )

        elif kw_val == "level":
            # Level cards in hand. Some may be units that level up, others
            # spells/gear that grant XP or trigger on level thresholds.
            hand_ids = [c.card_id for c in cards[:5]]
            p1_bf[0] = fodder[:2]  # friendly units to receive XP/buffs
            behavior = (
                "Play level cards. Units with Level gain XP from various "
                "sources. Verify XP accumulates and level-up effects trigger "
                "when thresholds are reached."
            )

        elif kw_val == "deathknell":
            # Deathknell units on BF + kill spells in hand to destroy them.
            p1_bf[0] = [c.card_id for c in units[:3]]
            hand_ids = kill_spells[:3]
            behavior = (
                "Use the kill/damage spells in your hand to destroy the "
                "deathknell units on your battlefield. Verify their death "
                "triggers fire and produce the expected effect."
            )

        elif kw_val == "equip":
            # Gear in hand, friendly units on BF to equip to.
            hand_ids = [c.card_id for c in (gear + spells)[:4]]
            p1_bf[0] = fodder[:3]
            behavior = (
                "Play gear cards targeting your units on BF0. Verify they "
                "attach correctly and grant their stat bonus or ability."
            )

        elif kw_val == "weaponmaster":
            # Weaponmaster units on BF + gear in hand to equip.
            p1_bf[0] = [c.card_id for c in units[:3]]
            hand_ids = [c.card_id for c in gear[:3]]
            behavior = (
                "Equip gear from hand onto the weaponmaster units on BF0. "
                "Verify the Weaponmaster bonus activates when gear is attached."
            )

        # ----- Movement/arrival mechanics -----

        elif kw_val == "ganking":
            # Ganking units on BF0, enemies on BF1. Ganking allows BF-to-BF movement.
            p1_bf[0] = [c.card_id for c in units[:3]]
            p2_bf[1] = fodder[:3]
            behavior = (
                "Move ganking units from BF0 to the enemy BF1. Ganking allows "
                "battlefield-to-battlefield movement (normally restricted). "
                "Verify the move is permitted and any arrival effects trigger."
            )

        elif kw_val == "hunt":
            # Hunt units on BF0, enemies on BF1. Hunt: must attack + can attack adjacent.
            p1_bf[0] = [c.card_id for c in units[:3]]
            p2_bf[1] = fodder[:3]
            behavior = (
                "Hunt units MUST attack if able and CAN attack units on adjacent "
                "battlefields. Move your hunt units to trigger combat and verify "
                "they are forced to attack and can reach adjacent BF targets."
            )

        # ----- Combat keywords (P1 on BF0, P2 on BF1, move to trigger) -----

        elif kw_val == "assault":
            p1_bf[0] = [c.card_id for c in units[:4]]
            p2_bf[1] = fodder[:3]
            behavior = (
                "Move your assault units from BF0 to the enemy BF1. This "
                "triggers combat. Verify assault units gain +N Might as "
                "attackers (N defaults to 1 if not specified)."
            )

        elif kw_val == "backline":
            # Mix backline + normal units so player can compare attacker roles.
            p1_bf[0] = [c.card_id for c in units[:3]] + fodder[:1]
            p2_bf[1] = fodder[:3]
            behavior = (
                "Move all units from BF0 to enemy BF1. In combat, verify "
                "backline units are NOT assigned as attackers. The non-backline "
                "unit should attack normally. Backline units CAN still defend."
            )

        elif kw_val == "deflect":
            p1_bf[0] = [c.card_id for c in units[:4]]
            p2_bf[1] = fodder[:3]
            behavior = (
                "Move deflect units from BF0 to enemy BF1. In combat, verify "
                "incoming damage to deflect units is reduced by the deflect "
                "value. Compare damage taken vs a unit without deflect."
            )

        elif kw_val == "quick_draw":
            # Only 1 QD unit (Jax) — put on BF0 with QD gear in hand to equip.
            p1_bf[0] = [c.card_id for c in units[:3]] + fodder[:1]
            hand_ids = [c.card_id for c in gear[:3]]
            p2_bf[1] = fodder[:3]
            behavior = (
                "Equip quick-draw gear onto units, then move from BF0 to enemy "
                "BF1. In combat, verify quick-draw units deal damage BEFORE "
                "normal units. If they kill an enemy, it shouldn't hit back."
            )

        elif kw_val == "shield":
            p1_bf[0] = [c.card_id for c in units[:4]]
            p2_bf[1] = fodder[:3]
            behavior = (
                "Move shield units from BF0 to enemy BF1 to trigger combat. "
                "Verify shield units gain +N Might while defending (N defaults "
                "to 1). Compare their combat might vs their base stat."
            )

        elif kw_val == "tank":
            p1_bf[0] = [c.card_id for c in units[:3]] + fodder[:1]
            p2_bf[1] = fodder[:3]
            behavior = (
                "Move units from BF0 to enemy BF1. In combat, verify tank "
                "units absorb damage before non-tank allies. The non-tank "
                "unit should only take damage after tanks are defeated."
            )

        # ----- Reaction-window mechanics (contested BF, bot triggers showdown) -----

        elif kw_val == "ambush":
            hand_ids = [c.card_id for c in units[:5]]
            p1_bf[0] = fodder[:2]   # \
            p2_bf[0] = fodder[2:4]  # / contested BF0 — bot starts showdown
            behavior = (
                "End your turn to let the bot act. The bot will start a "
                "showdown at the contested BF0. During the showdown window, "
                "play ambush units as reactions — Ambush grants Reaction timing."
            )

        elif kw_val == "reaction":
            hand_ids = [c.card_id for c in spells[:5]]
            p1_bf[0] = fodder[:2]   # \
            p2_bf[0] = fodder[2:4]  # / contested BF0 — bot starts showdown
            behavior = (
                "End your turn to let the bot act. The bot will start a "
                "showdown at the contested BF0. During the showdown window, "
                "play reaction spells from your hand."
            )

        else:
            # Fallback for any new keywords — hand cards only, no board clutter.
            hand_ids = [c.card_id for c in cards[:5]]
            behavior = (
                f"Play cards with the {kw_val} keyword and verify the mechanic "
                f"works as expected."
            )

        scenarios.append(ScenarioDef(
            scenario_id=f"keyword_{kw_val}",
            name=f"{kw_val.replace('_', ' ').title()} Cards",
            description=f"Test all cards with the {kw_val} keyword ({len(cards)} cards).",
            category="keyword",
            tags=[kw_val, "keyword"],
            expected_behavior=behavior,
            p1_hand=hand_ids,
            p1_base_units=[],
            p1_bf_units={k: v for k, v in p1_bf.items() if v},
            p2_bf_units={k: v for k, v in p2_bf.items() if v},
        ))

    return scenarios


# ---------------------------------------------------------------------------
# By IR type
# ---------------------------------------------------------------------------

def _generate_ir_type_scenarios(
    card_db: dict[str, CardDefinition],
    fodder: list[str],
) -> list[ScenarioDef]:
    by_ir: dict[str, list[CardDefinition]] = defaultdict(list)
    for c in card_db.values():
        ir_type = _primary_ir_type(c)
        if ir_type:
            by_ir[ir_type].append(c)

    scenarios: list[ScenarioDef] = []
    for ir_type, cards in sorted(by_ir.items()):
        if len(cards) < 3:
            continue

        hand_ids = [c.card_id for c in cards[:6]]
        p1_bf: dict[int, list[str]] = {}
        p2_bf: dict[int, list[str]] = {}

        # Enemy targets on BF1 (separate from P1), friendly on BF0
        if _ir_needs_enemy_targets(ir_type):
            p2_bf[1] = fodder[:3]                         # P2 controls BF1
        if _ir_needs_friendly_targets(ir_type):
            p1_bf[0] = fodder[:3]                         # P1 controls BF0

        scenarios.append(ScenarioDef(
            scenario_id=f"ir_{ir_type}",
            name=f"Effect: {ir_type.replace('_', ' ').title()}",
            description=f"Test all cards with {ir_type} effects ({len(cards)} cards).",
            category="ir_type",
            tags=[ir_type, "effect"],
            expected_behavior=f"Play these cards and verify the {ir_type} effect resolves correctly.",
            p1_hand=hand_ids,
            p1_base_units=[],
            p1_bf_units={k: v for k, v in p1_bf.items() if v},
            p2_bf_units={k: v for k, v in p2_bf.items() if v},
        ))

    return scenarios


# ---------------------------------------------------------------------------
# By card type
# ---------------------------------------------------------------------------

def _generate_card_type_scenarios(
    card_db: dict[str, CardDefinition],
    fodder: list[str],
) -> list[ScenarioDef]:
    scenarios: list[ScenarioDef] = []

    # Gear cards
    gear_cards = [c for c in card_db.values() if c.card_type == CardType.GEAR]
    if gear_cards:
        scenarios.append(ScenarioDef(
            scenario_id="type_gear",
            name="All Gear Cards",
            description=f"Test gear attachment mechanics ({len(gear_cards)} cards).",
            category="card_type",
            tags=["gear", "card_type"],
            expected_behavior="Play gear cards onto your units at the battlefield. Verify they attach correctly and grant their bonus.",
            p1_hand=[c.card_id for c in gear_cards[:6]],
            p1_base_units=[],
            p1_bf_units={0: fodder[:3]},                  # P1 controls BF0
            p2_bf_units={},
        ))

    # Action spells
    action_spells = [
        c for c in card_db.values()
        if c.card_type == CardType.SPELL and not c.has_keyword(Keyword.REACTION)
    ]
    if action_spells:
        scenarios.append(ScenarioDef(
            scenario_id="type_action_spell",
            name="Action Spells",
            description=f"Test action-speed spell cards ({len(action_spells)} cards).",
            category="card_type",
            tags=["spell", "action", "card_type"],
            expected_behavior="Play action spells during your Action phase. Verify effects resolve and cards go to trash.",
            p1_hand=[c.card_id for c in action_spells[:6]],
            p1_base_units=[],
            p1_bf_units={0: fodder[:2]},                  # P1 controls BF0
            p2_bf_units={1: fodder[:3]},                  # P2 controls BF1
        ))

    # Reaction spells — intentionally contested for showdown windows
    reaction_spells = [
        c for c in card_db.values()
        if c.card_type == CardType.SPELL and c.has_keyword(Keyword.REACTION)
    ]
    if reaction_spells:
        scenarios.append(ScenarioDef(
            scenario_id="type_reaction_spell",
            name="Reaction Spells",
            description=f"Test reaction-speed spell cards ({len(reaction_spells)} cards).",
            category="card_type",
            tags=["spell", "reaction", "card_type"],
            expected_behavior="End your turn so the bot triggers a showdown. Play reaction spells during the showdown window.",
            p1_hand=[c.card_id for c in reaction_spells[:6]],
            p1_base_units=[],
            p1_bf_units={0: fodder[:2]},                  # \
            p2_bf_units={0: fodder[2:4]},                 # / contested BF0!
        ))

    # Legends
    legends = [c for c in card_db.values() if c.card_type == CardType.LEGEND]
    if legends:
        scenarios.append(ScenarioDef(
            scenario_id="type_legend",
            name="Legend Cards",
            description=f"Test legend abilities ({len(legends)} cards).",
            category="card_type",
            tags=["legend", "card_type"],
            expected_behavior="Verify legend abilities activate from the legend zone. Legends cannot be killed or moved.",
            p1_hand=[],
            p1_base_units=[],
            p1_bf_units={0: fodder[:3]},                  # P1 controls BF0
            p2_bf_units={1: fodder[:3]},                  # P2 controls BF1
        ))

    # Battlefields
    bfs = [c for c in card_db.values() if c.card_type == CardType.BATTLEFIELD]
    if bfs:
        scenarios.append(ScenarioDef(
            scenario_id="type_battlefield",
            name="Battlefield Cards",
            description=f"Test battlefield effects ({len(bfs)} cards).",
            category="card_type",
            tags=["battlefield", "card_type"],
            expected_behavior="Play units to battlefields and verify passive effects apply. Check control mechanics.",
            p1_hand=fodder[:4],
            p1_base_units=[],
            p1_bf_units={},
            p2_bf_units={},
        ))

    return scenarios


# ---------------------------------------------------------------------------
# Individual card scenarios
# ---------------------------------------------------------------------------

def _generate_individual_scenarios(
    card_db: dict[str, CardDefinition],
    fodder: list[str],
) -> list[ScenarioDef]:
    scenarios: list[ScenarioDef] = []

    for card in sorted(card_db.values(), key=lambda c: c.card_id):
        if not _has_effect_ir(card):
            continue

        ir_type = _primary_ir_type(card) or "unknown"
        hand_ids: list[str] = []
        p1_bf: dict[int, list[str]] = {}
        p2_bf: dict[int, list[str]] = {}

        if card.card_type == CardType.UNIT:
            hand_ids = [card.card_id]
            if _ir_needs_enemy_targets(ir_type):
                p2_bf[1] = fodder[:3]                     # P2 controls BF1
            if _ir_needs_friendly_targets(ir_type):
                p1_bf[0] = fodder[:2]                     # P1 controls BF0
        elif card.card_type == CardType.SPELL:
            hand_ids = [card.card_id]
            if _ir_needs_enemy_targets(ir_type):
                p2_bf[1] = fodder[:3]                     # P2 controls BF1
            if _ir_needs_friendly_targets(ir_type):
                p1_bf[0] = fodder[:2]                     # P1 controls BF0
        elif card.card_type == CardType.GEAR:
            hand_ids = [card.card_id]
            p1_bf[0] = fodder[:2]                         # friendly units to equip
        else:
            hand_ids = [card.card_id]
            p1_bf[0] = fodder[:2]                         # P1 controls BF0
            p2_bf[1] = fodder[:2]                         # P2 controls BF1

        kw_tags = [ki.keyword.value for ki in card.keywords]
        tags = [card.card_type.value, ir_type] + kw_tags

        text_preview = card.text[:80] + "..." if len(card.text) > 80 else card.text
        scenarios.append(ScenarioDef(
            scenario_id=f"card_{card.card_id}",
            name=card.name,
            description=f"[{card.card_type.value}] {text_preview}",
            category="individual",
            tags=tags,
            expected_behavior=f"Play {card.name} and verify: {text_preview}",
            p1_hand=hand_ids,
            p1_base_units=[],
            p1_bf_units={k: v for k, v in p1_bf.items() if v},
            p2_bf_units={k: v for k, v in p2_bf.items() if v},
        ))

    return scenarios
