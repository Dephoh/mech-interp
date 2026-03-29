"""Keyword-specific logic that augments core engine behavior."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .card_types import CardInstance, KeywordInstance
from .enums import CardType, CombatRole, Domain, Keyword, ZoneType

if TYPE_CHECKING:
    from .game_state import GameState

logger = logging.getLogger("riftbound.keywords")


# ---------------------------------------------------------------------------
# Accelerate — unit enters ready if the Accelerate cost was paid (731)
# ---------------------------------------------------------------------------


def apply_accelerate(unit: CardInstance, paid: bool) -> None:
    """If Accelerate cost was paid, unit enters ready instead of exhausted."""
    if paid and unit.has_keyword(Keyword.ACCELERATE):
        unit.exhausted = False


def get_accelerate_cost(unit: CardInstance) -> tuple[int, dict[Domain, int]]:
    """Returns (energy, power_dict) needed to pay Accelerate."""
    energy = 1
    domains = unit.definition.domains
    if len(domains) == 1:
        power = {domains[0]: 1}
    else:
        # Multi-domain or no-domain: any power
        power = {}  # engine treats empty as "1 power of any domain"
    return energy, power


# ---------------------------------------------------------------------------
# Tank — must receive lethal damage before non-Tank units (741)
# ---------------------------------------------------------------------------


def check_tank_ordering(
    targets: list[CardInstance],
    assignment: dict[str, int],
) -> bool:
    """
    Validate that Tank units receive lethal before non-Tank units.
    Returns True if assignment is legal.
    """
    tank_units = [u for u in targets if u.has_keyword(Keyword.TANK)]
    non_tank_units = [u for u in targets if not u.has_keyword(Keyword.TANK)]

    # All Tank units must have lethal assigned before any non-Tank gets damage
    for tank in tank_units:
        lethal = tank.effective_might - tank.damage
        assigned = assignment.get(tank.instance_id, 0)
        if assigned < lethal:
            # Check if any non-tank unit got damage
            for nt in non_tank_units:
                if assignment.get(nt.instance_id, 0) > 0:
                    return False
    return True


def check_lethal_before_next(
    targets: list[CardInstance],
    assignment: dict[str, int],
) -> bool:
    """
    Validate that each unit receiving damage has lethal assigned
    before damage is assigned to the next.
    """
    units_with_damage = [
        u for u in targets if assignment.get(u.instance_id, 0) > 0
    ]
    if len(units_with_damage) <= 1:
        return True

    # The last unit doesn't need lethal (overflow)
    for unit in units_with_damage[:-1]:
        lethal = unit.effective_might - unit.damage
        assigned = assignment.get(unit.instance_id, 0)
        if assigned < lethal:
            return False
    return True


# ---------------------------------------------------------------------------
# Action / Reaction — timing permissions (732, 739)
# ---------------------------------------------------------------------------


def can_play_in_state(
    card_or_ability_keywords: list[KeywordInstance],
    is_showdown: bool,
    is_closed: bool,
) -> bool:
    """Check if a card/ability can be played given the current turn state."""
    has_action = any(k.keyword == Keyword.ACTION for k in card_or_ability_keywords)
    has_reaction = any(k.keyword == Keyword.REACTION for k in card_or_ability_keywords)

    if is_closed:
        return has_reaction
    if is_showdown:
        return has_action or has_reaction
    # Neutral open: always OK for the turn player
    return True


# ---------------------------------------------------------------------------
# Legion — conditional bonus if another Main Deck card was played this turn (738)
# ---------------------------------------------------------------------------


def process_legion(card: CardInstance, gs: GameState) -> bool:
    """Check if Legion condition is met (played another Main Deck card this turn)."""
    ps = gs.players[card.controller_id]
    # For spells/play effects: "played another card before this one"
    return ps.played_main_deck_this_turn > 0


# ---------------------------------------------------------------------------
# Deathknell — triggered ability that fires when the unit dies (734)
# ---------------------------------------------------------------------------


def process_deathknell(unit: CardInstance, gs: GameState) -> list[str]:
    """
    When a unit with Deathknell dies, create chain items for its effects.
    Returns list of log messages.
    """
    from .game_state import ChainItem
    logs: list[str] = []
    for ability in unit.definition.abilities:
        if ability.trigger_condition == "on_death" or unit.has_keyword(Keyword.DEATHKNELL):
            item = ChainItem.create(
                controller_id=unit.controller_id,
                source_instance_id=unit.instance_id,
                ability_id=ability.ability_id,
            )
            gs.chain.push(item)
            logs.append(f"{unit.name} Deathknell triggers")
            break  # one deathknell per unit for now
    return logs


# ---------------------------------------------------------------------------
# Temporary — unit dies at start of controller's Beginning Phase (742)
# ---------------------------------------------------------------------------


def should_die_temporary(unit: CardInstance) -> bool:
    """Check if a Temporary unit should be killed at start of Beginning Phase."""
    return unit.has_keyword(Keyword.TEMPORARY) and not unit.entered_this_turn


# ---------------------------------------------------------------------------
# Assault — +X might while unit is an attacker (733)
# ---------------------------------------------------------------------------
# NOTE: Assault is already handled in CardInstance.effective_might via
# keyword_value(Keyword.ASSAULT) when combat_role == ATTACKER.
# This helper is provided for explicit queries outside the might property.


def get_assault_bonus(unit: CardInstance) -> int:
    """Return the Assault might bonus (only applies while attacking)."""
    if unit.combat_role == CombatRole.ATTACKER:
        return unit.keyword_value(Keyword.ASSAULT)
    return 0


# ---------------------------------------------------------------------------
# Shield — +X might while unit is a defender (740)
# ---------------------------------------------------------------------------
# NOTE: Shield is already handled in CardInstance.effective_might via
# keyword_value(Keyword.SHIELD) when combat_role == DEFENDER.


def get_shield_bonus(unit: CardInstance) -> int:
    """Return the Shield might bonus (only applies while defending)."""
    if unit.combat_role == CombatRole.DEFENDER:
        return unit.keyword_value(Keyword.SHIELD)
    return 0


# ---------------------------------------------------------------------------
# Mighty — condition check: unit has might >= 5 (informal "mighty" trait)
# ---------------------------------------------------------------------------


def check_mighty(unit: CardInstance) -> bool:
    """Return True if the unit qualifies as Mighty (effective might >= 5)."""
    return unit.effective_might >= 5


# ---------------------------------------------------------------------------
# Deflect — spells/abilities targeting this unit cost extra power (735)
# ---------------------------------------------------------------------------


def get_deflect_cost(target: CardInstance) -> int:
    """
    Return the additional power cost imposed by Deflect on spells/abilities
    that choose this permanent. The power may be of any domain.
    """
    return target.keyword_value(Keyword.DEFLECT)


def check_deflect_payment(
    target: CardInstance,
    extra_power_paid: int,
) -> bool:
    """
    Validate that the caster paid enough extra power to overcome Deflect.
    Returns True if the payment is sufficient.
    """
    required = get_deflect_cost(target)
    return extra_power_paid >= required


# ---------------------------------------------------------------------------
# Quick-Draw — gear keyword: Reaction timing + auto-attach on play (745)
# ---------------------------------------------------------------------------


def has_quick_draw(card: CardInstance) -> bool:
    """
    Quick-Draw grants Reaction timing and auto-attaches on play.
    Returns True if the card has Quick-Draw.
    """
    return card.has_keyword(Keyword.QUICK_DRAW)


def apply_quick_draw_grants(card: CardInstance) -> None:
    """
    Grant Reaction keyword to a Quick-Draw card so it can be played
    during Closed States (rule 745.1.b / 745.1.c).
    Called during card pipeline when a Quick-Draw card enters hand.
    """
    if card.has_keyword(Keyword.QUICK_DRAW) and not card.has_keyword(Keyword.REACTION):
        card.granted_keywords.append(KeywordInstance(keyword=Keyword.REACTION))


# ---------------------------------------------------------------------------
# Hidden — card can be hidden facedown at a controlled battlefield (737)
# ---------------------------------------------------------------------------


def can_hide_card(card: CardInstance, gs: GameState) -> bool:
    """
    Check if a card with Hidden can be hidden facedown right now.
    Requires: card in hand or champion zone, it's the controller's turn,
    state is open, and the player controls at least one battlefield
    without a facedown card.
    """
    if not card.has_keyword(Keyword.HIDDEN):
        return False
    if card.zone not in (ZoneType.HAND, ZoneType.CHAMPION_ZONE):
        return False
    if gs.turn_player_id != card.controller_id:
        return False
    # Must have a controlled battlefield without a facedown card
    for bf in gs.battlefields.values():
        if bf.controller_id == card.controller_id and bf.facedown_card is None:
            return True
    return False


def hide_card_at_battlefield(
    card: CardInstance,
    battlefield_id: str,
    gs: GameState,
) -> list[str]:
    """
    Hide a card facedown at a battlefield the controller controls.
    Returns log messages.
    """
    bf = gs.battlefields.get(battlefield_id)
    if not bf:
        return [f"Invalid battlefield {battlefield_id}"]
    if bf.controller_id != card.controller_id:
        return ["Cannot hide at a battlefield you do not control"]
    if bf.facedown_card is not None:
        return ["Battlefield already has a facedown card"]

    # Remove from hand
    ps = gs.players[card.controller_id]
    if card.instance_id in ps.hand:
        ps.hand.remove(card.instance_id)

    # Place facedown
    card.zone = ZoneType.FACEDOWN_ZONE
    card.location_id = battlefield_id
    card.facedown = True
    card.hidden_at_battlefield = battlefield_id
    card.hidden_ready = False  # becomes ready next turn
    bf.facedown_card = card.instance_id

    logger.info("[HIDDEN] %s hidden at %s by %s", card.name, battlefield_id, card.controller_id)
    return [f"A card is hidden facedown at {battlefield_id}"]


def reveal_hidden_card(card: CardInstance, gs: GameState) -> list[str]:
    """
    Reveal a facedown hidden card. Used when combat starts at its battlefield
    or when the card is played from hidden.
    Returns log messages.
    """
    logs: list[str] = []
    bf_id = card.hidden_at_battlefield
    card.hidden_at_battlefield = None
    card.hidden_ready = False
    card.facedown = False

    if bf_id:
        bf = gs.battlefields.get(bf_id)
        if bf and bf.facedown_card == card.instance_id:
            bf.facedown_card = None

    logs.append(f"{card.name} is revealed from facedown")
    logger.info("[HIDDEN] %s revealed", card.name)
    return logs


def mark_hidden_ready(gs: GameState) -> list[str]:
    """
    At the start of a player's turn, mark their facedown cards as ready
    (they have survived one turn cycle facedown).
    """
    logs: list[str] = []
    for bf in gs.battlefields.values():
        if bf.facedown_card:
            card = gs.instances.get(bf.facedown_card)
            if (
                card
                and card.controller_id == gs.turn_player_id
                and card.hidden_at_battlefield
                and not card.hidden_ready
            ):
                card.hidden_ready = True
                # Grant Reaction timing (rule 737.6)
                if not card.has_keyword(Keyword.REACTION):
                    card.granted_keywords.append(
                        KeywordInstance(keyword=Keyword.REACTION)
                    )
                logs.append(f"Facedown card at {bf.battlefield_id} is now ready to play")
    return logs


# ---------------------------------------------------------------------------
# Ambush — when this unit enters the battlefield, deal might damage to a
#           target enemy unit (deployment trigger, not a runtime keyword func)
# ---------------------------------------------------------------------------


def check_ambush(unit: CardInstance) -> bool:
    """Return True if the unit has Ambush and should deal its might as damage on entry."""
    return unit.has_keyword(Keyword.AMBUSH)


def apply_ambush_damage(
    unit: CardInstance,
    target: CardInstance,
) -> list[str]:
    """
    Apply Ambush damage: the entering unit deals damage equal to its
    might to a target enemy unit.
    Returns log messages.
    """
    if not check_ambush(unit):
        return []
    dmg = unit.effective_might
    if dmg <= 0:
        return []
    target.damage += dmg
    logger.info("[AMBUSH] %s deals %d to %s", unit.name, dmg, target.name)
    return [f"{unit.name} Ambush deals {dmg} damage to {target.name}"]


# ---------------------------------------------------------------------------
# Backline — cannot be attacked unless it is the only unit on its side (741-ish)
# ---------------------------------------------------------------------------


def check_backline_protection(
    target: CardInstance,
    all_friendly_units: list[CardInstance],
) -> bool:
    """
    Return True if the target unit is protected by Backline
    (cannot be assigned damage unless it is the only friendly unit).
    """
    if not target.has_keyword(Keyword.BACKLINE):
        return False
    # Protected only if there are other friendly units
    non_backline = [
        u for u in all_friendly_units
        if u.instance_id != target.instance_id
    ]
    return len(non_backline) > 0


def get_valid_damage_targets(
    targets: list[CardInstance],
) -> list[CardInstance]:
    """
    Filter targets to exclude Backline-protected units.
    If ALL units have Backline, none are protected (all can be targeted).
    """
    non_backline = [u for u in targets if not u.has_keyword(Keyword.BACKLINE)]
    if non_backline:
        return non_backline
    # All have Backline — protection doesn't apply
    return list(targets)


# ---------------------------------------------------------------------------
# Ganking — unit may move from one battlefield to another (736)
# ---------------------------------------------------------------------------


def can_gank_move(unit: CardInstance) -> bool:
    """
    Return True if the unit has Ganking and may move from one battlefield
    to another (instead of only base-to-battlefield).
    """
    return unit.has_keyword(Keyword.GANKING)


# ---------------------------------------------------------------------------
# Hunt — unit must attack if able; can attack adjacent battlefields
# ---------------------------------------------------------------------------


def must_attack_hunt(unit: CardInstance) -> bool:
    """Return True if the unit has Hunt and must attack if able."""
    return unit.has_keyword(Keyword.HUNT)


def can_hunt_adjacent(unit: CardInstance) -> bool:
    """Return True if the unit can attack units in adjacent battlefields via Hunt."""
    return unit.has_keyword(Keyword.HUNT)


# ---------------------------------------------------------------------------
# Unique — only one copy of this card can be in play at a time
# ---------------------------------------------------------------------------


def check_unique_violation(card: CardInstance, gs: GameState) -> bool:
    """
    Return True if playing this card would violate the Unique constraint
    (another copy with the same card_id is already on the board under
    the same controller).
    """
    if not card.has_keyword(Keyword.UNIQUE):
        return False
    for inst in gs.instances.values():
        if (
            inst.instance_id != card.instance_id
            and inst.card_id == card.card_id
            and inst.controller_id == card.controller_id
            and inst.zone in (ZoneType.BASE, ZoneType.BATTLEFIELD)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Level — unit can gain XP and level up at thresholds
# ---------------------------------------------------------------------------
# NOTE: Level is a progression mechanic. The actual XP tracking and
# level-up thresholds are part of the card definition / effect IR.
# This provides the runtime check helpers.


def has_level(unit: CardInstance) -> bool:
    """Return True if the unit has the Level keyword."""
    return unit.has_keyword(Keyword.LEVEL)


# ---------------------------------------------------------------------------
# Predict — look at top N cards and reorder them when playing this card
# ---------------------------------------------------------------------------


def get_predict_count(card: CardInstance) -> int:
    """Return the number of cards to look at for Predict (the keyword value)."""
    return card.keyword_value(Keyword.PREDICT)


def apply_predict(
    card: CardInstance,
    gs: GameState,
    new_order: list[str],
) -> list[str]:
    """
    Apply Predict: reorder the top N cards of the controller's deck.
    ``new_order`` is the desired ordering of instance_ids (top first).
    Returns log messages.
    """
    n = get_predict_count(card)
    if n <= 0:
        return []
    ps = gs.players[card.controller_id]
    top_n = ps.main_deck[:n]
    if set(new_order) != set(top_n):
        return ["Invalid Predict reorder — cards don't match"]
    ps.main_deck[:n] = new_order
    logger.info("[PREDICT] %s reordered top %d cards", card.controller_id, n)
    return [f"{card.controller_id} reorders the top {n} cards (Predict)"]


# ---------------------------------------------------------------------------
# Repeat — spell may be executed again by paying its Repeat cost (746)
# ---------------------------------------------------------------------------


def get_repeat_cost(card: CardInstance) -> int:
    """
    Return the Repeat cost value. Each instance of Repeat can be paid
    separately to execute the spell one additional time.
    """
    return card.keyword_value(Keyword.REPEAT)


def can_pay_repeat(card: CardInstance) -> bool:
    """Return True if the spell has Repeat and it can be paid."""
    return card.has_keyword(Keyword.REPEAT)


# ---------------------------------------------------------------------------
# Weaponmaster — play effect: choose an Equipment, pay discounted Equip (747)
# ---------------------------------------------------------------------------
# NOTE: The actual effect execution (choosing equipment, paying cost,
# attaching) is handled by the effect resolver / card pipeline.
# This helper checks the keyword presence and discount amount.


def has_weaponmaster(unit: CardInstance) -> bool:
    """Return True if the unit has Weaponmaster."""
    return unit.has_keyword(Keyword.WEAPONMASTER)


def get_weaponmaster_discount() -> int:
    """
    Return the Equip cost discount granted by Weaponmaster.
    Per rule 747.1.c: cost reduced by [A] (one power of any domain).
    """
    return 1


# ---------------------------------------------------------------------------
# Vision — when played, look at top card of deck, may recycle it (743)
# ---------------------------------------------------------------------------


def apply_vision(unit: CardInstance, gs: GameState) -> list[str]:
    """
    Apply Vision: the controller looks at the top card of their deck.
    Each instance of Vision triggers separately (rule 743.2).
    Returns log messages describing what the player sees.
    """
    logs: list[str] = []
    count = max(1, sum(1 for k in unit.all_keywords() if k.keyword == Keyword.VISION))
    ps = gs.players[unit.controller_id]
    for i in range(count):
        if not ps.main_deck:
            logs.append(f"Vision: deck is empty, nothing to see")
            break
        top_id = ps.main_deck[0]
        top_card = gs.instances.get(top_id)
        if top_card:
            logs.append(
                f"Vision: {unit.controller_id} sees {top_card.name} on top of deck"
            )
            logger.info("[VISION] %s sees %s", unit.controller_id, top_card.name)
    return logs


def apply_vision_recycle(
    unit: CardInstance,
    gs: GameState,
    do_recycle: bool,
) -> list[str]:
    """
    Optionally recycle the top card seen via Vision.
    If ``do_recycle`` is True, the top card is moved to the bottom of the deck.
    """
    if not do_recycle:
        return []
    ps = gs.players[unit.controller_id]
    if not ps.main_deck:
        return ["Deck is empty, nothing to recycle"]
    card_id = ps.main_deck.pop(0)
    ps.main_deck.append(card_id)
    card = gs.instances.get(card_id)
    name = card.name if card else card_id
    logger.info("[VISION] %s recycled %s to bottom", unit.controller_id, name)
    return [f"Vision: {name} recycled to bottom of deck"]
