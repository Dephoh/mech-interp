"""Keyword-specific logic that augments core engine behavior."""

from __future__ import annotations

from .card_types import CardInstance, KeywordInstance
from .enums import CardType, CombatRole, Domain, Keyword, ZoneType


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


def process_legion(card: CardInstance, gs) -> bool:
    """Check if Legion condition is met (played another Main Deck card this turn)."""
    ps = gs.players[card.controller_id]
    # For spells/play effects: "played another card before this one"
    return ps.played_main_deck_this_turn > 0


def process_deathknell(unit: CardInstance, gs) -> list[str]:
    """
    When a unit with Deathknell dies, create chain items for its effects.
    Returns list of log messages.
    """
    from .game_state import ChainItem
    logs = []
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


def should_die_temporary(unit: CardInstance) -> bool:
    """Check if a Temporary unit should be killed at start of Beginning Phase."""
    return unit.has_keyword(Keyword.TEMPORARY) and not unit.entered_this_turn
