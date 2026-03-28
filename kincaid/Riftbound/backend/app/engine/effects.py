"""Effect function registry for card-specific abilities.

Each registered effect is a callable(source, gs, targets) -> list[str] of log messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .card_types import CardInstance
    from .game_state import GameState

EffectFn = Callable[["CardInstance", "GameState", list[str]], list[str]]

EFFECT_REGISTRY: dict[str, EffectFn] = {}


def register_effect(name: str):
    def decorator(fn: EffectFn) -> EffectFn:
        EFFECT_REGISTRY[name] = fn
        return fn
    return decorator


def resolve_effect(
    name: str,
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    fn = EFFECT_REGISTRY.get(name)
    if fn is None:
        return [f"Unknown effect: {name}"]
    return fn(source, gs, targets)


# ---------------------------------------------------------------------------
# Built-in effects for starter cards
# ---------------------------------------------------------------------------

@register_effect("rune_add_energy")
def rune_add_energy(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    from .enums import ZoneType
    ps = gs.players[source.controller_id]
    ps.rune_pool.add_energy(1)
    source.exhausted = True
    return [f"{source.name} exhausted: +1 Energy"]


@register_effect("rune_recycle_power")
def rune_recycle_power(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    from .enums import Domain
    ps = gs.players[source.controller_id]
    if source.definition.domains:
        domain = source.definition.domains[0]
        ps.rune_pool.add_power(domain, 1)
        # Recycle: move rune to bottom of rune deck
        _recycle_rune(source, gs)
        return [f"{source.name} recycled: +1 {domain.value} Power"]
    return [f"{source.name} recycled but has no domain"]


def _recycle_rune(source: CardInstance, gs: GameState) -> None:
    from .enums import ZoneType
    ps = gs.players[source.owner_id]
    # Remove from current location
    if source.instance_id in (gs.base_runes.get(source.controller_id) or []):
        gs.base_runes[source.controller_id].remove(source.instance_id)
    # Put on bottom of rune deck
    source.zone = ZoneType.RUNE_DECK
    source.location_id = None
    source.exhausted = False
    ps.rune_deck.append(source.instance_id)


@register_effect("deal_damage_to_unit")
def deal_damage_to_unit(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Deal 3 damage to a target unit."""
    if not targets:
        return ["No target"]
    target = gs.get_instance(targets[0])
    if target:
        target.damage += 3
        return [f"{source.name} deals 3 damage to {target.name}"]
    return ["Target not found"]


@register_effect("deal_damage_to_unit_at_bf")
def deal_damage_to_unit_at_bf(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Deal 2 damage to a unit at a battlefield."""
    if not targets:
        return ["No target"]
    target = gs.get_instance(targets[0])
    if target:
        target.damage += 2
        return [f"{source.name} deals 2 damage to {target.name}"]
    return ["Target not found"]


@register_effect("buff_friendly_unit")
def buff_friendly_unit(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Place a buff counter on a friendly unit (max 1)."""
    if not targets:
        return ["No target"]
    target = gs.get_instance(targets[0])
    if target and not target.buff_counter:
        target.buff_counter = True
        return [f"{target.name} gains a buff (+1 Might)"]
    elif target and target.buff_counter:
        return [f"{target.name} already has a buff"]
    return ["Target not found"]


@register_effect("counter_spell")
def counter_spell(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Counter a spell on the chain."""
    if not targets:
        return ["No target"]
    # Find the chain item to counter
    for i, item in enumerate(gs.chain.stack):
        if item.item_id == targets[0] or item.card_instance_id == targets[0]:
            countered = gs.chain.stack.pop(i)
            if countered.card_instance_id:
                card = gs.get_instance(countered.card_instance_id)
                if card:
                    from .enums import ZoneType
                    card.zone = ZoneType.TRASH
                    ps = gs.players[card.owner_id]
                    ps.trash.append(card.instance_id)
                    return [f"{card.name} was countered"]
            return ["Ability countered"]
    return ["Chain item not found"]


@register_effect("give_might_boost")
def give_might_boost(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Give a friendly unit +3 Might this turn."""
    if not targets:
        return ["No target"]
    target = gs.get_instance(targets[0])
    if target:
        target.might_modifiers.append(3)
        return [f"{target.name} gains +3 Might this turn"]
    return ["Target not found"]


@register_effect("move_unit_to_battlefield")
def move_unit_to_battlefield(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Move a friendly unit to a battlefield. targets = [unit_id, battlefield_id]."""
    if len(targets) < 2:
        return ["Need unit and battlefield targets"]
    # The actual move logic is handled by the action executor
    return [f"Move effect resolved"]


@register_effect("heal_unit")
def heal_unit(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Heal 3 damage from a friendly unit."""
    if not targets:
        return ["No target"]
    target = gs.get_instance(targets[0])
    if target:
        healed = min(target.damage, 3)
        target.damage -= healed
        return [f"{target.name} healed {healed} damage"]
    return ["Target not found"]


@register_effect("reckless_charge")
def reckless_charge(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Deal 4 to an enemy unit, deal 1 to a friendly unit. targets=[enemy_id, friendly_id]."""
    logs = []
    if len(targets) >= 1:
        enemy = gs.get_instance(targets[0])
        if enemy:
            enemy.damage += 4
            logs.append(f"{source.name} deals 4 damage to {enemy.name}")
    if len(targets) >= 2:
        friendly = gs.get_instance(targets[1])
        if friendly:
            friendly.damage += 1
            logs.append(f"{source.name} deals 1 damage to {friendly.name}")
    return logs or ["No targets"]


@register_effect("draw_one")
def draw_one(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Draw 1 card."""
    from .game_state import _draw_cards
    drawn = _draw_cards(gs, source.controller_id, 1)
    if drawn:
        card = gs.get_instance(drawn[0])
        name = card.name if card else "a card"
        return [f"{source.controller_id} draws {name}"]
    return ["Deck empty"]


@register_effect("vision_effect")
def vision_effect(source: CardInstance, gs: GameState, targets: list[str]) -> list[str]:
    """Look at top of deck. May recycle it."""
    ps = gs.players[source.controller_id]
    if ps.main_deck:
        top_id = ps.main_deck[0]
        top_card = gs.get_instance(top_id)
        name = top_card.name if top_card else "unknown"
        # For now: auto-reveal to controller. Recycle choice deferred to client action.
        return [f"Vision: top card is {name}"]
    return ["Vision: deck is empty"]
