"""Scoring: Hold, Conquer, Burn-Out, and win condition checks."""

from __future__ import annotations

from .enums import ControlStatus, ZoneType
from .game_state import GameState, _draw_cards


def check_win(gs: GameState) -> str | None:
    """Check if any player has reached the victory score. Returns winner_id or None."""
    for pid, ps in gs.players.items():
        if ps.score >= gs.victory_score:
            return pid
    return None


def score_hold(gs: GameState, player_id: str) -> list[str]:
    """
    Called during Beginning Phase. Score 1 point for each battlefield
    the turn player controls. Each battlefield can only be scored once
    per turn by either method.
    """
    ps = gs.players[player_id]
    logs: list[str] = []

    for bf_id, bf in gs.battlefields.items():
        if (
            bf.control_status == ControlStatus.CONTROLLED
            and bf.controller_id == player_id
            and bf_id not in ps.battlefields_scored_this_turn
        ):
            if _can_score_point(gs, player_id, "hold"):
                ps.score += 1
                ps.battlefields_scored_this_turn.add(bf_id)
                bf.scored_this_turn_by = player_id
                logs.append(f"{ps.display_name} scores 1 point from Holding {_bf_name(gs, bf_id)}")
                gs.log.add(f"Hold: {ps.display_name} +1 (total: {ps.score})")
            else:
                logs.append(f"{ps.display_name} cannot score final point from Hold right now")

    return logs


def score_conquer(gs: GameState, player_id: str, battlefield_id: str) -> list[str]:
    """
    Called when a player takes control of a battlefield after Showdown/Combat.
    """
    ps = gs.players[player_id]
    bf = gs.battlefields[battlefield_id]
    logs: list[str] = []

    if battlefield_id in ps.battlefields_scored_this_turn:
        logs.append(f"{_bf_name(gs, battlefield_id)} already scored this turn")
        return logs

    if _can_score_point(gs, player_id, "conquer"):
        ps.score += 1
        ps.battlefields_scored_this_turn.add(battlefield_id)
        bf.scored_this_turn_by = player_id
        logs.append(f"{ps.display_name} scores 1 point from Conquering {_bf_name(gs, battlefield_id)}")
        gs.log.add(f"Conquer: {ps.display_name} +1 (total: {ps.score})")
    else:
        # At 7 points but haven't scored all battlefields — draw a card instead
        drawn = _draw_cards(gs, player_id, 1)
        logs.append(
            f"{ps.display_name} at {ps.score} points — must score all battlefields "
            f"to conquer for the final point. Draws a card instead."
        )

    return logs


def _can_score_point(gs: GameState, player_id: str, method: str) -> bool:
    """
    Check if the player can gain a point. The 7→8 restriction:
    - Hold: always OK
    - Conquer: only if all battlefields have been scored this turn
    """
    ps = gs.players[player_id]

    # Not at the final point threshold — always OK
    if ps.score < gs.victory_score - 1:
        return True

    # At the final point
    if method == "hold":
        return True

    if method == "conquer":
        # Must have scored every battlefield this turn
        all_bf_ids = set(gs.battlefields.keys())
        return all_bf_ids.issubset(ps.battlefields_scored_this_turn)

    return False


def perform_burn_out(gs: GameState, player_id: str) -> list[str]:
    """
    Burn Out: player's deck is empty and they need to draw.
    1. Recycle trash into main deck (shuffled)
    2. Opponent gains 1 point
    3. Continue the draw
    """
    import random

    ps = gs.players[player_id]
    logs: list[str] = []

    # Recycle trash into main deck
    if ps.trash:
        recycled = list(ps.trash)
        random.shuffle(recycled)
        for iid in recycled:
            inst = gs.instances[iid]
            inst.zone = ZoneType.MAIN_DECK
            inst.location_id = None
            inst.damage = 0
            inst.exhausted = False
            inst.buff_counter = False
            inst.stunned = False
            inst.granted_keywords.clear()
            inst.might_modifiers.clear()
        ps.main_deck.extend(recycled)
        ps.trash.clear()
        logs.append(f"{ps.display_name} burns out: trash recycled into deck")

    # Opponent gains 1 point
    opp_id = gs.opponent_id(player_id)
    opp = gs.players[opp_id]
    opp.score += 1
    logs.append(f"{opp.display_name} gains 1 point from Burn Out (total: {opp.score})")
    gs.log.add(f"Burn Out: {opp.display_name} +1 (total: {opp.score})")

    return logs


def _bf_name(gs: GameState, bf_id: str) -> str:
    bf = gs.battlefields.get(bf_id)
    if bf:
        inst = gs.get_instance(bf.card_instance_id)
        if inst:
            return inst.name
    return bf_id
