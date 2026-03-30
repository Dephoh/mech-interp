"""Game state persistence: serialize/deserialize GameState to/from JSON-safe dicts.

This module handles full round-trip serialization of GameState, including all
nested dataclasses (PlayerState, BattlefieldState, CardInstance, etc.).

CardDefinitions are NOT serialized inline -- we store only the card_id and
look them up from CardDB on deserialization. This keeps save files small and
avoids duplicating the card database.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .card_db import CardDB
from .card_types import CardDefinition, CardInstance, KeywordInstance
from .enums import (
    AbilityType,
    CardType,
    CombatRole,
    ControlStatus,
    Domain,
    GameMode,
    Keyword,
    Phase,
    SuperType,
    ZoneType,
)
from .game_state import (
    ActiveModifier,
    ActiveReplacement,
    BattlefieldState,
    ChainItem,
    ChainState,
    CombatState,
    GameLog,
    GameState,
    PlayerState,
    RunePool,
    ShowdownState,
)

logger = logging.getLogger("riftbound.persistence")


# ---------------------------------------------------------------------------
# Serialization: GameState -> dict
# ---------------------------------------------------------------------------


def _rune_pool_to_dict(rp: RunePool) -> dict[str, Any]:
    return {
        "energy": rp.energy,
        "power": {d.value: v for d, v in rp.power.items()},
    }


def _player_state_to_dict(ps: PlayerState) -> dict[str, Any]:
    return {
        "player_id": ps.player_id,
        "display_name": ps.display_name,
        "score": ps.score,
        "hand": list(ps.hand),
        "main_deck": list(ps.main_deck),
        "rune_deck": list(ps.rune_deck),
        "trash": list(ps.trash),
        "banishment": list(ps.banishment),
        "legend_zone": ps.legend_zone,
        "champion_zone": ps.champion_zone,
        "rune_pool": _rune_pool_to_dict(ps.rune_pool),
        "battlefields_scored_this_turn": list(ps.battlefields_scored_this_turn),
        "played_main_deck_this_turn": ps.played_main_deck_this_turn,
        "is_first_turn": ps.is_first_turn,
        "goes_second": ps.goes_second,
    }


def _keyword_instance_to_dict(ki: KeywordInstance) -> dict[str, Any]:
    return {"keyword": ki.keyword.value, "value": ki.value}


def _card_instance_to_dict(inst: CardInstance) -> dict[str, Any]:
    return {
        "instance_id": inst.instance_id,
        "card_id": inst.definition.card_id,
        "owner_id": inst.owner_id,
        "controller_id": inst.controller_id,
        "zone": inst.zone.value,
        "location_id": inst.location_id,
        "exhausted": inst.exhausted,
        "damage": inst.damage,
        "buff_counter": inst.buff_counter,
        "stunned": inst.stunned,
        "combat_role": inst.combat_role.value,
        "facedown": inst.facedown,
        "entered_this_turn": inst.entered_this_turn,
        "hidden_at_battlefield": inst.hidden_at_battlefield,
        "hidden_ready": inst.hidden_ready,
        "accelerated": inst.accelerated,
        "attached_to": inst.attached_to,
        "attached_cards": list(inst.attached_cards),
        "granted_keywords": [_keyword_instance_to_dict(k) for k in inst.granted_keywords],
        "might_modifiers": list(inst.might_modifiers),
        "aura_might_bonus": inst.aura_might_bonus,
        "gear_might_bonus": inst.gear_might_bonus,
        "trait_might_set": inst.trait_might_set,
        "aura_keywords": [_keyword_instance_to_dict(k) for k in inst.aura_keywords],
    }


def _battlefield_to_dict(bf: BattlefieldState) -> dict[str, Any]:
    return {
        "battlefield_id": bf.battlefield_id,
        "card_instance_id": bf.card_instance_id,
        "control_status": bf.control_status.value,
        "controller_id": bf.controller_id,
        "contested_by": bf.contested_by,
        "units": list(bf.units),
        "facedown_card": bf.facedown_card,
        "scored_this_turn_by": bf.scored_this_turn_by,
        "showdown_staged": bf.showdown_staged,
        "combat_staged": bf.combat_staged,
    }


def _chain_item_to_dict(item: ChainItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "source_instance_id": item.source_instance_id,
        "ability_id": item.ability_id,
        "card_instance_id": item.card_instance_id,
        "controller_id": item.controller_id,
        "pending": item.pending,
        "targets": list(item.targets),
    }


def _chain_state_to_dict(cs: ChainState) -> dict[str, Any]:
    return {
        "stack": [_chain_item_to_dict(item) for item in cs.stack],
        "passed_players": list(cs.passed_players),
    }


def _combat_state_to_dict(cs: CombatState) -> dict[str, Any]:
    return {
        "battlefield_id": cs.battlefield_id,
        "attacker_id": cs.attacker_id,
        "defender_id": cs.defender_id,
        "phase": cs.phase,
        "attacker_assignment": cs.attacker_assignment,
        "defender_assignment": cs.defender_assignment,
    }


def _showdown_state_to_dict(ss: ShowdownState) -> dict[str, Any]:
    return {
        "battlefield_id": ss.battlefield_id,
        "initiator_id": ss.initiator_id,
        "focus_player_id": ss.focus_player_id,
        "passed_players": list(ss.passed_players),
        "is_combat_showdown": ss.is_combat_showdown,
    }


def _active_modifier_to_dict(mod: ActiveModifier) -> dict[str, Any]:
    return {
        "source_instance_id": mod.source_instance_id,
        "ability_id": mod.ability_id,
        "stat": mod.stat,
        "amount": mod.amount,
        "target_spec": mod.target_spec,
        "exclude_self": mod.exclude_self,
        "duration": mod.duration,
    }


def _active_replacement_to_dict(rep: ActiveReplacement) -> dict[str, Any]:
    return {
        "source_instance_id": rep.source_instance_id,
        "ability_id": rep.ability_id,
        "watch_event": rep.watch_event,
        "condition": rep.condition,
        "alternative_ir": rep.alternative_ir,
        "target_spec": rep.target_spec,
    }


def game_state_to_dict(gs: GameState) -> dict[str, Any]:
    """Serialize a GameState to a JSON-safe dictionary."""
    return {
        "game_id": gs.game_id,
        "game_mode": gs.game_mode.value,
        "players": {pid: _player_state_to_dict(ps) for pid, ps in gs.players.items()},
        "player_order": list(gs.player_order),
        "turn_player_id": gs.turn_player_id,
        "active_player_id": gs.active_player_id,
        "phase": gs.phase.value,
        "turn_number": gs.turn_number,
        "teams": dict(gs.teams),
        "instances": {iid: _card_instance_to_dict(inst) for iid, inst in gs.instances.items()},
        "battlefields": {bf_id: _battlefield_to_dict(bf) for bf_id, bf in gs.battlefields.items()},
        "base_units": {pid: list(uids) for pid, uids in gs.base_units.items()},
        "base_gear": {pid: list(gids) for pid, gids in gs.base_gear.items()},
        "base_runes": {pid: list(rids) for pid, rids in gs.base_runes.items()},
        "chain": _chain_state_to_dict(gs.chain),
        "active_combat": _combat_state_to_dict(gs.active_combat) if gs.active_combat else None,
        "active_showdown": _showdown_state_to_dict(gs.active_showdown) if gs.active_showdown else None,
        "pending_combats": list(gs.pending_combats),
        "pending_showdowns": list(gs.pending_showdowns),
        "game_over": gs.game_over,
        "winner_id": gs.winner_id,
        "mulligan_done": dict(gs.mulligan_done),
        "log": [dict(e) for e in gs.log.entries],
        "victory_score": gs.victory_score,
        "active_modifiers": [_active_modifier_to_dict(m) for m in gs.active_modifiers],
        "active_replacements": [_active_replacement_to_dict(r) for r in gs.active_replacements],
        "pending_choice": gs.pending_choice,
        "last_effect_succeeded": gs.last_effect_succeeded,
    }


# ---------------------------------------------------------------------------
# Deserialization: dict -> GameState
# ---------------------------------------------------------------------------


def _rune_pool_from_dict(d: dict[str, Any]) -> RunePool:
    return RunePool(
        energy=d["energy"],
        power={Domain(k): v for k, v in d["power"].items()},
    )


def _player_state_from_dict(d: dict[str, Any]) -> PlayerState:
    ps = PlayerState(
        player_id=d["player_id"],
        display_name=d["display_name"],
    )
    ps.score = d["score"]
    ps.hand = list(d["hand"])
    ps.main_deck = list(d["main_deck"])
    ps.rune_deck = list(d["rune_deck"])
    ps.trash = list(d["trash"])
    ps.banishment = list(d["banishment"])
    ps.legend_zone = d["legend_zone"]
    ps.champion_zone = d["champion_zone"]
    ps.rune_pool = _rune_pool_from_dict(d["rune_pool"])
    ps.battlefields_scored_this_turn = set(d["battlefields_scored_this_turn"])
    ps.played_main_deck_this_turn = d["played_main_deck_this_turn"]
    ps.is_first_turn = d["is_first_turn"]
    ps.goes_second = d["goes_second"]
    return ps


def _keyword_instance_from_dict(d: dict[str, Any]) -> KeywordInstance:
    return KeywordInstance(keyword=Keyword(d["keyword"]), value=d.get("value", 0))


def _card_instance_from_dict(
    d: dict[str, Any], card_db: dict[str, CardDefinition]
) -> CardInstance:
    card_id = d["card_id"]
    definition = card_db.get(card_id)
    if definition is None:
        raise ValueError(
            f"Card definition not found for card_id={card_id!r}. "
            f"Ensure CardDB is loaded before deserializing."
        )
    inst = CardInstance(
        instance_id=d["instance_id"],
        definition=definition,
        owner_id=d["owner_id"],
        controller_id=d["controller_id"],
        zone=ZoneType(d["zone"]),
        location_id=d.get("location_id"),
    )
    inst.exhausted = d.get("exhausted", False)
    inst.damage = d.get("damage", 0)
    inst.buff_counter = d.get("buff_counter", False)
    inst.stunned = d.get("stunned", False)
    inst.combat_role = CombatRole(d.get("combat_role", "none"))
    inst.facedown = d.get("facedown", False)
    inst.entered_this_turn = d.get("entered_this_turn", True)
    inst.hidden_at_battlefield = d.get("hidden_at_battlefield")
    inst.hidden_ready = d.get("hidden_ready", False)
    inst.accelerated = d.get("accelerated", False)
    inst.attached_to = d.get("attached_to")
    inst.attached_cards = list(d.get("attached_cards", []))
    inst.granted_keywords = [
        _keyword_instance_from_dict(k) for k in d.get("granted_keywords", [])
    ]
    inst.might_modifiers = list(d.get("might_modifiers", []))
    inst.aura_might_bonus = d.get("aura_might_bonus", 0)
    inst.gear_might_bonus = d.get("gear_might_bonus", 0)
    inst.trait_might_set = d.get("trait_might_set")
    inst.aura_keywords = [
        _keyword_instance_from_dict(k) for k in d.get("aura_keywords", [])
    ]
    return inst


def _battlefield_from_dict(d: dict[str, Any]) -> BattlefieldState:
    return BattlefieldState(
        battlefield_id=d["battlefield_id"],
        card_instance_id=d["card_instance_id"],
        control_status=ControlStatus(d["control_status"]),
        controller_id=d.get("controller_id"),
        contested_by=d.get("contested_by"),
        units=list(d.get("units", [])),
        facedown_card=d.get("facedown_card"),
        scored_this_turn_by=d.get("scored_this_turn_by"),
        showdown_staged=d.get("showdown_staged", False),
        combat_staged=d.get("combat_staged", False),
    )


def _chain_item_from_dict(d: dict[str, Any]) -> ChainItem:
    return ChainItem(
        item_id=d["item_id"],
        source_instance_id=d.get("source_instance_id"),
        ability_id=d.get("ability_id"),
        card_instance_id=d.get("card_instance_id"),
        controller_id=d["controller_id"],
        pending=d.get("pending", True),
        targets=list(d.get("targets", [])),
    )


def _chain_state_from_dict(d: dict[str, Any]) -> ChainState:
    cs = ChainState()
    cs.stack = [_chain_item_from_dict(item) for item in d.get("stack", [])]
    cs.passed_players = set(d.get("passed_players", []))
    return cs


def _combat_state_from_dict(d: dict[str, Any]) -> CombatState:
    return CombatState(
        battlefield_id=d["battlefield_id"],
        attacker_id=d["attacker_id"],
        defender_id=d["defender_id"],
        phase=d.get("phase", "showdown"),
        attacker_assignment=d.get("attacker_assignment"),
        defender_assignment=d.get("defender_assignment"),
    )


def _showdown_state_from_dict(d: dict[str, Any]) -> ShowdownState:
    return ShowdownState(
        battlefield_id=d["battlefield_id"],
        initiator_id=d["initiator_id"],
        focus_player_id=d["focus_player_id"],
        passed_players=set(d.get("passed_players", [])),
        is_combat_showdown=d.get("is_combat_showdown", False),
    )


def _active_modifier_from_dict(d: dict[str, Any]) -> ActiveModifier:
    return ActiveModifier(
        source_instance_id=d["source_instance_id"],
        ability_id=d["ability_id"],
        stat=d["stat"],
        amount=d["amount"],
        target_spec=d["target_spec"],
        exclude_self=d.get("exclude_self", False),
        duration=d.get("duration", "continuous"),
    )


def _active_replacement_from_dict(d: dict[str, Any]) -> ActiveReplacement:
    return ActiveReplacement(
        source_instance_id=d["source_instance_id"],
        ability_id=d["ability_id"],
        watch_event=d["watch_event"],
        condition=d["condition"],
        alternative_ir=d["alternative_ir"],
        target_spec=d["target_spec"],
    )


def game_state_from_dict(
    d: dict[str, Any],
    card_db: dict[str, CardDefinition] | None = None,
) -> GameState:
    """Deserialize a GameState from a dictionary.

    If card_db is not provided, uses CardDB.all_cards() (requires CardDB to be loaded).
    """
    if card_db is None:
        card_db = CardDB.all_cards()

    gs = GameState(game_id=d["game_id"])
    gs.game_mode = GameMode(d.get("game_mode", "standard_1v1"))
    gs.players = {
        pid: _player_state_from_dict(ps_data)
        for pid, ps_data in d["players"].items()
    }
    gs.player_order = list(d["player_order"])
    gs.turn_player_id = d["turn_player_id"]
    gs.active_player_id = d["active_player_id"]
    gs.phase = Phase(d["phase"])
    gs.turn_number = d["turn_number"]
    gs.teams = {k: v for k, v in d.get("teams", {}).items()}
    gs.instances = {
        iid: _card_instance_from_dict(inst_data, card_db)
        for iid, inst_data in d["instances"].items()
    }
    gs.battlefields = {
        bf_id: _battlefield_from_dict(bf_data)
        for bf_id, bf_data in d["battlefields"].items()
    }
    gs.base_units = {pid: list(uids) for pid, uids in d["base_units"].items()}
    gs.base_gear = {pid: list(gids) for pid, gids in d["base_gear"].items()}
    gs.base_runes = {pid: list(rids) for pid, rids in d["base_runes"].items()}
    gs.chain = _chain_state_from_dict(d["chain"])
    gs.active_combat = (
        _combat_state_from_dict(d["active_combat"]) if d.get("active_combat") else None
    )
    gs.active_showdown = (
        _showdown_state_from_dict(d["active_showdown"]) if d.get("active_showdown") else None
    )
    gs.pending_combats = list(d.get("pending_combats", []))
    gs.pending_showdowns = list(d.get("pending_showdowns", []))
    gs.game_over = d["game_over"]
    gs.winner_id = d.get("winner_id")
    gs.mulligan_done = dict(d["mulligan_done"])
    gs.log = GameLog(entries=[dict(e) for e in d.get("log", [])])
    gs.victory_score = d.get("victory_score", 8)
    gs.active_modifiers = [
        _active_modifier_from_dict(m) for m in d.get("active_modifiers", [])
    ]
    gs.active_replacements = [
        _active_replacement_from_dict(r) for r in d.get("active_replacements", [])
    ]
    gs.pending_choice = d.get("pending_choice")
    gs.last_effect_succeeded = d.get("last_effect_succeeded", False)
    return gs


# ---------------------------------------------------------------------------
# File-based persistence
# ---------------------------------------------------------------------------

DEFAULT_SAVE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "game_states")
)


def _save_path(room_id: str, save_dir: str | None = None) -> str:
    """Return the file path for a room's saved state."""
    save_dir = save_dir or DEFAULT_SAVE_DIR
    return os.path.join(save_dir, f"{room_id}.json")


def save_game_state(
    room_id: str,
    gs: GameState,
    save_dir: str | None = None,
) -> str:
    """Serialize and save a GameState to a JSON file.

    Returns the file path written.
    """
    save_dir = save_dir or DEFAULT_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = _save_path(room_id, save_dir)
    data = game_state_to_dict(gs)
    # Include room_id in the saved data for provenance
    data["_room_id"] = room_id
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.debug("Saved game state for room=%s to %s", room_id, path)
    return path


def load_game_state(
    room_id: str,
    save_dir: str | None = None,
    card_db: dict[str, CardDefinition] | None = None,
) -> GameState | None:
    """Load a saved GameState from a JSON file.

    Returns None if no save file exists.
    """
    path = _save_path(room_id, save_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Strip provenance key before deserialization
        data.pop("_room_id", None)
        gs = game_state_from_dict(data, card_db=card_db)
        logger.info("Loaded game state for room=%s from %s", room_id, path)
        return gs
    except Exception:
        logger.exception("Failed to load game state for room=%s from %s", room_id, path)
        return None


def delete_game_state(room_id: str, save_dir: str | None = None) -> bool:
    """Delete a saved game state file. Returns True if file was deleted."""
    path = _save_path(room_id, save_dir)
    if os.path.exists(path):
        os.remove(path)
        logger.debug("Deleted game state file for room=%s", room_id)
        return True
    return False


def list_saved_rooms(save_dir: str | None = None) -> list[str]:
    """Return room_ids that have saved game state files."""
    save_dir = save_dir or DEFAULT_SAVE_DIR
    if not os.path.isdir(save_dir):
        return []
    room_ids = []
    for fname in os.listdir(save_dir):
        if fname.endswith(".json"):
            room_ids.append(fname[:-5])  # strip .json
    return room_ids
