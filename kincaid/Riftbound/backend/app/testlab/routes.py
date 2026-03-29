"""FastAPI routes for the Card Test Lab."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from .builder import build_game_from_scenario
from .scenario_generator import generate_all_scenarios
from .scenarios import ScenarioDef

logger = logging.getLogger("riftbound.testlab.routes")

router = APIRouter(prefix="/testlab", tags=["testlab"])

_room_manager = None


def set_room_manager(rm: Any) -> None:
    global _room_manager
    _room_manager = rm


def _find_scenario(scenario_id: str) -> ScenarioDef | None:
    for s in generate_all_scenarios():
        if s.scenario_id == scenario_id:
            return s
    return None


@router.get("/scenarios")
def list_scenarios():
    scenarios = generate_all_scenarios()
    return [
        {
            "scenario_id": s.scenario_id,
            "name": s.name,
            "description": s.description,
            "category": s.category,
            "tags": s.tags,
            "expected_behavior": s.expected_behavior,
        }
        for s in scenarios
    ]


@router.post("/load")
async def load_scenario(body: dict):
    scenario_id = body.get("scenario_id", "")
    scenario = _find_scenario(scenario_id)
    if not scenario:
        return {"error": f"Scenario '{scenario_id}' not found"}

    if not _room_manager:
        return {"error": "Room manager not initialized"}

    gs = build_game_from_scenario(scenario)
    room_id = _room_manager.create_room()
    session = _room_manager.get_room(room_id)
    session.game_state = gs
    session._testlab = True

    logger.info("[TESTLAB] Loaded scenario=%s room=%s", scenario_id, room_id)
    return {
        "room_id": room_id,
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "description": scenario.description,
        "expected_behavior": scenario.expected_behavior,
    }


@router.post("/reset")
async def reset_scenario(body: dict):
    room_id = body.get("room_id", "")
    scenario_id = body.get("scenario_id", "")

    if not _room_manager:
        return {"error": "Room manager not initialized"}

    session = _room_manager.get_room(room_id)
    if not session:
        return {"error": f"Room '{room_id}' not found"}

    scenario = _find_scenario(scenario_id)
    if not scenario:
        return {"error": f"Scenario '{scenario_id}' not found"}

    gs = build_game_from_scenario(scenario)
    session.game_state = gs

    logger.info("[TESTLAB] Reset scenario=%s room=%s", scenario_id, room_id)
    return {"ok": True, "scenario_id": scenario_id}


@router.post("/switch")
async def switch_scenario(body: dict):
    room_id = body.get("room_id", "")
    scenario_id = body.get("scenario_id", "")

    if not _room_manager:
        return {"error": "Room manager not initialized"}

    session = _room_manager.get_room(room_id)
    if not session:
        return {"error": f"Room '{room_id}' not found"}

    scenario = _find_scenario(scenario_id)
    if not scenario:
        return {"error": f"Scenario '{scenario_id}' not found"}

    gs = build_game_from_scenario(scenario)
    session.game_state = gs

    # Broadcast new state to connected players
    await _room_manager._broadcast_state(session)

    logger.info("[TESTLAB] Switched to scenario=%s room=%s", scenario_id, room_id)
    return {
        "ok": True,
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "description": scenario.description,
        "expected_behavior": scenario.expected_behavior,
    }
