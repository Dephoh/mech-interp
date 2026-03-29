"""FastAPI entry point: REST + WebSocket routes."""

from __future__ import annotations

import logging
import os
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engine.card_db import CardDB
from .room_manager import RoomManager

# ---------------------------------------------------------------------------
# Logging setup — writes to file + console
# ---------------------------------------------------------------------------

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_FILE = os.path.join(LOG_DIR, "server.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("riftbound")

FRONTEND_DIST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
)


# ---------------------------------------------------------------------------
# Lifespan: load card DB once on startup
# ---------------------------------------------------------------------------

room_manager: RoomManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global room_manager
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)
    CardDB.load_all(data_dir)
    logger.info("Card DB loaded: %d card definitions", len(CardDB.all_cards()))
    room_manager = RoomManager()
    logger.info("RoomManager initialized, server ready")
    yield


app = FastAPI(title="Riftbound Simulator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/rooms")
async def create_room():
    room_id = room_manager.create_room()
    return {"room_id": room_id}


@app.get("/rooms/{room_id}")
async def get_room(room_id: str):
    session = room_manager.get_room(room_id)
    if not session:
        return {"error": "Room not found"}
    return {
        "room_id": room_id,
        "player_count": session.player_count,
        "is_full": session.is_full,
        "game_started": session.game_state is not None,
    }


@app.get("/debug/{room_id}")
async def debug_room(room_id: str):
    """Dump full server-side state for a room. Use this for debugging."""
    session = room_manager.get_room(room_id)
    if not session:
        return {"error": "Room not found"}
    gs = session.game_state
    if not gs:
        return {"error": "Game not started", "player_count": session.player_count}
    return {
        "room_id": room_id,
        "phase": gs.phase.value,
        "turn_number": gs.turn_number,
        "turn_player_id": gs.turn_player_id,
        "active_player_id": gs.active_player_id,
        "game_over": gs.game_over,
        "mulligan_done": gs.mulligan_done,
        "players": {
            pid: {
                "display_name": ps.display_name,
                "score": ps.score,
                "hand_count": len(ps.hand),
                "deck_count": len(ps.main_deck),
                "rune_deck_count": len(ps.rune_deck),
                "rune_board_count": len([r for r in gs.base_runes.get(pid, [])]),
            }
            for pid, ps in gs.players.items()
        },
        "battlefields": {
            bf_id: {
                "control": bf.control_status.value,
                "controller": bf.controller_id,
                "unit_count": len(bf.units),
            }
            for bf_id, bf in gs.battlefields.items()
        },
        "chain_depth": len(gs.chain.stack),
        "active_combat": gs.active_combat is not None,
        "active_showdown": gs.active_showdown is not None,
        "game_log": [e["message"] for e in gs.log.entries[-50:]],
        "server_log_tail": _read_log_tail(100),
    }


def _read_log_tail(n: int) -> list[str]:
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n:]]
    except FileNotFoundError:
        return []


@app.get("/cards")
async def list_cards():
    all_cards = CardDB.all_cards()
    return {
        cid: {
            "card_id": cdef.card_id,
            "name": cdef.name,
            "card_type": cdef.card_type.value,
            "domains": [d.value for d in cdef.domains],
            "cost_energy": cdef.cost_energy,
            "base_might": cdef.base_might,
            "text": cdef.text,
        }
        for cid, cdef in all_cards.items()
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    player_id: str | None = None
    session = None
    is_reconnect = False
    logger.info("[WS] New connection for room=%s", room_id)

    try:
        # First message must be JOIN_ROOM or RECONNECT
        data = await websocket.receive_json()
        msg_type = data.get("type", "")

        if msg_type == "RECONNECT":
            # --- Reconnection flow ---
            reconnect_token = data.get("reconnect_token", "")
            logger.info("[WS] RECONNECT attempt room=%s token=%s...",
                         room_id, reconnect_token[:8] if reconnect_token else "NONE")
            result = await room_manager.reconnect_player(room_id, reconnect_token, websocket)
            if not result:
                logger.warning("[WS] RECONNECT failed room=%s token=%s...",
                                room_id, reconnect_token[:8] if reconnect_token else "NONE")
                await websocket.send_json({
                    "type": "ERROR",
                    "action_type": "RECONNECT",
                    "error_code": "RECONNECT_FAILED",
                    "message": "Reconnection failed: invalid token or session expired",
                    "details": {},
                })
                await websocket.close()
                return

            session, conn, slot_idx = result
            player_id = conn.player_id
            is_reconnect = True

            logger.info("[WS] RECONNECT success room=%s player=%s slot=%d",
                         room_id, player_id, slot_idx)

            # Send reconnect success
            await websocket.send_json({
                "type": "RECONNECT_SUCCESS",
                "player_slot": slot_idx,
                "room_id": room_id,
                "your_player_id": player_id,
            })

            # Re-sync full game state
            if session.game_state:
                from .protocol.serializers import serialize_for_player
                state = serialize_for_player(session.game_state, player_id)
                await websocket.send_json({
                    "type": "STATE_UPDATE",
                    "state": state,
                })

            # Notify opponent
            await room_manager._notify_reconnect(session, player_id)

        else:
            # --- Normal join flow ---
            player_name = data.get("player_name", "Player")
            deck = data.get("deck", {})
            logger.info("[WS] JOIN_ROOM room=%s player_name=%s deck_keys=%s",
                         room_id, player_name,
                         list(deck.keys()) if isinstance(deck, dict) else "BAD")

            session, player_id, slot_idx, reconnect_token = await room_manager.join_room(
                room_id, player_name, deck, websocket
            )
            logger.info("[WS] Player joined room=%s player_id=%s slot=%d count=%d/2",
                         room_id, player_id, slot_idx, session.player_count)

            # Acknowledge join (includes reconnect token)
            await websocket.send_json({
                "type": "ROOM_JOINED",
                "player_slot": slot_idx,
                "room_id": room_id,
                "reconnect_token": reconnect_token,
            })

            # If room is now full, start the game
            if session.is_full:
                logger.info("[WS] Room %s is full -- starting game", room_id)
                try:
                    await room_manager.start_game(session)
                except Exception:
                    logger.exception("[WS] FAILED to start game in room=%s", room_id)
                    raise
                # Notify both players
                for slot in session.slots:
                    if slot and slot.connected and slot.websocket:
                        await slot.websocket.send_json({
                            "type": "GAME_STARTED",
                            "your_player_id": slot.player_id,
                        })
                # Send initial state
                await room_manager._broadcast_state(session)
                logger.info("[WS] Initial state broadcast complete for room=%s phase=%s",
                             room_id,
                             session.game_state.phase.value if session.game_state else "NONE")
            else:
                logger.info("[WS] Waiting for opponent in room=%s", room_id)
                await websocket.send_json({"type": "WAITING_FOR_OPPONENT"})

        # Main message loop (shared for both join and reconnect)
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "UNKNOWN")
            logger.info("[WS] ACTION room=%s player=%s type=%s payload_keys=%s",
                         room_id, player_id, msg_type,
                         [k for k in data.keys() if k != "type"])
            try:
                await room_manager.handle_action(session, player_id, data)
            except Exception:
                logger.exception(
                    "[WS] EXCEPTION handling action %s in room=%s player=%s",
                    msg_type, room_id, player_id,
                )
                # Structured error with context about what failed
                gs = session.game_state if session else None
                await websocket.send_json({
                    "type": "ERROR",
                    "action_type": msg_type,
                    "error_code": "INTERNAL_ERROR",
                    "message": f"Internal server error while processing {msg_type}",
                    "details": {
                        "phase": gs.phase.value if gs else None,
                        "turn_number": gs.turn_number if gs else None,
                    },
                })

    except WebSocketDisconnect:
        logger.info("[WS] Disconnected room=%s player=%s", room_id, player_id)
    except ValueError as e:
        logger.error("[WS] ValueError room=%s player=%s: %s", room_id, player_id, e)
        try:
            await websocket.send_json({
                "type": "ERROR",
                "action_type": "",
                "error_code": "INVALID_REQUEST",
                "message": str(e),
                "details": {},
            })
            await websocket.close()
        except Exception:
            pass
    except Exception:
        logger.exception("[WS] Unhandled exception room=%s player=%s", room_id, player_id)
    finally:
        if player_id and session:
            # If game is in progress, use grace period; otherwise remove immediately
            if session.game_state and not session.game_state.game_over:
                room_manager.mark_disconnected(room_id, player_id)
                logger.info("[WS] Player %s marked disconnected in room=%s (60s grace period)",
                             player_id, room_id)
            else:
                room_manager.remove_player(room_id, player_id)
                logger.info("[WS] Removed player=%s from room=%s", player_id, room_id)


# ---------------------------------------------------------------------------
# Serve built frontend (must be last — catches everything not matched above)
# ---------------------------------------------------------------------------

CARD_IMAGES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "cards", "images")
)
logger.info(f"CHECKING CARD_IMAGES_DIR: {CARD_IMAGES_DIR} - Exists? {os.path.isdir(CARD_IMAGES_DIR)}")
if os.path.isdir(CARD_IMAGES_DIR):
    app.mount("/card-images", StaticFiles(directory=CARD_IMAGES_DIR), name="card-images")
else:
    logger.error("CARD_IMAGES_DIR NOT FOUND!")

if os.path.isdir(FRONTEND_DIST):
    # Serve static assets (JS/CSS/images)
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes."""
        # Serve specific static files if they exist (e.g. favicon, robots.txt)
        requested = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(requested):
            return FileResponse(requested)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
