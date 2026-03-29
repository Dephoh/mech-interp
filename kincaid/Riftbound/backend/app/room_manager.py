"""In-memory room registry: maps room_id to GameSession."""

from __future__ import annotations

import logging

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from .engine.action_executor import execute_action
from .engine.action_validator import validate_action
from .engine.card_db import CardDB
from .engine.enums import ActionType
from .engine.game_state import GameState, create_game
from .protocol.serializers import serialize_for_player

logger = logging.getLogger("riftbound.room")


RECONNECT_TIMEOUT_SECONDS = 60


@dataclass
class PlayerConnection:
    player_id: str
    display_name: str
    websocket: WebSocket | None
    deck: dict[str, Any] | None = None  # raw deck payload
    reconnect_token: str = ""  # UUID for reconnection
    connected: bool = True  # False when WS dropped but slot reserved
    disconnect_task: asyncio.Task | None = field(default=None, repr=False)


@dataclass
class GameSession:
    room_id: str
    slots: list[PlayerConnection | None] = field(default_factory=lambda: [None, None])
    game_state: GameState | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_full(self) -> bool:
        return all(s is not None for s in self.slots)

    @property
    def player_count(self) -> int:
        return sum(1 for s in self.slots if s is not None)

    @property
    def connected_count(self) -> int:
        return sum(1 for s in self.slots if s is not None and s.connected)

    def get_slot(self, player_id: str) -> int | None:
        for i, s in enumerate(self.slots):
            if s and s.player_id == player_id:
                return i
        return None

    def get_connection(self, player_id: str) -> PlayerConnection | None:
        for s in self.slots:
            if s and s.player_id == player_id:
                return s
        return None

    def get_connection_by_token(self, token: str) -> PlayerConnection | None:
        for s in self.slots:
            if s and s.reconnect_token == token:
                return s
        return None


class RoomManager:
    def __init__(self) -> None:
        self.rooms: dict[str, GameSession] = {}

    def create_room(self) -> str:
        room_id = str(uuid.uuid4())[:8]
        session = GameSession(room_id=room_id)
        self.rooms[room_id] = session
        return room_id

    def get_room(self, room_id: str) -> GameSession | None:
        return self.rooms.get(room_id)

    async def join_room(
        self,
        room_id: str,
        player_name: str,
        deck: dict[str, Any],
        websocket: WebSocket,
    ) -> tuple[GameSession, str, int, str]:
        """
        Join a room. Returns (session, player_id, slot_index, reconnect_token).
        Creates the room if it doesn't exist.
        """
        if room_id not in self.rooms:
            self.rooms[room_id] = GameSession(room_id=room_id)

        session = self.rooms[room_id]

        # Find open slot
        slot_idx = None
        for i, s in enumerate(session.slots):
            if s is None:
                slot_idx = i
                break

        if slot_idx is None:
            raise ValueError("Room is full")

        player_id = str(uuid.uuid4())[:12]
        reconnect_token = str(uuid.uuid4())
        conn = PlayerConnection(
            player_id=player_id,
            display_name=player_name,
            websocket=websocket,
            deck=deck,
            reconnect_token=reconnect_token,
            connected=True,
        )
        session.slots[slot_idx] = conn

        return session, player_id, slot_idx, reconnect_token

    async def reconnect_player(
        self,
        room_id: str,
        reconnect_token: str,
        websocket: WebSocket,
    ) -> tuple[GameSession, PlayerConnection, int] | None:
        """
        Attempt to reconnect a player using their reconnect token.
        Returns (session, connection, slot_index) on success, or None on failure.
        """
        session = self.rooms.get(room_id)
        if not session:
            return None

        conn = session.get_connection_by_token(reconnect_token)
        if not conn or conn.connected:
            return None

        slot_idx = session.get_slot(conn.player_id)
        if slot_idx is None:
            return None

        # Cancel the pending disconnect timer
        if conn.disconnect_task and not conn.disconnect_task.done():
            conn.disconnect_task.cancel()
            conn.disconnect_task = None

        # Restore connection
        conn.websocket = websocket
        conn.connected = True
        logger.info("[RECONNECT] Player %s reconnected to room %s",
                     conn.player_id, room_id)

        return session, conn, slot_idx

    def mark_disconnected(self, room_id: str, player_id: str) -> None:
        """
        Mark a player as disconnected but keep their slot reserved.
        Starts a timer that will remove them after RECONNECT_TIMEOUT_SECONDS.
        """
        session = self.rooms.get(room_id)
        if not session:
            return

        conn = session.get_connection(player_id)
        if not conn:
            return

        conn.connected = False
        conn.websocket = None

        # Start expiry timer
        async def _expire_slot():
            try:
                await asyncio.sleep(RECONNECT_TIMEOUT_SECONDS)
                # Timer expired — fully remove the player
                logger.info("[RECONNECT] Timeout expired for player %s in room %s, removing",
                             player_id, room_id)
                self._force_remove_player(room_id, player_id)
            except asyncio.CancelledError:
                # Reconnect happened before expiry
                pass

        conn.disconnect_task = asyncio.ensure_future(_expire_slot())

        # Notify opponent that this player disconnected
        asyncio.ensure_future(self._notify_disconnect(session, player_id))

    async def _notify_disconnect(self, session: GameSession, player_id: str) -> None:
        """Notify remaining connected players that someone disconnected."""
        for slot in session.slots:
            if slot and slot.connected and slot.player_id != player_id and slot.websocket:
                try:
                    await slot.websocket.send_json({
                        "type": "PLAYER_DISCONNECTED",
                        "player_id": player_id,
                    })
                except Exception:
                    pass

    async def _notify_reconnect(self, session: GameSession, player_id: str) -> None:
        """Notify remaining connected players that someone reconnected."""
        for slot in session.slots:
            if slot and slot.connected and slot.player_id != player_id and slot.websocket:
                try:
                    await slot.websocket.send_json({
                        "type": "PLAYER_RECONNECTED",
                        "player_id": player_id,
                    })
                except Exception:
                    pass

    async def start_game(self, session: GameSession) -> None:
        """Initialize the GameState once both players have joined."""
        if session.game_state is not None:
            return

        configs = []
        for slot in session.slots:
            if slot and slot.deck:
                configs.append({
                    "player_id": slot.player_id,
                    "display_name": slot.display_name,
                    "legend_id": slot.deck["legend_id"],
                    "champion_id": slot.deck["champion_id"],
                    "main_deck": slot.deck["main_deck"],
                    "rune_deck": slot.deck["rune_deck"],
                    "battlefields": slot.deck["battlefields"],
                })

        card_defs = CardDB.all_cards()
        session.game_state = create_game(session.room_id, configs, card_defs)

    async def handle_action(
        self, session: GameSession, player_id: str, data: dict[str, Any]
    ) -> None:
        """Validate and execute an action, then broadcast state."""
        gs = session.game_state
        msg_type = data.get("type", "")

        if not gs:
            logger.warning("[ACTION] Game not started for room=%s", session.room_id)
            await self._send_error(
                session, player_id, "Game not started",
                action_type=msg_type, error_code="GAME_NOT_STARTED",
            )
            return

        action_type = _msg_type_to_action(msg_type)
        if not action_type:
            logger.warning("[ACTION] Unknown msg type=%s from player=%s", msg_type, player_id)
            await self._send_error(
                session, player_id, f"Unknown message type: {msg_type}",
                action_type=msg_type, error_code="UNKNOWN_ACTION",
            )
            return

        # Build payload from message data
        payload = {k: v for k, v in data.items() if k != "type"}

        async with session.lock:
            # Validate
            result = validate_action(gs, player_id, action_type, payload)
            if not result.ok:
                logger.info("[ACTION] REJECTED %s from player=%s: %s (phase=%s)",
                            action_type.value, player_id, result.error, gs.phase.value)
                await self._send_to_player(session, player_id, {
                    "type": "ACTION_REJECTED",
                    "reason": result.error,
                })
                return

            # Execute
            logger.info("[ACTION] EXECUTE %s from player=%s (phase=%s)",
                        action_type.value, player_id, gs.phase.value)
            try:
                logs = execute_action(gs, player_id, action_type, payload)
            except Exception:
                logger.exception(
                    "[ACTION] EXCEPTION during execute %s room=%s player=%s phase=%s",
                    action_type.value, session.room_id, player_id, gs.phase.value,
                )
                await self._send_error(
                    session, player_id,
                    f"Internal error while executing {msg_type}",
                    action_type=msg_type,
                    error_code="EXECUTION_ERROR",
                    details={
                        "phase": gs.phase.value,
                        "turn_number": gs.turn_number,
                    },
                )
                return
            logger.info("[ACTION] Phase after=%s logs=%s", gs.phase.value, logs)

            # Broadcast updated state to both players
            await self._broadcast_state(session)

            # Send logs
            if logs:
                await self._broadcast(session, {
                    "type": "GAME_LOG",
                    "entries": [{"message": m} for m in logs],
                })

            # Send CHOICE_REQUIRED if an effect is waiting for player input
            if gs.pending_choice:
                choice = gs.pending_choice
                await self._send_to_player(session, choice["controller_id"], {
                    "type": "CHOICE_REQUIRED",
                    "prompt": choice.get("prompt", "Make a choice"),
                    "options": choice.get("options", []),
                    "target": choice.get("target"),
                    "min_choices": choice.get("min_choices", 1),
                    "max_choices": choice.get("max_choices", 1),
                    "controller_id": choice["controller_id"],
                })

            # Check game over
            if gs.game_over:
                logger.info("[ACTION] GAME OVER room=%s winner=%s", session.room_id, gs.winner_id)
                await self._broadcast(session, {
                    "type": "GAME_OVER",
                    "winner_id": gs.winner_id or "",
                    "reason": "Game ended",
                })

            # Auto-pass for testlab P2
            testlab = getattr(session, "_testlab", None)
            if testlab and gs and not gs.game_over:
                await self._testlab_auto_pass(session, gs)

    async def _testlab_auto_pass(self, session: GameSession, gs: GameState) -> None:
        """Auto-pass for testlab P2 opponent.

        When it is P2's ACTION phase and contested battlefields exist,
        P2 will advance phase which (after this fix) triggers showdowns.
        During showdowns P1 gets focus windows where they can play
        ambush units and reaction spells.
        """
        from .engine.enums import Domain, Phase
        from .engine.combat import open_showdown, start_combat

        p2_id = "testlab-p2"
        max_iter = 40
        did_something = False

        for _ in range(max_iter):
            if gs.game_over:
                break
            if gs.active_player_id != p2_id:
                break
            # Pass on chain
            if not gs.chain.is_empty:
                execute_action(gs, p2_id, ActionType.PASS_PRIORITY, {})
                did_something = True
                continue
            # Pass on showdown focus
            if gs.active_showdown and gs.active_showdown.focus_player_id == p2_id:
                execute_action(gs, p2_id, ActionType.PASS_FOCUS, {})
                did_something = True
                continue
            # Handle staged combats: start combat at the first staged battlefield
            staged_combat_bf = None
            for bf in gs.battlefields.values():
                if bf.combat_staged:
                    staged_combat_bf = bf
                    break
            if staged_combat_bf:
                logs = start_combat(gs, staged_combat_bf.battlefield_id)
                did_something = True
                continue
            # During P2's ACTION phase, start showdowns at contested battlefields
            # so P1 gets reaction windows (for ambush, reaction spells, etc.)
            if (gs.turn_player_id == p2_id
                    and gs.phase == Phase.ACTION
                    and not gs.active_showdown
                    and not gs.active_combat):
                contested_bf = None
                for bf in gs.battlefields.values():
                    units_by = bf.units_by_player(gs.instances)
                    if len(units_by) >= 2:
                        contested_bf = bf
                        break
                if contested_bf:
                    logs = open_showdown(gs, contested_bf.battlefield_id, p2_id)
                    did_something = True
                    continue
            # If it's P2's turn, advance phase
            if gs.turn_player_id == p2_id:
                execute_action(gs, p2_id, ActionType.ADVANCE_PHASE, {})
                did_something = True
                continue
            break

        if did_something:
            # Replenish P1 resources
            p1 = gs.players.get("testlab-p1")
            if p1:
                p1.rune_pool.energy = 99
                for dom in Domain:
                    p1.rune_pool.power[dom] = 99
            await self._broadcast_state(session)

    async def _broadcast_state(self, session: GameSession) -> None:
        """Send per-player serialized state to each connected player."""
        gs = session.game_state
        if not gs:
            return
        for slot in session.slots:
            if slot and slot.connected and slot.websocket:
                state = serialize_for_player(gs, slot.player_id)
                await self._send_to_player(session, slot.player_id, {
                    "type": "STATE_UPDATE",
                    "state": state,
                })

    async def _broadcast(self, session: GameSession, msg: dict[str, Any]) -> None:
        for slot in session.slots:
            if slot and slot.connected and slot.websocket:
                try:
                    await slot.websocket.send_json(msg)
                except Exception:
                    pass

    async def _send_to_player(
        self, session: GameSession, player_id: str, msg: dict[str, Any]
    ) -> None:
        conn = session.get_connection(player_id)
        if conn and conn.connected and conn.websocket:
            try:
                await conn.websocket.send_json(msg)
            except Exception:
                pass

    async def _send_error(
        self,
        session: GameSession,
        player_id: str,
        message: str,
        *,
        action_type: str = "",
        error_code: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        await self._send_to_player(session, player_id, {
            "type": "ERROR",
            "action_type": action_type,
            "error_code": error_code,
            "message": message,
            "details": details or {},
        })

    def _force_remove_player(self, room_id: str, player_id: str) -> None:
        """Unconditionally remove a player slot and clean up empty rooms."""
        session = self.rooms.get(room_id)
        if not session:
            return
        for i, slot in enumerate(session.slots):
            if slot and slot.player_id == player_id:
                # Cancel any pending disconnect timer
                if slot.disconnect_task and not slot.disconnect_task.done():
                    slot.disconnect_task.cancel()
                session.slots[i] = None
                break
        # Clean up empty rooms
        if session.player_count == 0:
            del self.rooms[room_id]

    def remove_player(self, room_id: str, player_id: str) -> None:
        """Remove player immediately (no reconnect grace period)."""
        self._force_remove_player(room_id, player_id)


def _msg_type_to_action(msg_type: str) -> ActionType | None:
    mapping = {
        "MULLIGAN_CHOICE": ActionType.MULLIGAN_CHOICE,
        "ADVANCE_PHASE": ActionType.ADVANCE_PHASE,
        "PASS_PRIORITY": ActionType.PASS_PRIORITY,
        "PASS_FOCUS": ActionType.PASS_FOCUS,
        "PLAY_CARD": ActionType.PLAY_CARD,
        "ACTIVATE_ABILITY": ActionType.ACTIVATE_ABILITY,
        "MOVE_UNIT": ActionType.MOVE_UNIT,
        "EXHAUST_RUNE": ActionType.EXHAUST_RUNE,
        "RECYCLE_RUNE": ActionType.RECYCLE_RUNE,
        "HIDE_CARD": ActionType.HIDE_CARD,
        "ASSIGN_DAMAGE": ActionType.ASSIGN_DAMAGE,
        "CONCEDE": ActionType.CONCEDE,
        "SUBMIT_CHOICE": ActionType.SUBMIT_CHOICE,
    }
    return mapping.get(msg_type)
