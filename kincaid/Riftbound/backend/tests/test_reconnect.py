"""IN-02: Player reconnection handling tests.

Tests the reconnection flow:
- (a) Slot reserved for 60s after disconnect
- (b) Reconnection via reconnect token
- (c) Full game state re-sent on reconnect
- Edge cases: invalid token, expired session, double disconnect
"""

from __future__ import annotations

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from starlette.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from app.engine.card_db import CardDB
from app.main import app
from app.room_manager import (
    GameSession,
    PlayerConnection,
    RoomManager,
    RECONNECT_TIMEOUT_SECONDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module", autouse=True)
def _load_card_db():
    """Ensure CardDB is populated before any test in this module."""
    CardDB.load_all(DATA_DIR)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _make_deck():
    """Return a minimal deck payload for joining a room."""
    all_cards = CardDB.all_cards()
    units = [c.card_id for c in all_cards.values() if c.card_type.value == "unit"]
    spells = [c.card_id for c in all_cards.values() if c.card_type.value == "spell"]
    runes = [c.card_id for c in all_cards.values() if c.card_type.value == "rune"]
    legends = [c.card_id for c in all_cards.values() if c.card_type.value == "legend"]
    battlefields = [c.card_id for c in all_cards.values() if c.card_type.value == "battlefield"]

    main_pool = units + spells
    main_deck = (main_pool * ((40 // len(main_pool)) + 1))[:40] if main_pool else ["fake"] * 40
    rune_deck = (runes * ((12 // max(len(runes), 1)) + 1))[:12] if runes else ["fake"] * 12
    bf_list = (battlefields * 3)[:3] if battlefields else ["fake_bf"] * 3

    return {
        "legend_id": legends[0] if legends else "fake_legend",
        "champion_id": units[0] if units else "fake_champion",
        "main_deck": main_deck,
        "rune_deck": rune_deck,
        "battlefields": bf_list,
    }


def _setup_two_player_game(client):
    """Helper: create room, join two players, drain setup messages.

    Returns (room_id, ws1, ws2, p1_id, p2_id, p1_token, p2_token).
    Must be called inside nested with-blocks for ws1 and ws2.
    """
    create_resp = client.post("/rooms")
    return create_resp.json()["room_id"]


# ---------------------------------------------------------------------------
# Unit tests: RoomManager reconnection logic
# ---------------------------------------------------------------------------


class TestRoomManagerDisconnect:
    """Test mark_disconnected keeps slot reserved."""

    def test_mark_disconnected_preserves_slot(self):
        """After mark_disconnected, the player slot still exists."""
        rm = RoomManager()
        room_id = rm.create_room()
        session = rm.get_room(room_id)

        # Manually add a player
        conn = PlayerConnection(
            player_id="p1",
            display_name="Alice",
            websocket=MagicMock(),
            reconnect_token="tok-abc-123",
            connected=True,
        )
        session.slots[0] = conn

        # Mark disconnected
        rm.mark_disconnected(room_id, "p1")

        # Slot is preserved
        assert session.slots[0] is not None
        assert session.slots[0].player_id == "p1"
        assert session.slots[0].connected is False
        assert session.slots[0].websocket is None

    def test_mark_disconnected_starts_expiry_task(self):
        """mark_disconnected creates a disconnect_task."""
        rm = RoomManager()
        room_id = rm.create_room()
        session = rm.get_room(room_id)

        conn = PlayerConnection(
            player_id="p1",
            display_name="Alice",
            websocket=MagicMock(),
            reconnect_token="tok-abc-123",
            connected=True,
        )
        session.slots[0] = conn

        rm.mark_disconnected(room_id, "p1")

        assert conn.disconnect_task is not None
        # Clean up the task
        conn.disconnect_task.cancel()

    def test_remove_player_clears_slot(self):
        """remove_player completely removes the slot."""
        rm = RoomManager()
        room_id = rm.create_room()
        session = rm.get_room(room_id)

        conn = PlayerConnection(
            player_id="p1",
            display_name="Alice",
            websocket=MagicMock(),
            reconnect_token="tok-abc-123",
            connected=True,
        )
        session.slots[0] = conn

        rm.remove_player(room_id, "p1")

        assert session.slots[0] is None


class TestRoomManagerReconnect:
    """Test reconnect_player restores connection."""

    @pytest.mark.asyncio
    async def test_reconnect_with_valid_token(self):
        """A disconnected player can reconnect with their token."""
        rm = RoomManager()
        room_id = rm.create_room()
        session = rm.get_room(room_id)

        conn = PlayerConnection(
            player_id="p1",
            display_name="Alice",
            websocket=None,
            reconnect_token="tok-valid-123",
            connected=False,
        )
        session.slots[0] = conn

        new_ws = AsyncMock()
        result = await rm.reconnect_player(room_id, "tok-valid-123", new_ws)

        assert result is not None
        returned_session, returned_conn, slot_idx = result
        assert returned_session is session
        assert returned_conn is conn
        assert slot_idx == 0
        assert conn.connected is True
        assert conn.websocket is new_ws

    @pytest.mark.asyncio
    async def test_reconnect_with_invalid_token(self):
        """An invalid token returns None."""
        rm = RoomManager()
        room_id = rm.create_room()
        session = rm.get_room(room_id)

        conn = PlayerConnection(
            player_id="p1",
            display_name="Alice",
            websocket=None,
            reconnect_token="tok-valid-123",
            connected=False,
        )
        session.slots[0] = conn

        new_ws = AsyncMock()
        result = await rm.reconnect_player(room_id, "tok-wrong-999", new_ws)
        assert result is None

    @pytest.mark.asyncio
    async def test_reconnect_when_already_connected(self):
        """Cannot reconnect if the player is already connected."""
        rm = RoomManager()
        room_id = rm.create_room()
        session = rm.get_room(room_id)

        conn = PlayerConnection(
            player_id="p1",
            display_name="Alice",
            websocket=MagicMock(),
            reconnect_token="tok-valid-123",
            connected=True,
        )
        session.slots[0] = conn

        new_ws = AsyncMock()
        result = await rm.reconnect_player(room_id, "tok-valid-123", new_ws)
        assert result is None

    @pytest.mark.asyncio
    async def test_reconnect_cancels_expiry_task(self):
        """Reconnecting cancels the pending disconnect timer."""
        rm = RoomManager()
        room_id = rm.create_room()
        session = rm.get_room(room_id)

        conn = PlayerConnection(
            player_id="p1",
            display_name="Alice",
            websocket=MagicMock(),
            reconnect_token="tok-valid-123",
            connected=True,
        )
        session.slots[0] = conn

        # Mark disconnected to start the timer
        rm.mark_disconnected(room_id, "p1")
        assert conn.disconnect_task is not None
        task = conn.disconnect_task

        # Reconnect
        new_ws = AsyncMock()
        result = await rm.reconnect_player(room_id, "tok-valid-123", new_ws)

        assert result is not None
        # Task is in cancelling state (cancel() was called on it).
        # It will become cancelled() once it actually runs the CancelledError.
        assert task.cancelling() > 0 or task.cancelled()
        # The disconnect_task field should be cleared
        assert conn.disconnect_task is None

    @pytest.mark.asyncio
    async def test_reconnect_nonexistent_room(self):
        """Reconnect to a room that does not exist returns None."""
        rm = RoomManager()
        new_ws = AsyncMock()
        result = await rm.reconnect_player("no-such-room", "tok-123", new_ws)
        assert result is None


class TestGameSessionHelpers:
    """Test GameSession helper methods for reconnection."""

    def test_get_connection_by_token_found(self):
        session = GameSession(room_id="test")
        conn = PlayerConnection(
            player_id="p1",
            display_name="Alice",
            websocket=None,
            reconnect_token="tok-abc",
        )
        session.slots[0] = conn
        assert session.get_connection_by_token("tok-abc") is conn

    def test_get_connection_by_token_not_found(self):
        session = GameSession(room_id="test")
        conn = PlayerConnection(
            player_id="p1",
            display_name="Alice",
            websocket=None,
            reconnect_token="tok-abc",
        )
        session.slots[0] = conn
        assert session.get_connection_by_token("tok-wrong") is None

    def test_connected_count(self):
        session = GameSession(room_id="test")
        session.slots[0] = PlayerConnection(
            player_id="p1", display_name="A", websocket=None, connected=True,
        )
        session.slots[1] = PlayerConnection(
            player_id="p2", display_name="B", websocket=None, connected=False,
        )
        assert session.connected_count == 1
        assert session.player_count == 2


class TestReconnectTimeout:
    """Test the 60-second timeout configuration."""

    def test_timeout_constant_is_60(self):
        assert RECONNECT_TIMEOUT_SECONDS == 60


# ---------------------------------------------------------------------------
# Integration tests: WebSocket reconnection flow
# ---------------------------------------------------------------------------


class TestWebSocketReconnectToken:
    """Test that ROOM_JOINED includes a reconnect_token."""

    def test_room_joined_includes_reconnect_token(self, client: TestClient):
        """When a player joins, ROOM_JOINED contains a reconnect_token."""
        room_id = client.post("/rooms").json()["room_id"]

        with client.websocket_connect(f"/ws/{room_id}") as ws:
            ws.send_json({
                "type": "JOIN_ROOM",
                "player_name": "Alice",
                "deck": _make_deck(),
            })
            msg = ws.receive_json()
            assert msg["type"] == "ROOM_JOINED"
            assert "reconnect_token" in msg
            assert isinstance(msg["reconnect_token"], str)
            assert len(msg["reconnect_token"]) > 0


class TestWebSocketReconnectFlow:
    """Test the full reconnect flow via WebSocket."""

    def test_reconnect_after_disconnect(self, client: TestClient):
        """Player disconnects and reconnects with token, gets RECONNECT_SUCCESS + STATE_UPDATE."""
        room_id = client.post("/rooms").json()["room_id"]

        # Player 1 joins
        with client.websocket_connect(f"/ws/{room_id}") as ws1:
            ws1.send_json({
                "type": "JOIN_ROOM",
                "player_name": "Alice",
                "deck": _make_deck(),
            })
            p1_joined = ws1.receive_json()
            assert p1_joined["type"] == "ROOM_JOINED"
            p1_token = p1_joined["reconnect_token"]
            ws1.receive_json()  # WAITING_FOR_OPPONENT

            # Player 2 joins - game starts
            with client.websocket_connect(f"/ws/{room_id}") as ws2:
                ws2.send_json({
                    "type": "JOIN_ROOM",
                    "player_name": "Bob",
                    "deck": _make_deck(),
                })
                ws2.receive_json()  # ROOM_JOINED
                ws2.receive_json()  # GAME_STARTED
                ws2.receive_json()  # STATE_UPDATE

                # Drain P1 game start messages
                ws1.receive_json()  # GAME_STARTED
                ws1.receive_json()  # STATE_UPDATE

            # ws2 is now disconnected (context manager exited)
            # P2 gets marked as disconnected with grace period

            # P1 should receive PLAYER_DISCONNECTED notification
            p1_disconnect_msg = ws1.receive_json()
            assert p1_disconnect_msg["type"] == "PLAYER_DISCONNECTED"

            # Now P1 disconnects too (ws1 context will exit)
        # ws1 is now disconnected

        # P1 reconnects with their token
        with client.websocket_connect(f"/ws/{room_id}") as ws1_new:
            ws1_new.send_json({
                "type": "RECONNECT",
                "reconnect_token": p1_token,
            })

            # Should get RECONNECT_SUCCESS
            reconnect_msg = ws1_new.receive_json()
            assert reconnect_msg["type"] == "RECONNECT_SUCCESS"
            assert reconnect_msg["room_id"] == room_id
            assert "your_player_id" in reconnect_msg

            # Should get STATE_UPDATE with full game state
            state_msg = ws1_new.receive_json()
            assert state_msg["type"] == "STATE_UPDATE"
            assert "state" in state_msg
            state = state_msg["state"]
            assert "phase" in state
            assert "you" in state
            assert "opponent" in state

    def test_reconnect_with_invalid_token_fails(self, client: TestClient):
        """Reconnect with a bad token returns ERROR and closes."""
        room_id = client.post("/rooms").json()["room_id"]

        # Set up a full game so the room persists after disconnect
        with client.websocket_connect(f"/ws/{room_id}") as ws1:
            ws1.send_json({
                "type": "JOIN_ROOM",
                "player_name": "Alice",
                "deck": _make_deck(),
            })
            ws1.receive_json()  # ROOM_JOINED
            ws1.receive_json()  # WAITING_FOR_OPPONENT

            with client.websocket_connect(f"/ws/{room_id}") as ws2:
                ws2.send_json({
                    "type": "JOIN_ROOM",
                    "player_name": "Bob",
                    "deck": _make_deck(),
                })
                ws2.receive_json()  # ROOM_JOINED
                ws2.receive_json()  # GAME_STARTED
                ws2.receive_json()  # STATE_UPDATE
                ws1.receive_json()  # GAME_STARTED
                ws1.receive_json()  # STATE_UPDATE

            # ws2 disconnected (game in progress, slot reserved via grace period)
            ws1.receive_json()  # PLAYER_DISCONNECTED

        # ws1 also disconnected (game in progress, slot reserved)
        # Now try to reconnect with a fake token
        with client.websocket_connect(f"/ws/{room_id}") as ws_bad:
            ws_bad.send_json({
                "type": "RECONNECT",
                "reconnect_token": "totally-fake-token-that-does-not-exist",
            })
            error_msg = ws_bad.receive_json()
            assert error_msg["type"] == "ERROR"
            assert error_msg["error_code"] == "RECONNECT_FAILED"

    def test_reconnect_preserves_game_state(self, client: TestClient):
        """After reconnect, the game state reflects the current phase and hands."""
        room_id = client.post("/rooms").json()["room_id"]

        with client.websocket_connect(f"/ws/{room_id}") as ws1:
            ws1.send_json({
                "type": "JOIN_ROOM",
                "player_name": "Alice",
                "deck": _make_deck(),
            })
            p1_joined = ws1.receive_json()
            p1_token = p1_joined["reconnect_token"]
            ws1.receive_json()  # WAITING_FOR_OPPONENT

            with client.websocket_connect(f"/ws/{room_id}") as ws2:
                ws2.send_json({
                    "type": "JOIN_ROOM",
                    "player_name": "Bob",
                    "deck": _make_deck(),
                })
                p2_joined = ws2.receive_json()  # ROOM_JOINED
                p2_token = p2_joined["reconnect_token"]
                ws2.receive_json()  # GAME_STARTED
                p2_state = ws2.receive_json()  # STATE_UPDATE
                initial_phase = p2_state["state"]["phase"]

                ws1.receive_json()  # GAME_STARTED
                ws1.receive_json()  # STATE_UPDATE

        # Both disconnected. Reconnect P1
        with client.websocket_connect(f"/ws/{room_id}") as ws1_new:
            ws1_new.send_json({
                "type": "RECONNECT",
                "reconnect_token": p1_token,
            })
            reconnect_msg = ws1_new.receive_json()
            assert reconnect_msg["type"] == "RECONNECT_SUCCESS"

            state_msg = ws1_new.receive_json()
            assert state_msg["type"] == "STATE_UPDATE"

            # The game state should show the same phase as when we disconnected
            assert state_msg["state"]["phase"] == initial_phase
            # Player should see their hand
            assert "you" in state_msg["state"]
            assert "hand" in state_msg["state"]["you"]


class TestWebSocketDisconnectNotification:
    """Test that opponents are notified about disconnect/reconnect."""

    def test_opponent_notified_on_disconnect(self, client: TestClient):
        """When P2 disconnects, P1 receives PLAYER_DISCONNECTED."""
        room_id = client.post("/rooms").json()["room_id"]

        with client.websocket_connect(f"/ws/{room_id}") as ws1:
            ws1.send_json({
                "type": "JOIN_ROOM",
                "player_name": "Alice",
                "deck": _make_deck(),
            })
            ws1.receive_json()  # ROOM_JOINED
            ws1.receive_json()  # WAITING_FOR_OPPONENT

            with client.websocket_connect(f"/ws/{room_id}") as ws2:
                ws2.send_json({
                    "type": "JOIN_ROOM",
                    "player_name": "Bob",
                    "deck": _make_deck(),
                })
                ws2.receive_json()  # ROOM_JOINED
                ws2.receive_json()  # GAME_STARTED
                ws2.receive_json()  # STATE_UPDATE
                ws1.receive_json()  # GAME_STARTED
                ws1.receive_json()  # STATE_UPDATE

            # ws2 disconnected
            msg = ws1.receive_json()
            assert msg["type"] == "PLAYER_DISCONNECTED"
            assert "player_id" in msg

    def test_opponent_notified_on_reconnect(self, client: TestClient):
        """When P2 reconnects, P1 receives PLAYER_RECONNECTED."""
        room_id = client.post("/rooms").json()["room_id"]

        with client.websocket_connect(f"/ws/{room_id}") as ws1:
            ws1.send_json({
                "type": "JOIN_ROOM",
                "player_name": "Alice",
                "deck": _make_deck(),
            })
            p1_joined = ws1.receive_json()
            ws1.receive_json()  # WAITING_FOR_OPPONENT

            # P2 joins and then disconnects
            with client.websocket_connect(f"/ws/{room_id}") as ws2:
                ws2.send_json({
                    "type": "JOIN_ROOM",
                    "player_name": "Bob",
                    "deck": _make_deck(),
                })
                p2_joined = ws2.receive_json()  # ROOM_JOINED
                p2_token = p2_joined["reconnect_token"]
                ws2.receive_json()  # GAME_STARTED
                ws2.receive_json()  # STATE_UPDATE
                ws1.receive_json()  # GAME_STARTED
                ws1.receive_json()  # STATE_UPDATE

            # P2 disconnected - P1 gets notification
            disconnect_msg = ws1.receive_json()
            assert disconnect_msg["type"] == "PLAYER_DISCONNECTED"

            # P2 reconnects
            with client.websocket_connect(f"/ws/{room_id}") as ws2_new:
                ws2_new.send_json({
                    "type": "RECONNECT",
                    "reconnect_token": p2_token,
                })
                # P2 gets RECONNECT_SUCCESS + STATE_UPDATE
                ws2_new.receive_json()  # RECONNECT_SUCCESS
                ws2_new.receive_json()  # STATE_UPDATE

                # P1 should get PLAYER_RECONNECTED
                reconnect_msg = ws1.receive_json()
                assert reconnect_msg["type"] == "PLAYER_RECONNECTED"
