"""TS-01: API / WebSocket integration tests for Riftbound server."""

from __future__ import annotations

import sys
import os

# Ensure app package is importable (same pattern as conftest.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from starlette.testclient import TestClient

from app.engine.card_db import CardDB
from app.main import app


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
    """Synchronous Starlette TestClient that triggers lifespan events."""
    # TestClient handles lifespan automatically, but we pre-loaded CardDB above
    # so the lifespan will just reinitialize the RoomManager.
    with TestClient(app) as c:
        yield c


def _make_deck():
    """Return a minimal deck payload for joining a room."""
    # Grab some real card ids from the DB
    all_cards = CardDB.all_cards()
    units = [c.card_id for c in all_cards.values() if c.card_type.value == "unit"]
    spells = [c.card_id for c in all_cards.values() if c.card_type.value == "spell"]
    runes = [c.card_id for c in all_cards.values() if c.card_type.value == "rune"]
    legends = [c.card_id for c in all_cards.values() if c.card_type.value == "legend"]
    battlefields = [c.card_id for c in all_cards.values() if c.card_type.value == "battlefield"]

    # Build a 40-card main deck from available units+spells (cycle as needed)
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


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health_returns_200(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"


class TestCreateRoom:
    def test_create_room_returns_room_id(self, client: TestClient):
        resp = client.post("/rooms")
        assert resp.status_code == 200
        body = resp.json()
        assert "room_id" in body
        assert isinstance(body["room_id"], str)
        assert len(body["room_id"]) > 0


class TestGetRoom:
    def test_get_room_returns_info(self, client: TestClient):
        # Create a room first
        create_resp = client.post("/rooms")
        room_id = create_resp.json()["room_id"]

        resp = client.get(f"/rooms/{room_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["room_id"] == room_id
        assert body["player_count"] == 0
        assert body["game_started"] is False
        assert body["is_full"] is False

    def test_get_room_not_found(self, client: TestClient):
        resp = client.get("/rooms/nonexistent_bad_id")
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body


class TestListCards:
    def test_list_cards_returns_definitions(self, client: TestClient):
        resp = client.get("/cards")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)
        assert len(body) > 0
        # Check structure of an arbitrary card
        first_card = next(iter(body.values()))
        assert "card_id" in first_card
        assert "name" in first_card
        assert "card_type" in first_card


# ---------------------------------------------------------------------------
# WebSocket tests
# ---------------------------------------------------------------------------


class TestWebSocketJoinRoom:
    def test_ws_join_room(self, client: TestClient):
        """Single player joins a room and receives ROOM_JOINED."""
        create_resp = client.post("/rooms")
        room_id = create_resp.json()["room_id"]

        with client.websocket_connect(f"/ws/{room_id}") as ws:
            ws.send_json({
                "type": "JOIN_ROOM",
                "player_name": "Alice",
                "deck": _make_deck(),
            })
            msg = ws.receive_json()
            assert msg["type"] == "ROOM_JOINED"
            assert msg["room_id"] == room_id
            assert msg["player_slot"] == 0

            # First player also receives WAITING_FOR_OPPONENT
            msg2 = ws.receive_json()
            assert msg2["type"] == "WAITING_FOR_OPPONENT"

    def test_ws_two_players_join(self, client: TestClient):
        """Two players join the same room, both get ROOM_JOINED."""
        create_resp = client.post("/rooms")
        room_id = create_resp.json()["room_id"]

        with client.websocket_connect(f"/ws/{room_id}") as ws1:
            ws1.send_json({
                "type": "JOIN_ROOM",
                "player_name": "Alice",
                "deck": _make_deck(),
            })
            msg1 = ws1.receive_json()
            assert msg1["type"] == "ROOM_JOINED"
            assert msg1["player_slot"] == 0

            # First player waits
            wait_msg = ws1.receive_json()
            assert wait_msg["type"] == "WAITING_FOR_OPPONENT"

            with client.websocket_connect(f"/ws/{room_id}") as ws2:
                ws2.send_json({
                    "type": "JOIN_ROOM",
                    "player_name": "Bob",
                    "deck": _make_deck(),
                })
                msg2 = ws2.receive_json()
                assert msg2["type"] == "ROOM_JOINED"
                assert msg2["player_slot"] == 1

    def test_ws_game_starts_with_two_players(self, client: TestClient):
        """When two players join, GAME_STARTED and STATE_UPDATE are sent."""
        create_resp = client.post("/rooms")
        room_id = create_resp.json()["room_id"]

        with client.websocket_connect(f"/ws/{room_id}") as ws1:
            ws1.send_json({
                "type": "JOIN_ROOM",
                "player_name": "Alice",
                "deck": _make_deck(),
            })
            # Consume ROOM_JOINED + WAITING_FOR_OPPONENT
            ws1.receive_json()
            ws1.receive_json()

            with client.websocket_connect(f"/ws/{room_id}") as ws2:
                ws2.send_json({
                    "type": "JOIN_ROOM",
                    "player_name": "Bob",
                    "deck": _make_deck(),
                })
                # ws2 gets ROOM_JOINED
                join_msg = ws2.receive_json()
                assert join_msg["type"] == "ROOM_JOINED"

                # ws2 gets GAME_STARTED
                started_msg = ws2.receive_json()
                assert started_msg["type"] == "GAME_STARTED"
                assert "your_player_id" in started_msg

                # ws2 gets STATE_UPDATE
                state_msg = ws2.receive_json()
                assert state_msg["type"] == "STATE_UPDATE"
                assert "state" in state_msg

                # ws1 also gets GAME_STARTED + STATE_UPDATE
                p1_started = ws1.receive_json()
                assert p1_started["type"] == "GAME_STARTED"
                assert "your_player_id" in p1_started

                p1_state = ws1.receive_json()
                assert p1_state["type"] == "STATE_UPDATE"

    def test_ws_invalid_action_returns_rejection(self, client: TestClient):
        """Sending an unknown action type after game starts returns rejection."""
        create_resp = client.post("/rooms")
        room_id = create_resp.json()["room_id"]

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
                # Drain join messages for ws2
                ws2.receive_json()  # ROOM_JOINED
                ws2.receive_json()  # GAME_STARTED
                ws2.receive_json()  # STATE_UPDATE

                # Drain game-start messages for ws1
                ws1.receive_json()  # GAME_STARTED
                ws1.receive_json()  # STATE_UPDATE

                # Send an unknown action type from ws1
                ws1.send_json({"type": "TOTALLY_FAKE_ACTION"})
                error_msg = ws1.receive_json()
                # The server sends an ERROR for unknown actions
                assert error_msg["type"] == "ERROR"
