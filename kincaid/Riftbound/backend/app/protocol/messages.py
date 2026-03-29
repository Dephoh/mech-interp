"""Pydantic models for all WebSocket message types."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Client → Server messages
# ---------------------------------------------------------------------------


class JoinRoom(BaseModel):
    type: Literal["JOIN_ROOM"] = "JOIN_ROOM"
    room_id: str
    player_name: str
    deck: DeckPayload


class DeckPayload(BaseModel):
    legend_id: str
    champion_id: str
    main_deck: list[str]  # card_ids (40)
    rune_deck: list[str]  # card_ids (12)
    battlefields: list[str]  # card_ids (3)


class MulliganChoice(BaseModel):
    type: Literal["MULLIGAN_CHOICE"] = "MULLIGAN_CHOICE"
    keep_indices: list[int]


class AdvancePhase(BaseModel):
    type: Literal["ADVANCE_PHASE"] = "ADVANCE_PHASE"


class PassPriority(BaseModel):
    type: Literal["PASS_PRIORITY"] = "PASS_PRIORITY"


class PassFocus(BaseModel):
    type: Literal["PASS_FOCUS"] = "PASS_FOCUS"


class PlayCard(BaseModel):
    type: Literal["PLAY_CARD"] = "PLAY_CARD"
    instance_id: str
    targets: list[str] = Field(default_factory=list)
    pay_accelerate: bool = False
    destination: ZoneDestination | None = None  # where to place a unit (base or battlefield)


class ActivateAbility(BaseModel):
    type: Literal["ACTIVATE_ABILITY"] = "ACTIVATE_ABILITY"
    source_id: str
    ability_id: str
    targets: list[str] = Field(default_factory=list)


class MoveUnit(BaseModel):
    type: Literal["MOVE_UNIT"] = "MOVE_UNIT"
    instance_id: str
    destination: ZoneDestination


class ZoneDestination(BaseModel):
    zone: str  # "battlefield" | "base"
    id: str | None = None  # battlefield_id


class ExhaustRune(BaseModel):
    type: Literal["EXHAUST_RUNE"] = "EXHAUST_RUNE"
    instance_id: str


class RecycleRune(BaseModel):
    type: Literal["RECYCLE_RUNE"] = "RECYCLE_RUNE"
    instance_id: str


class HideCard(BaseModel):
    type: Literal["HIDE_CARD"] = "HIDE_CARD"
    instance_id: str
    battlefield_id: str


class AssignDamage(BaseModel):
    type: Literal["ASSIGN_DAMAGE"] = "ASSIGN_DAMAGE"
    assignments: dict[str, int]  # target_instance_id -> damage


class Concede(BaseModel):
    type: Literal["CONCEDE"] = "CONCEDE"


class Reconnect(BaseModel):
    type: Literal["RECONNECT"] = "RECONNECT"
    reconnect_token: str


class SubmitChoice(BaseModel):
    """Client response to a CHOICE_REQUIRED message."""

    type: Literal["SUBMIT_CHOICE"] = "SUBMIT_CHOICE"
    chosen_option_index: int | None = None        # for modal choices (pick an option)
    chosen_target_ids: list[str] = Field(default_factory=list)  # for target selection


# Union of all client messages
ClientMessage = (
    JoinRoom
    | MulliganChoice
    | AdvancePhase
    | PassPriority
    | PassFocus
    | PlayCard
    | ActivateAbility
    | MoveUnit
    | ExhaustRune
    | RecycleRune
    | HideCard
    | AssignDamage
    | Concede
    | Reconnect
    | SubmitChoice
)


def parse_client_message(data: dict[str, Any]) -> ClientMessage:
    """Parse a raw dict into the appropriate client message type."""
    msg_type = data.get("type")
    parsers = {
        "JOIN_ROOM": JoinRoom,
        "MULLIGAN_CHOICE": MulliganChoice,
        "ADVANCE_PHASE": AdvancePhase,
        "PASS_PRIORITY": PassPriority,
        "PASS_FOCUS": PassFocus,
        "PLAY_CARD": PlayCard,
        "ACTIVATE_ABILITY": ActivateAbility,
        "MOVE_UNIT": MoveUnit,
        "EXHAUST_RUNE": ExhaustRune,
        "RECYCLE_RUNE": RecycleRune,
        "HIDE_CARD": HideCard,
        "ASSIGN_DAMAGE": AssignDamage,
        "CONCEDE": Concede,
        "RECONNECT": Reconnect,
        "SUBMIT_CHOICE": SubmitChoice,
    }
    parser = parsers.get(msg_type)
    if not parser:
        raise ValueError(f"Unknown message type: {msg_type}")
    return parser.model_validate(data)


# ---------------------------------------------------------------------------
# Server → Client messages
# ---------------------------------------------------------------------------


class RoomJoined(BaseModel):
    type: Literal["ROOM_JOINED"] = "ROOM_JOINED"
    player_slot: int  # 0 or 1
    room_id: str
    reconnect_token: str = ""  # UUID token for reconnection


class GameStarted(BaseModel):
    type: Literal["GAME_STARTED"] = "GAME_STARTED"
    your_player_id: str


class StateUpdate(BaseModel):
    type: Literal["STATE_UPDATE"] = "STATE_UPDATE"
    state: dict[str, Any]  # serialized ClientGameState


class ActionRequired(BaseModel):
    type: Literal["ACTION_REQUIRED"] = "ACTION_REQUIRED"
    request: dict[str, Any]


class ActionRejected(BaseModel):
    type: Literal["ACTION_REJECTED"] = "ACTION_REJECTED"
    reason: str


class GameLog(BaseModel):
    type: Literal["GAME_LOG"] = "GAME_LOG"
    entries: list[dict[str, Any]]


class GameOver(BaseModel):
    type: Literal["GAME_OVER"] = "GAME_OVER"
    winner_id: str
    reason: str


class WaitingForOpponent(BaseModel):
    type: Literal["WAITING_FOR_OPPONENT"] = "WAITING_FOR_OPPONENT"


class ChoiceRequired(BaseModel):
    """Sent when the engine halts on a player_choice node and needs input."""

    type: Literal["CHOICE_REQUIRED"] = "CHOICE_REQUIRED"
    prompt: str
    options: list[dict[str, Any]] = Field(default_factory=list)
    target: dict[str, Any] | None = None
    min_choices: int = 1
    max_choices: int = 1
    controller_id: str = ""


class ErrorMessage(BaseModel):
    type: Literal["ERROR"] = "ERROR"
    action_type: str = ""  # what the player tried to do (e.g. "PLAY_CARD")
    error_code: str = ""  # machine-readable code (e.g. "INVALID_TARGET")
    message: str = ""  # human-readable description
    details: dict[str, Any] = Field(default_factory=dict)  # optional extra context


class ReconnectSuccess(BaseModel):
    type: Literal["RECONNECT_SUCCESS"] = "RECONNECT_SUCCESS"
    player_slot: int
    room_id: str
    your_player_id: str


class PlayerDisconnected(BaseModel):
    type: Literal["PLAYER_DISCONNECTED"] = "PLAYER_DISCONNECTED"
    player_id: str


class PlayerReconnected(BaseModel):
    type: Literal["PLAYER_RECONNECTED"] = "PLAYER_RECONNECTED"
    player_id: str


# Rebuild models that have forward references
JoinRoom.model_rebuild()
PlayCard.model_rebuild()
