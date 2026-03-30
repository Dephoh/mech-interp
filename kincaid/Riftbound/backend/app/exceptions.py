"""Structured exception hierarchy for the Riftbound server.

Each exception carries an ``error_code`` (machine-readable) and ``action_type``
(which client action triggered it) so the WebSocket handler can build a
structured JSON error response without guessing.

Hierarchy
---------
RiftboundError (base — never raised directly)
├── GameNotStartedError     — action received before game initialisation
├── UnknownActionError      — unrecognised message type
├── ActionValidationError   — rule-level rejection (e.g. wrong phase, wrong player)
├── ActionExecutionError    — unexpected failure during execute_action
├── RoomFullError           — room capacity exceeded
├── RoomNotFoundError       — room_id does not exist
└── GameCorruptedError      — state is irrecoverably bad (triggers snapshot)
"""

from __future__ import annotations

from typing import Any


class RiftboundError(Exception):
    """Base class for all Riftbound server errors."""

    error_code: str = "RIFTBOUND_ERROR"
    action_type: str = ""

    def __init__(
        self,
        message: str = "",
        *,
        action_type: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if action_type:
            self.action_type = action_type
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the structured error JSON sent over WebSocket."""
        return {
            "type": "ERROR",
            "action_type": self.action_type,
            "error_code": self.error_code,
            "message": str(self),
            "details": self.details,
        }


class GameNotStartedError(RiftboundError):
    error_code = "GAME_NOT_STARTED"


class UnknownActionError(RiftboundError):
    error_code = "UNKNOWN_ACTION"


class ActionValidationError(RiftboundError):
    """Raised when an action fails rule-level validation."""

    error_code = "VALIDATION_ERROR"


class ActionExecutionError(RiftboundError):
    """Raised when execute_action hits an unexpected exception."""

    error_code = "EXECUTION_ERROR"


class RoomFullError(RiftboundError):
    error_code = "ROOM_FULL"


class RoomNotFoundError(RiftboundError):
    error_code = "ROOM_NOT_FOUND"


class GameCorruptedError(RiftboundError):
    """Raised when the game state is irrecoverably broken.

    The WebSocket handler should log a full game-state snapshot when it
    catches this exception.
    """

    error_code = "GAME_CORRUPTED"
