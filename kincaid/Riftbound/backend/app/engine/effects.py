"""Legacy effect function registry — DEPRECATED.

This module previously housed named effect functions for card abilities
(the ``effect_script`` system).  All card abilities have been migrated to
the composable Effect IR system (``effect_ir.py`` / ``effect_resolver.py``).

The module is retained only as a no-op safety net: if any code path still
calls ``resolve_effect``, it returns a clear "unknown effect" message rather
than crashing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .card_types import CardInstance
    from .game_state import GameState


def resolve_effect(
    name: str,
    source: CardInstance,
    gs: GameState,
    targets: list[str],
) -> list[str]:
    """No-op stub — all effects should now use effect_ir.

    Returns a warning message so callers can detect stale code paths.
    """
    return [f"[DEPRECATED] Legacy effect_script invoked: {name}"]
