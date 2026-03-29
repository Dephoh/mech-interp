# Riftbound Work Dashboard

**Last Updated:** 2026-03-29
**Tests:** 179 collected (10 files) | **Backend:** ~95% | **Frontend:** ~92%

> **How to use**: Copy an item's **Agent Prompt** into a Claude Code session along with
> the listed files. Update **Status** as work completes: `TODO` -> `IN_PROGRESS` -> `DONE`.

---

## Backend Engine

| ID | Title | Pri | Status | Files |
|----|-------|-----|--------|-------|
| BE-01 | Hidden card reveal in combat | P1 | TODO | `combat.py`, `card_types.py`, `cleanup.py` |
| BE-02 | Serializer completeness audit | P2 | TODO | `protocol/serializers.py`, `game_state.py`, `card_types.py` |
| BE-03 | Missing keyword implementations | P2 | TODO | `keywords.py`, `enums.py`, `data/rules/21_keywords.txt` |
| BE-04 | Replacement effects wiring verification | P2 | TODO | `trigger_system.py`, `effect_resolver.py`, `game_state.py` |
| BE-05 | Continuous modifier / layer system | P2 | TODO | `effect_primitives.py`, `game_state.py`, `data/rules/18_layers.txt` |
| BE-06 | Card pipeline IR coverage expansion | P2 | TODO | `card_pipeline.py`, `data/card_definitions.json` |
| BE-07 | Multi-player game modes (FFA3/FFA4/2v2) | P3 | TODO | `game_state.py`, `scoring.py`, `data/rules/19_scoring_and_modes.txt` |
| BE-08 | Audit legacy effects.py usage | P3 | TODO | `effects.py`, `chain.py` |

### Agent Prompts

**BE-01** — `combat.py` (447 lines) has no handling for facedown/hidden cards during combat.
`CardInstance` has `hidden_at_battlefield` and `hidden_ready` fields (`card_types.py:173-174`),
and `BattlefieldState` has a `facedown_card` slot (`game_state.py:104`), but `combat.py` never
references them. `cleanup.py:215-222` handles facedown cleanup but combat doesn't integrate it.
Per the rules (`data/rules/12_combat.txt`), hidden cards should be revealed when combat begins
at their battlefield. Implement hidden card reveal in `start_combat()` (line 106) and verify
facedown cards are cleaned up in `resolve_combat()` (line 316).

**BE-02** — Audit `serializers.py` (293 lines) against `CardInstance` fields in `card_types.py`
and `GameState` in `game_state.py`. The `_serialize_card()` function (line 229) does NOT include
`location_id`, `hidden_at_battlefield`, or `hidden_ready`. Also confirm all `BattlefieldState`
fields serialize correctly (especially `facedown_card`). Cross-reference with frontend's
`CardView` type in `frontend/src/ws/messageTypes.ts` to ensure the frontend gets every field
it expects.

**BE-03** — `keywords.py` (119 lines) implements only: accelerate, tank, legion, deathknell,
temporary. Functions: `apply_accelerate` (line 9), `check_tank_ordering` (line 27),
`check_lethal_before_next` (line 50), `can_play_in_state` (line 73), `process_legion` (line 90),
`process_deathknell` (line 97), `should_die_temporary` (line 117). These keywords from
`enums.py` have NO runtime logic: DEFLECT, GANKING, HIDDEN, HUNT, LEVEL, PREDICT, QUICK_DRAW,
REPEAT, WEAPONMASTER, AMBUSH, BACKLINE, UNIQUE, VISION, ASSAULT, SHIELD, MIGHTY. Reference
`backend/data/rules/21_keywords.txt` for each keyword's rules. Some are passive (MIGHTY,
ASSAULT, SHIELD check might thresholds in cleanup) and some need trigger hooks.

**BE-04** — `check_replacements()` in `trigger_system.py` (line 328) has real code that walks
`gs.active_replacements` (line 335, 387), and `effect_resolver.py` (line 309) can register
replacements, and `game_state.py` (line 354) can add them. Verify the full pipeline works:
card with replacement ability enters play -> resolver registers `ActiveReplacement` -> when
the watched event fires, `check_replacements()` intercepts it -> replacement effect resolves
instead. Test with a concrete card that has a REPLACEMENT node in its effect_ir.

**BE-05** — `card_types.py` tracks `might_modifiers` on `CardInstance`, cleared each cleanup.
The rules (`data/rules/18_layers.txt`) define a 3-layer continuous effect system: (1) Trait-
Altering, (2) Ability-Altering, (3) Arithmetic (increases before decreases). Currently
`effective_might` is computed simply. Build a proper layer evaluation function that orders
modifiers by layer, applies them in sequence, and recalculates `effective_might` and active
keywords each cleanup pass. Also handle dependencies between effects in the same layer.

**BE-06** — `card_pipeline.py` (1197 lines) converts CMS card text to effect_ir via regex
and pattern matching. Many abilities still have text-only definitions without full IR trees.
Run the pipeline on all cards and report coverage: how many abilities have effect_ir vs how
many are text-only. Expand regex patterns for the highest-impact ability text patterns that
don't currently parse into IR. Diff `card_definitions.json` before and after to verify.

**BE-07** — `enums.py` has no GameMode enum. Game creation in `game_state.py` assumes 2 players.
The rules (`data/rules/19_scoring_and_modes.txt`) define FFA3 (3 players, 3 battlefields),
FFA4 (4 players, 3 battlefields), 2v2 (4 players, teams, victory=11). Requires: GameMode enum,
variable player counts in game creation, multi-opponent logic replacing `opponent_id()`, team
scoring in `scoring.py`, and 3-battlefield setup.

**BE-08** — `effects.py` (209 lines) has 11 named effect functions (old system). `chain.py`
still imports `resolve_effect` from `effects.py` (line 19) and calls it at lines 202 and 292
for abilities that use `effect_script` rather than `effect_ir`. Audit which cards still use
`effect_script`, whether they can be converted to `effect_ir`, and remove dead effect functions.

---

## Frontend UI

| ID | Title | Pri | Status | Files |
|----|-------|-----|--------|-------|
| FE-01 | Wire up DamageAssignmentModal | P0 | TODO | `DamageAssignmentModal.tsx`, `GameBoard.tsx`, `uiStore.ts` |
| FE-02 | Wire up ChoiceModal + multi-select | P0 | TODO | `ChoiceModal.tsx`, `GameBoard.tsx`, `uiStore.ts`, `App.tsx` |
| FE-03 | Extract shared isValidTarget() | P1 | TODO | `GameBoard.tsx`, `BaseZone.tsx`, `BattlefieldZone.tsx` |
| FE-04 | Remove or implement context menu | P2 | TODO | `uiStore.ts` |
| FE-05 | On-board damage display | P2 | TODO | `CardView.tsx`, `BattlefieldZone.tsx` |
| FE-06 | Disconnect / reconnect UI | P2 | TODO | `useWebSocket.ts`, `App.tsx`, `GameBoard.tsx` |
| FE-07 | Card image path configuration | P3 | TODO | `CardView.tsx`, `GameBoard.tsx`, `Sidebar.tsx` |
| FE-08 | Mobile responsiveness | P3 | TODO | `App.css` |

### Agent Prompts

**FE-01** — `DamageAssignmentModal.tsx` (81 lines) is fully implemented but **never imported
or rendered** anywhere. In `GameBoard.tsx`, import and render it when combat damage assignment
is needed. The `showDamageModal` flag exists in `uiStore.ts` (line 27) with `setDamageModal`
(line 61). Compute targets from the current combat state's battlefield units filtered to
opponent units, and totalDamage from your units' `combat_might`. On submit, send `ASSIGN_DAMAGE`
message via WebSocket. This is **P0** because combat damage assignment literally cannot work
without it.

**FE-02** — `ChoiceModal.tsx` (55 lines) is implemented but **never imported or rendered**.
`App.tsx` (line 91-99) already handles `CHOICE_REQUIRED` messages and sets `pendingChoice` in
`uiStore.ts` (line 30) via `setPendingChoice` (line 78). But no component reads `pendingChoice`
to render the modal. Wire it into `GameBoard.tsx`. Also: the modal only supports single-click
selection despite `PendingChoice` having `minChoices` and `maxChoices` fields (uiStore lines
18-19). Upgrade to support multi-select when `maxChoices > 1`.

**FE-03** — `isValidTarget()` is copy-pasted identically in 3 files: `GameBoard.tsx` (line 16),
`BaseZone.tsx` (line 21), `BattlefieldZone.tsx` (line 18). Extract to a shared utility (e.g.
`frontend/src/utils/targeting.ts`) and import in all three.

**FE-04** — `uiStore.ts` defines `contextMenu` state (line 28), `openContextMenu` (line 62),
`closeContextMenu` (line 63), but nothing calls them or renders a context menu. Either implement
a right-click context menu on cards (view details, activate ability, move options) or remove
the dead state to reduce confusion.

**FE-05** — Units take damage (`card.damage` field is serialized) but damage is only visible
in the DamageAssignmentModal. Add a visual damage indicator to `CardView.tsx` for units with
`damage > 0`: a red badge showing damage. Optionally show `effective_might / base_might` when
they differ (e.g. "3/5").

**FE-06** — `useWebSocket.ts` sets `status = "disconnected"` on close but has no reconnect
logic. When WebSocket closes unexpectedly: (a) show a banner in the UI, (b) implement
exponential backoff retry, (c) re-send JOIN_ROOM on reconnect. The `status` field is already
exported and can drive conditional rendering. See also IN-02 for backend reconnect support.

**FE-07** — Card image URL is hardcoded as `/card-images/${cardId}.png` in `CardView.tsx`,
`GameBoard.tsx`, and `Sidebar.tsx`. Extract a `getCardImageUrl(cardId)` utility that reads
from `VITE_CARD_IMAGE_BASE` env var, defaulting to `/card-images/`.

**FE-08** — Zero responsive breakpoints in CSS. `App.css` (817 lines) is desktop-only. Add
responsive CSS for `game-layout`, `main-board`, `player-strip`, `battlefield-area`, and
`sidebar` classes. At minimum: stack sidebar below board on narrow screens, make card zones
horizontally scrollable.

---

## Testing

| ID | Title | Pri | Status | Files |
|----|-------|-----|--------|-------|
| TS-01 | API / WebSocket integration tests | P1 | TODO | new: `tests/test_api.py` |
| TS-02 | card_pipeline.py unit tests | P1 | TODO | new: `tests/test_card_pipeline.py` |
| TS-03 | Combat edge case tests | P1 | TODO | `tests/test_combat.py` |
| TS-04 | End-to-end 2-player game test | P2 | TODO | new: `tests/test_e2e.py` |
| TS-05 | Layer system verification tests | P2 | TODO | new: `tests/test_layers.py` |
| TS-06 | Performance / load testing | P3 | TODO | new: `tests/test_performance.py` |

### Agent Prompts

**TS-01** — Zero API or WebSocket tests exist. All 179 tests are engine-internal. Create
`test_api.py` using `httpx.AsyncClient` with FastAPI's `TestClient`. Test: `POST /rooms`
creates a room, `GET /rooms/{id}` returns info, `GET /health` returns ok. Then test WebSocket:
connect to `/ws/{room_id}`, send `JOIN_ROOM`, receive `ROOM_JOINED`. Entry point is
`backend/app/main.py` (`app = FastAPI()`).

**TS-02** — `card_pipeline.py` is 1197 lines with zero tests. Create `test_card_pipeline.py`.
Test: (a) the main conversion function for each card type (unit, spell, gear, rune), (b) regex
pattern matching for 10+ common ability text patterns, (c) edge cases: missing fields, unknown
keywords, malformed text. Import from `backend/app/engine/card_pipeline`.

**TS-03** — `test_combat.py` has ~9 tests. Missing coverage: (a) Tank keyword ordering in
damage assignment (uses `check_tank_ordering` from `keywords.py:27`), (b) lethal-before-next
rule (uses `check_lethal_before_next` from `keywords.py:50`), (c) facedown card reveal during
combat (once BE-01 is done), (d) simultaneous elimination of both sides, (e) combat with
gear-attached units (gear detaches to base on death).

**TS-04** — Create an end-to-end test simulating a full 2-player game via WebSocket: two
clients connect, mulligan, play cards, move units, trigger combat, assign damage, verify scores.
Use `pytest-asyncio` and `httpx` WebSocket support, or the `/sandbox` endpoint for a
pre-configured game state.

**TS-05** — Once BE-05 (layer system) is implemented, test: (a) Trait-Altering layer before
Ability-Altering, (b) Arithmetic layer increases before decreases, (c) dependency resolution
between effects in same layer, (d) timestamp ordering for independent effects. Reference
`backend/data/rules/18_layers.txt`.

**TS-06** — No load tests exist. Create a test that: (a) creates 50 concurrent rooms, (b) runs
100 actions per room, (c) measures response times. Also profile `card_pipeline.py` conversion
of all cards and `effect_resolver.py` tree walking for complex IR trees.

---

## Infrastructure

| ID | Title | Pri | Status | Files |
|----|-------|-----|--------|-------|
| IN-01 | Game state persistence | P2 | TODO | `room_manager.py`, `game_state.py` |
| IN-02 | Player reconnection handling | P1 | TODO | `main.py`, `room_manager.py`, `useWebSocket.ts` |
| IN-03 | Rate limiting and auth | P2 | TODO | `main.py` |
| IN-04 | Structured error responses | P2 | TODO | `main.py`, `room_manager.py` |

### Agent Prompts

**IN-01** — All game state lives in-memory in `RoomManager` (`room_manager.py`, 269 lines).
Server restart = all games lost. Implement: (a) serialize `GameState` to JSON (it's dataclasses
with simple types), (b) save to SQLite or file after each action, (c) add `GET /rooms/{id}/state`
endpoint, (d) reload active games on startup.

**IN-02** — When WebSocket disconnects (`main.py:355`), the player is removed (`main.py:368`)
with no way back. Implement: (a) keep slot reserved for 60s after disconnect, (b) allow
reconnection with same player_id via a reconnect token, (c) re-send current game state on
reconnect. Frontend: `useWebSocket.ts` needs auto-reconnect with exponential backoff (see FE-06).

**IN-03** — `main.py` sets `allow_origins=["*"]`. No rate limiting. Add: (a) `slowapi` or
similar rate limiting for REST endpoints, (b) WebSocket message rate limiting in the message
loop (`main.py:342`), (c) optional API key for room creation.

**IN-04** — The WebSocket handler (`main.py:348-353`) catches exceptions but sends a generic
"Internal server error". Improve: (a) structured error JSON with action type and details,
(b) log full game state snapshot on error for debugging, (c) prevent one corrupted game from
affecting other rooms.

---

## File Quick Reference

All paths relative to `backend/app/engine/` unless noted otherwise.

| Area | Key Files | Lines | Tests |
|------|-----------|-------|-------|
| Engine core | 17 modules in `engine/` | ~7,500 | 179 (10 files) |
| Protocol | `protocol/serializers.py`, `protocol/messages.py` | ~350 | 0 |
| Server | `main.py`, `room_manager.py` | ~665 | 0 |
| Frontend | 9 TSX + 2 stores + 2 WS files | ~1,835 | 0 |
| Rules | 22 files in `data/rules/` | ~2,500 | N/A |
| Card data | `data/card_definitions.json` | 24,139 | N/A |
