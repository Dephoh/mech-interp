# Riftbound

Card game simulator: Python/FastAPI backend (WebSocket real-time), React/TypeScript/Zustand frontend.

## Running

```bash
# All tests (from backend/)
cd backend && python -m pytest -q

# Full eval with composite score (from project root)
python eval/run_eval.py

# Game server
./play.sh   # or play.bat on Windows
```

## Evaluation System

`eval/run_eval.py` produces a composite score (0-100) used as a verifiable reward signal:

| Check | Weight | What it measures |
|-------|--------|-----------------|
| Tests | 50% | pytest pass rate across all backend tests |
| Contracts | 30% | Serialized state validates against `contracts/state_update.schema.json` |
| IR Coverage | 20% | % of card abilities with `effect_ir` trees (vs text-only) |

**Agent work MUST NOT decrease the composite score.** Run eval before and after changes.

## Agent Workflow

1. Read the task from `DASHBOARD.md` (includes agent prompts with exact file paths and line numbers)
2. Read all referenced files before making changes
3. Implement the change
4. Run `cd backend && python -m pytest -q` — all tests must pass
5. Run `python eval/run_eval.py` — composite score must not decrease
6. New features require new tests in `backend/tests/`

## Architecture Constraints

- **Immutable definitions**: `CardDefinition` is `frozen=True` — never mutate at runtime
- **State mutations**: All game state changes go through `GameState` methods or `effect_primitives.py`
- **Effect IR**: Nodes are plain dicts with a `"type"` key — must stay JSON-serializable
- **Contract boundary**: `protocol/serializers.py` is the single bridge between backend state and frontend display. Its output MUST validate against `contracts/state_update.schema.json`
- **Frontend types**: `frontend/src/ws/messageTypes.ts` must match the contract schema
- **No direct card data mutation**: Card abilities are expressed as Effect IR trees, resolved by `effect_resolver.py`, executed by `effect_primitives.py`

## Key Files

| File | Purpose |
|------|---------|
| `backend/app/engine/game_state.py` | Authoritative game state (all mutable data) |
| `backend/app/engine/effect_ir.py` | Effect IR node types and constructors |
| `backend/app/engine/effect_resolver.py` | IR tree walker — resolves effects into state mutations |
| `backend/app/engine/effect_primitives.py` | Atomic game mutations (deal damage, draw, etc.) |
| `backend/app/engine/card_pipeline.py` | CMS card JSON -> engine CardDefinitions with IR |
| `backend/app/protocol/serializers.py` | GameState -> client-safe JSON (contract boundary) |
| `frontend/src/ws/messageTypes.ts` | TypeScript types for all WebSocket messages |
| `contracts/state_update.schema.json` | JSON Schema contract between backend and frontend |
| `DASHBOARD.md` | Task backlog with agent prompts |
| `eval/run_eval.py` | Composite evaluation runner |

## Test Helpers

Tests use shared helpers from `backend/tests/helpers.py`:
- `make_game(phase=Phase.ACTION)` — minimal 2-player game state
- `add_unit(gs, player_id, zone, might=3, bf_id=None)` — add a unit to the game
- `make_unit_def(...)`, `make_spell_def(...)` — create card definitions
- `load_card_db()` — load real card definitions from `data/card_definitions.json`
