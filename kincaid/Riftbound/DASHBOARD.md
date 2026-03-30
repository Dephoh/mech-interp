# Riftbound Work Dashboard

**Last Updated:** 2026-03-29
**Tests:** 814+ collected (25+ files) | **Backend:** ~98% | **Frontend:** ~98%

> **How to use**: Copy an item's **Agent Prompt** into a Claude Code session along with
> the listed files. Update **Status** as work completes: `TODO` -> `IN_PROGRESS` -> `DONE`.

---

## Backend Engine

| ID | Title | Pri | Status | Notes |
|----|-------|-----|--------|-------|
| BE-01 | Hidden card reveal in combat | P1 | DONE | Already implemented; 8 new tests added |
| BE-02 | Serializer completeness audit | P2 | DONE | Added `supertypes`, `effect_text`, `scored_this_turn_by`; updated TS types + JSON schema |
| BE-03 | Missing keyword implementations | P2 | DONE | Weaponmaster, Quick-Draw, Predict, Hunt, Level, Hidden ready, Vision wired; 51 new tests |
| BE-04 | Replacement effects wiring verification | P2 | DONE | Fixed passive ability registration on board entry; 16 new tests |
| BE-05 | Continuous modifier / layer system | P2 | DONE | `layers.py` with 3-layer eval (rules 450-457), wired into cleanup; 58 new tests |
| BE-06 | Card pipeline IR coverage expansion | P2 | DONE | +819 lines, 80+ new patterns; IR coverage 50.8% -> 91.8% at ability level |
| BE-07 | Multi-player game modes (FFA3/FFA4/2v2) | P3 | DONE | GameMode enum, opponent_ids/teammate_id/next_player, team scoring; 27 new tests |
| BE-08 | Audit legacy effects.py usage | P3 | DONE | Gutted 11 dead effect functions, removed effect_script fallback, cleaned card data |

---

## Frontend UI

| ID | Title | Pri | Status | Notes |
|----|-------|-----|--------|-------|
| FE-01 | Wire up DamageAssignmentModal | P0 | DONE | Already wired (import, store hooks, auto-trigger, rendering, WS message) |
| FE-02 | Wire up ChoiceModal + multi-select | P0 | DONE | Fixed multi-select submit; added `chosen_option_indices` to protocol + backend |
| FE-03 | Extract shared isValidTarget() | P1 | DONE | Already extracted to `utils/targeting.ts` |
| FE-04 | Remove or implement context menu | P2 | DONE | Removed dead `contextMenu` state from uiStore.ts |
| FE-05 | On-board damage display | P2 | DONE | Improved damage/might badge positioning; no badge overlap |
| FE-06 | Disconnect / reconnect UI | P2 | DONE | Exponential backoff, auto-rejoin, connection status banners with animations |
| FE-07 | Card image path configuration | P3 | DONE | `getCardImageUrl()` utility reading from `VITE_CARD_IMAGE_BASE` env var |
| FE-08 | Mobile responsiveness | P3 | DONE | 3 breakpoints (1024/768/480px), stacked layout, horizontal scroll zones |

---

## Testing

| ID | Title | Pri | Status | Notes |
|----|-------|-----|--------|-------|
| TS-01 | API / WebSocket integration tests | P1 | DONE | 9 tests (5 REST + 4 WebSocket); fixed keyword import stubs |
| TS-02 | card_pipeline.py unit tests | P1 | DONE | 177 tests (120 new); card types, 49 effect patterns, edge cases |
| TS-03 | Combat edge case tests | P1 | DONE | 11 new tests: tank ordering, lethal-before-next, simultaneous elimination, gear detach |
| TS-04 | End-to-end 2-player game test | P2 | DONE | 15 tests: full turn cycle, scoring, combat, game over, multi-turn victory |
| TS-05 | Layer system verification tests | P2 | DONE | 58 tests: all 3 layers, modifier ordering, cleanup, effective_might, Mighty threshold |
| TS-06 | Performance / load testing | P3 | DONE | 14 tests: 50 concurrent rooms, pipeline benchmark, resolver profiling; `@pytest.mark.slow` |

---

## Infrastructure

| ID | Title | Pri | Status | Notes |
|----|-------|-----|--------|-------|
| IN-01 | Game state persistence | P2 | DONE | `persistence.py` with serialize/deserialize, auto-save, `GET /rooms/{id}/state`; 31 new tests |
| IN-02 | Player reconnection handling | P1 | DONE | Backend already done; added frontend wiring (token, RECONNECT msg, notifications); 18 new tests |
| IN-03 | Rate limiting and auth | P2 | DONE | 60 req/min REST, 30 msg/sec WS, optional API key; zero new dependencies |
| IN-04 | Structured error responses | P2 | DONE | `exceptions.py` (7 types), game state snapshots on error, per-action isolation |

---

## File Quick Reference

All paths relative to `backend/app/engine/` unless noted otherwise.

| Area | Key Files | Lines | Tests |
|------|-----------|-------|-------|
| Engine core | 18 modules in `engine/` (incl. `layers.py`, `persistence.py`) | ~10,000 | 814+ (25+ files) |
| Protocol | `protocol/serializers.py`, `protocol/messages.py` | ~380 | 9 (API tests) |
| Server | `main.py`, `room_manager.py`, `exceptions.py` | ~900 | 18 (reconnect) + 31 (persistence) |
| Frontend | 10 TSX + 2 stores + 2 WS files + 2 utils | ~2,200 | 0 |
| Rules | 22 files in `data/rules/` | ~2,500 | N/A |
| Card data | `data/card_definitions.json` (91.8% IR coverage) | 24,139 | 177 (pipeline) |
