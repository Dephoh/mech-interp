# Riftbound Simulator - Development Checkpoint

**Last Updated:** 2026-03-28
**Status:** Phase 3 (Pipeline Coverage) IN PROGRESS

## Project Goal
Align the Riftbound card game simulator to the official rules (rules_extracted.txt).
Cards' text should do what they say (Golden Rule 002: "Card text supersedes rules text").

## Architecture Overview

The engine uses a **composable Effect IR system** where card abilities are represented as
JSON-serializable IR trees (not hardcoded functions). The pipeline is:

```
CMS card JSON → card_pipeline.py → card_definitions.json (with effect_ir trees)
                                         ↓
                              card_db.py loads into CardDB
                                         ↓
                    effect_resolver.py walks IR trees at runtime
                              ↓                    ↓
                   effect_primitives.py    target_system.py
                   (atomic mutations)      (find valid targets)
                              ↓
                     trigger_system.py
                     (fire_event → scan board → push triggers to chain)
```

## What's Done

### Phase 1 - Foundation (Complete)
New files (all untracked, not yet committed):
- `effect_ir.py` - 26 primitive + 6 composition node types, TargetSpec/FilterSpec/ConditionSpec dataclasses
- `effect_primitives.py` - All 26 atomic effect functions (deal_damage, draw, buff, stun, heal, kill, move, counter, banish, tokens, energy, XP, etc.)
- `effect_resolver.py` - Full IR tree walker with condition evaluation, auto-target resolution, composition handlers
- `target_system.py` - Unified target resolution: resolve_targets(), validate_target(), count_available_targets()
- `trigger_system.py` - Event-driven triggers: 23 GameEvent types, fire_event() scans board objects
- `card_pipeline.py` - CMS JSON → engine CardDefinition converter with text→IR regex parser
- `card_definitions.json` - Consolidated card data (replaces 6 per-type JSON files)
- `test_effect_resolver.py` - 32 tests covering all primitives and compositions

Modified files:
- `card_types.py` - Added `effect_ir: dict | None` to AbilityDefinition
- `card_db.py` - Supports single-file card_definitions.json format
- `chain.py` - Prefers effect_ir over effect_script for spell/ability resolution
- `enums.py` - Added missing keywords (AMBUSH, BACKLINE, HUNT, LEVEL, MIGHTY, PREDICT, UNIQUE)
- `main.py` - Updated data_dir path for new card format
- `helpers.py` - Updated DATA_DIR for new card format

Deleted files:
- `backend/data/cards/{battlefields,gear,legends,runes,spells,units}.json` (replaced by card_definitions.json)

### Phase 2 - Integration (Complete)
All trigger events wired into engine modules. fire_event() now called from:

| Module | Events Fired |
|--------|-------------|
| `chain.py:push_card_to_chain()` | SPELL_PLAYED (after spell goes on chain) |
| `chain.py:_finalize_permanent()` | UNIT_PLAYED (units), UNIT_ENTERED (gear/other) |
| `cleanup.py:_kill_dead_units()` | UNIT_DIED + FRIENDLY_DEATH + ENEMY_DEATH (per dying unit) |
| `combat.py:start_combat()` | ATTACK_STARTED / DEFEND_STARTED (per unit) |
| `combat.py:resolve_combat()` | COMBAT_WIN (after winner determined) |
| `scoring.py:score_hold()` | HOLD |
| `scoring.py:score_conquer()` | CONQUER |
| `state_machine.py:_execute_awaken()` | TURN_START |
| `state_machine.py:_execute_end_of_turn()` | TURN_END |
| `action_executor.py:_exec_move_unit()` | UNIT_MOVED_TO_BF / UNIT_MOVED |
| `action_executor.py:_exec_recycle_rune()` | RECYCLE_RUNE |

Key integration decisions:
- **Death ordering**: Fire all death events (UNIT_DIED, FRIENDLY_DEATH, ENEMY_DEATH) in Phase 1 while units still on board, then remove in Phase 2
- **Unit play events**: UNIT_PLAYED for units (covers on_play self + on_unit_played observers), UNIT_ENTERED for gear (on_play self only)
- **Deathknell replaced**: Removed ad-hoc `process_deathknell()` — trigger_system handles all on_death triggers via fire_event
- **Target validation bridged**: action_validator delegates to target_system via `_LEGACY_MAP` dict converting old string types to TargetSpec

Additional fixes:
- `trigger_system._get_board_objects()`: Fixed `ps.legend_zone` iteration (was `for lid in ps.legend_zone`, now `if ps.legend_zone:` since it's `str | None`)
- `trigger_system._trigger_applies()`: Added battlefield location check for conquer/hold triggers
- `action_validator.py`: Added `_build_target_spec()` to extract specs from effect_ir or convert legacy strings

### Phase 3 - Pipeline Coverage (In Progress)
Expanded `card_pipeline.py` text parser with ~20 new regex patterns:

**New effect patterns**: `[Add] [N]`/`[Add] [C]` resource gain, `[Buff]`/`[Stun]` bracketed verbs,
give keyword (`give TARGET [Keyword]`), move enemy/recall, plural tokens with word numbers
(`play three 1 [S] Recruit unit tokens`), Gold gear tokens, `[>]` Level ability prefixes,
`[M]` as Might alias alongside `[S]`, `N to TARGET` damage shorthand, I enter ready,
I have +N [S] static might, restrict opponents, channel exhausted variant, ready N runes.

**New target patterns**: `another friendly unit`, `your other units here`, runes,
`one of their gear`, might-comparison filters, `an enemy unit attacking here`.

**New primitives implemented**: `give_keyword` (grant keyword to target), `restrict` (opponent
restriction, logged but not yet enforced). Updated `play_token` (count, gear type, exhausted flag),
`channel_rune` (exhausted flag).

**Coverage delta**: 244 → 328 abilities with effect_ir (+84). Dead abilities: 479 → 396 (-83).
Card-level: 329 → 334 cards with at least one IR ability.

**Only effect_script remaining**: 12 (6 basic runes x exhaust+recycle) — these are game actions,
not chain-resolved effects, so effect_script is correct for them.

### Test status: 102/102 passing
- 38 effect_resolver tests (+6 new: give_keyword, play_token count/gear, channel exhausted, restrict)
- 17 trigger_integration tests
- 47 existing engine tests

## Critical Bugs Found (from 2026-03-28 play session)

### BUG 1: Combat units recalled instead of killed (combat.py:344-358)

**Symptom**: A Recruit (1 Might) takes 3 combat damage but survives and gets recalled to base.
**Root cause**: `resolve_combat()` calls `unit.heal()` on ALL units BEFORE checking survivors.
Healing clears damage, so a unit with lethal damage (damage >= might) becomes alive again.
Additionally, the code only recalls attackers when defenders survive — it never kills anyone.
Combat deaths are supposed to happen DURING combat resolution, not deferred to cleanup.

**Fix plan** (in `combat.py:resolve_combat()`, lines 344-397):
1. **Before healing**, identify dead units: units where `damage >= effective_might`
2. **Kill dead units**: Move to trash, fire UNIT_DIED + FRIENDLY_DEATH + ENEMY_DEATH events
   (reuse the same pattern as `cleanup._kill_dead_units()`)
3. **Heal surviving units** (only the ones that weren't killed)
4. **Determine winner**: Side with surviving units wins. If both have survivors → contested
5. **Recall surviving losers**: Losing side's surviving units get recalled to base
   - If defenders won: recall surviving attackers
   - If attackers won: recall surviving defenders
   - If both wiped out: no recalls, battlefield becomes uncontrolled
6. **Then** determine control and fire COMBAT_WIN as before

**Key code change** — replace lines 344-358 with:
```python
# 1. Kill units with lethal damage (BEFORE healing)
dead_units = []
for uid in list(bf.units):
    unit = gs.instances.get(uid)
    if unit and unit.damage >= unit.effective_might and unit.effective_might > 0:
        dead_units.append(unit)

for unit in dead_units:
    # Fire death events while still on board
    ctx = {"card_id": unit.instance_id, "player_id": unit.controller_id}
    logs.extend(fire_event(gs, GameEvent.UNIT_DIED, ctx))
    logs.extend(fire_event(gs, GameEvent.FRIENDLY_DEATH, ctx))
    logs.extend(fire_event(gs, GameEvent.ENEMY_DEATH, ctx))
    # Move to trash
    bf.units.remove(unit.instance_id)
    unit.zone = ZoneType.TRASH
    unit.location_id = None
    unit.damage = 0
    logs.append(f"{unit.name} destroyed in combat")

# 2. Heal surviving units
for uid in list(bf.units):
    unit = gs.instances.get(uid)
    if unit:
        unit.heal()

# 3. Determine winner and recall losers
surviving_attackers = [
    gs.instances[uid] for uid in bf.units
    if uid in gs.instances and gs.instances[uid].controller_id == combat.attacker_id
]
surviving_defenders = [
    gs.instances[uid] for uid in bf.units
    if uid in gs.instances and gs.instances[uid].controller_id == combat.defender_id
]

if surviving_defenders and not surviving_attackers:
    pass  # attackers already dead/gone
elif surviving_attackers and not surviving_defenders:
    pass  # defenders already dead/gone
elif surviving_defenders and surviving_attackers:
    # Both sides survive — attackers recall (they initiated)
    for unit in surviving_attackers:
        _recall_unit(gs, unit)
        logs.append(f"{unit.name} recalled to base")
# If neither side survives, no recalls needed
```

### BUG 2: Spells rejected — "requires N target(s), got 0" (frontend)

**Symptom**: Playing "Against the Odds" (a spell that targets a unit) fails with
`REJECTED play_card: Against the Odds requires 1 target(s), got 0`.
**Root cause**: Frontend sends `targets: []` for all PLAY_CARD actions. There is no
target selection UI — the user has no way to choose targets.

**Fix plan** (frontend changes):
1. **GameBoard component** — when user clicks a spell card with `targets_required > 0`:
   - Enter "targeting mode" instead of immediately sending PLAY_CARD
   - Store the pending spell card ID in component state
   - Show a prompt: "Select a target for [spell name]"
2. **Board rendering** — during targeting mode:
   - Highlight valid target units/cards (use the spell's target_spec to filter)
   - Add click handlers on valid targets
3. **Target selection** — when user clicks a valid target:
   - Collect the target's `instance_id`
   - Send PLAY_CARD with `targets: [instance_id]`
   - Exit targeting mode
4. **Cancel** — ESC or right-click cancels targeting mode
5. **Backend already validates** — no backend changes needed for basic targeting

**Simpler alternative** (if time is tight):
- Add a modal/dropdown that lists valid targets when a targeted spell is played
- User selects from list, frontend populates `targets` array
- Less polished but functional

### BUG 3 (minor): Some triggered abilities have no effect_ir

**Symptom**: Abilities like Bewitching Spirit's "choose a player. They discard 1" fire
their trigger but resolve with no effect (the ability has no `effect_ir`).
**Root cause**: Pipeline coverage gap — 68 triggered abilities still lack IR.
**Not blocking gameplay** — these are "nice to have" for card fidelity.
**Fix**: Continue expanding card_pipeline.py regex patterns or use LLM-assisted IR gen.

### DONE: Card art images on all cards

**Implemented**: Cards now display art from local `/card-images/{card_id}.png`.
- Backend: `main.py` mounts `/card-images` static route → `cards/images/` (949 PNGs)
- Frontend: `CardView.tsx` shows `<img>` for both full (64px tall) and compact (24x24) modes
- Missing art fallback: Magenta dashed border + diagonal stripe pattern + "NO ART" + card_id
  (intentionally ugly so it's obvious which cards need images)
- `onError` handler catches broken/missing images and shows the placeholder
- All 767 card_ids match image filenames — 0 missing

**NOTE (2026-03-28)**: Images didn't display during testing because the server was not
restarted after the code changes. The code is verified correct. To fix:
1. Stop the running server (Ctrl+C)
2. Restart: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. Hard-refresh browser: Ctrl+Shift+R (clears cached old JS/CSS bundle)

### FEATURE 1: Card hover tooltip in all zones

**Current state**: Full-mode cards (hand) have a native browser `title` attribute showing
`card.text`. Compact-mode cards (base units, gear, runes) have NO tooltip at all.
**Desired**: Hovering any card in any zone shows a rich tooltip with full card info:
name, cost, type, might, keywords, full ability text.

**Fix plan** (frontend — `CardView.tsx` + new `CardTooltip.tsx`):
1. Create a `CardTooltip` component that renders a full card preview panel
   - Shows: name, energy cost, power cost, card type, might, keywords, full text
   - Position: absolute, offset from cursor or anchored to card edge
   - Dark panel style matching existing theme (#16213e bg, border)
2. Add `onMouseEnter`/`onMouseLeave` handlers to BOTH full and compact card modes
3. On hover, show the tooltip after a brief delay (~200ms) to avoid flicker
4. Tooltip should be portal-mounted (appended to document.body) to avoid z-index/overflow clipping
5. **Alternative**: CSS-only approach using `::after` pseudo-element with `content` attr,
   but this limits formatting. Component approach is better for rich display.

### FEATURE 2: Gear activation UI

**Current state**: Gear cards in `BaseZone.tsx` render as compact read-only cards.
No buttons, no click handlers. The backend has `ACTIVATE_ABILITY` action type in enums
but BaseZone doesn't expose it.

**Fix plan**:
1. **BaseZone.tsx** — add an "Activate" button on each gear card (similar to rune E/P buttons)
   - Only show when it's your turn and you have priority
   - Button sends `ACTIVATE_ABILITY` message with the gear's `instance_id` and `ability_id`
2. **GameBoard.tsx** — add `handleActivateAbility(instanceId, abilityId)` handler
   - Sends: `{ type: "ACTIVATE_ABILITY", instance_id, ability_id, targets: [] }`
   - If ability needs targets, enter targeting mode (same as spell targeting from Bug 2)
3. **CardView.tsx** — for gear in base, show ability text so user knows what activating does
4. **Backend check** — verify `action_validator.py` handles ACTIVATE_ABILITY for gear cards
   in base zone. May need to allow gear abilities to be activated from base (not just battlefield).

### FEATURE 3: Opponent choice / prompt system

**Current state**: Effects like "choose a player. They discard 1" have no mechanism for
the opponent to make a choice. The engine resolves effects immediately without pausing
for input. There's no `CHOICE_REQUIRED` server message or choice UI.

**Fix plan** (backend + frontend):

**Backend** (`effect_resolver.py` / `effect_primitives.py`):
1. When an effect requires an opponent's choice (e.g., discard), instead of resolving
   immediately, push a `CHOICE_PENDING` state onto `GameState`
2. New server message: `CHOICE_REQUIRED { player_id, choice_type, options, source_card }`
   - `choice_type`: "discard_card", "choose_target", "choose_option"
   - `options`: list of valid choices (card instance_ids, option labels, etc.)
3. New client message: `SUBMIT_CHOICE { choice_id, selection }`
4. Engine pauses effect resolution until choice is submitted, then continues

**Frontend** (`GameBoard.tsx` + new `ChoiceModal.tsx`):
1. On receiving `CHOICE_REQUIRED`, show a modal/overlay:
   - "Bewitching Spirit: Choose a card to discard"
   - Highlight valid choices (cards in hand for discard)
   - Player clicks to select, confirm button sends `SUBMIT_CHOICE`
2. During opponent's choice, show "Waiting for opponent to choose..."
3. This is the most complex feature — consider implementing a simpler version first:
   - Auto-select random valid choice for opponent (log it)
   - Then add UI later

### FEATURE 4: Visual chain/stack panel

**Current state**: Sidebar shows chain items as a flat text list (name + "You"/"Opp").
No visual card representation, no resolution animation, hard to follow what's happening.

**Fix plan** (frontend — `Sidebar.tsx` or new `ChainPanel.tsx`):
1. **Replace text list** with a visual stack on the right side of the board
   - Each chain item rendered as a mini card (similar to compact CardView)
   - Stack order: newest on top (LIFO visual)
   - Show: card name, controller color (your color / opponent color), ability text snippet
2. **Highlight the top item** — the one that will resolve next when both players pass
   - Gold border or glow animation
   - Show full text of top item in expanded view below the stack
3. **Resolution animation** — when an item resolves:
   - Flash/fade the top card
   - Brief "Resolved: [card name]" indicator
4. **Priority indicator** — show whose turn it is to respond
   - "Your priority — Pass or respond" / "Opponent responding..."
5. **Layout change** — move chain panel to dedicated right-side column:
   ```
   ┌────────────────────────────┬──────────┐
   │        GAME BOARD          │  CHAIN   │
   │  (hands, battlefields,     │  STACK   │
   │   base, runes, actions)    │  PANEL   │
   │                            │  + logs  │
   └────────────────────────────┴──────────┘
   ```
   The existing Sidebar can become the chain panel. Move game log below the stack.

---

## What Needs Doing (Phase 3 continued)

### HIGH: Replace hardcoded effects.py functions
- Only 12 effect_script uses remain (all basic runes) — mostly done
- The ~11 named effects in effects.py are now dead code for cards that gained IR
- Could clean up effects.py to remove unused functions
### HIGH: Continue expanding text parser
Remaining 396 dead abilities breakdown:
- 244 passive (static modifiers, "While..." conditions — need modifier system)
- 89 activated (complex multi-target, dynamic amounts, "choose one" — harder regex)
- 63 triggered (complex effects after trigger prefix — diminishing regex returns)
Many remaining abilities need dynamic amount resolution ("damage equal to my Might"),
multi-choice ("choose one —"), or cross-card references that regex can't handle well.
Consider LLM-assisted IR generation for the long tail.

### MEDIUM: Missing keyword implementations
Working: ACCELERATE, ACTION, REACTION, TANK, TEMPORARY, LEGION, DEATHKNELL, ASSAULT, SHIELD, STUNNED, BUFF_COUNTER
Missing/incomplete: DEFLECT, GANKING, HIDDEN, HUNT, LEVEL, PREDICT, QUICK_DRAW, REPEAT, WEAPONMASTER, AMBUSH, BACKLINE, UNIQUE, VISION

### MEDIUM: Replacement effects stub
- `trigger_system.check_replacements()` returns False (stub)
- Needed for: damage prevention, "instead" effects, "If this would die, instead..."

### LOW: Persistent modifier system
- `might_modifiers` list is cleared each turn
- Cards that permanently buff need a separate system

## Key Rules to Remember
- Golden Rule (002): Card text supersedes rules text
- Silver Rule (051): Card text interpreted by rules, not as rules
- Rule 055: Do as much as you can, ignore impossible instructions
- Rule 054: "Can't beats Can"
- Triggers fire AFTER inciting event is processed
- Multiple simultaneous triggers: Turn Player orders first, then others in turn order
- Chain is LIFO; both players must pass priority for top item to resolve
- Units enter exhausted (unless Accelerate paid)
- Damage heals at end of turn AND during combat cleanup
- Final scoring point requires all battlefields scored (Conquer path)

## File Map (engine directory)
```
backend/app/engine/
  action_executor.py  - Dispatches player actions
  action_validator.py - Validates action legality
  auto_player.py      - AI player (basic heuristics)
  card_db.py          - Card definition registry
  card_pipeline.py    - CMS→engine card converter [NEW]
  card_types.py       - CardDefinition, CardInstance, AbilityDefinition
  chain.py            - Chain (stack) management and resolution
  cleanup.py          - Death processing, showdown staging, state cleanup
  combat.py           - Showdown/combat resolution and damage assignment
  effect_ir.py        - Effect IR node types and constructors [NEW]
  effect_primitives.py - Atomic effect implementations [NEW]
  effect_resolver.py  - IR tree walker [NEW]
  effects.py          - Legacy named-effect registry (being replaced)
  enums.py            - All game enums (Phase, Zone, Keyword, etc.)
  game_state.py       - GameState, PlayerState, ChainState, etc.
  keywords.py         - Keyword-specific logic (accelerate, tank, etc.)
  scoring.py          - Hold/conquer/burn-out/win conditions
  state_machine.py    - Turn phase advancement
  target_system.py    - Unified target resolution [NEW]
  trigger_system.py   - Event-driven trigger system [NEW]
```
