# Riftbound Turn Structure - Rules Reference
> Source of truth: backend/data/rules/09_turn_structure.txt, backend/data/rules/19_scoring_and_modes.txt

## Turn Phases (in strict order)

### 1. AWAKEN PHASE (315.1)
- Turn Player readies ALL game objects they control that can be readied
- Includes: units, gear, runes on the board

### 2. BEGINNING PHASE (315.2)
- **Beginning Step (315.2.a)**: "start of beginning phase" triggers fire
  - Temporary units are killed here (742.1.b): "At the start of this permanent's controller's Beginning Phase, before scoring, kill this."
- **Scoring Step (315.2.b)**: Holding occurs
  - If Turn Player controls a battlefield, they score 1 point (Hold)
  - Hold triggers fire here
  - In 2v2: teammate-controlled battlefields are excluded from scoring

### 3. CHANNEL PHASE (315.3)
- Turn Player channels 2 runes from Rune Deck (315.3.b)
- If fewer than 2 runes remain, channel as many as possible
- **1v1 Duel (462.7)**: Player going SECOND channels 3 runes (extra 1) on their first turn

### 4. DRAW PHASE (315.4)
- Turn Player draws 1 card (315.4.b)
- If deck empty: Burn Out first, then draw 1 (315.4.b.1-2)
- **CRITICAL (315.4.d)**: "As the Draw Phase ends, each player's Rune Pool empties."
  - Note: this empties ALL players' pools, not just the turn player
- **FFA3/FFA4/2v2 (464.7, 465.7, 466.7)**: Player going FIRST does not draw on their first turn

### 5. ACTION PHASE (316)
- No defined structure — player takes discretionary actions freely (316.2)
- Only Turn Player can play cards/activate abilities in Neutral Open (316.2.b)
- This is where combat, showdowns, card plays happen
- Ends when Turn Player indicates they are done (316.6)

### 6. END OF TURN PHASE (317)
- **Ending Step (317.1)**: "end of turn" triggers fire
- **End of Turn Cleanup (317.2)**:
  - Standard cleanup steps (rule 318-322)
  - PLUS: "2c. Heal all Units" — ALL unit damage is cleared
- **Expiration Step (317.3)**: 
  - All "this turn" effects expire simultaneously (317.3.a)
  - **CRITICAL (317.3.b)**: "As the Expiration Step ends, all players' Rune Pools empty."

## Rune Pool Emptying (rule 163)
**Empties at TWO specific times per turn:**
1. End of Draw Phase (315.4.d) — each player's pool empties
2. End of Expiration Step (317.3.b) — each player's pool empties

This means:
- Resources added during Channel Phase (by auto-tapping runes) are LOST before Action Phase
- Players must tap/recycle runes DURING the Action Phase to pay for cards
- Resources persist through the entire Action Phase until end of turn

## Turn States (rules 307-310)
Four possible states:
1. **Neutral Open**: No showdown, no chain — default during Action Phase
2. **Neutral Closed**: No showdown, chain exists — after playing a card/ability
3. **Showdown Open**: Showdown in progress, no chain
4. **Showdown Closed**: Showdown in progress, chain exists

## Priority (rule 312)
- Only ONE player has priority at a time
- Priority holder is the one who can take discretionary actions
- Granted when:
  - Neutral Open during Action Phase (turn player)
  - Showdown State and gaining Focus
  - Closed State and controlling the newest chain item
  - Closed State, next in turn order after current priority holder passes

## 1v1 Duel Mode Specifics (462)
- 2 players, victory score 8
- 2 battlefields (one per player)
- Player going second channels 3 runes (not 2) on first turn
