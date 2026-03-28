# Riftbound Combat & Scoring - Rules Reference
> Source of truth: rules_extracted.txt (rules 437-449)

## Combat Trigger (437-438)
- Combat occurs when: Cleanup runs, chain is empty, and a Battlefield has units from 2 opposing players
- Combat is "Staged" before steps begin (439)
- Turn Player picks which combat resolves first if multiple (439.1)
- Only between exactly 2 players (440)

## Steps of Combat (441-444)

### Step 1: Showdown Step (442)
1. Establish Attacker/Defender:
   - Attacker = player whose units applied Contested status (442.1.a.1)
   - Defender = other player (442.1.a.2)
   - Units at battlefield get matching designations (442.1.a.3-4)
2. Attacker gains Focus (442.1.a.1.a)
3. Initial Chain: triggered abilities from Attack/Defend triggers placed on chain
   - Focus holder (Attacker) places first, then others in turn order, Defender last (442.1.b.1)
4. Showdown proceeds: Focus holder can play Action/Reaction spells or pass
5. When all pass → Showdown closes

### Step 2: Combat Damage Step (443)
- Only if BOTH attacking and defending units remain (443.1.a)
- Sum Might of all Attacking units (443.1.b)
- Sum Might of all Defending units (443.1.c)
- Starting with Attacker, each player ASSIGNS damage equal to their summed Might (443.1.d)
- **Assignment rules**:
  - Assigning ≠ Dealing damage (443.1.d.1)
  - Must assign LETHAL damage to a unit before moving to next (443.1.d.3)
  - Cannot assign MORE than lethal unless no other units remain (443.1.d.4)
  - Tank units must receive damage FIRST (741.1.b)
  - Stunned units don't contribute Might but still need lethal to kill (410.1.b-c)
- After ALL assignment: damage is DEALT simultaneously (443.1.d.1.a)

### Step 3: Resolution Step (444)
1. **Combat Cleanup** (444.1):
   - Normal cleanup steps (322.1-322.12) PLUS:
   - 2c. Heal ALL units (444.1.a.1)
   - 2d. Recall Attackers if Defenders still present (444.1.a.2)
   - 2e. Remove Attacker/Defender designations (444.1.a.3)
2. **Control Resolution** (444.2):
   - If one player's units remain → establish Control + clear Contested
   - If that player hasn't scored this battlefield this turn → CONQUER (score 1 point)
   - If no units remain → Battlefield becomes Uncontrolled

## Scoring (445-449)
Two methods:
1. **Conquer** (446.1): Gain control of a battlefield not yet scored this turn
2. **Hold** (446.2): Maintain control during Beginning Phase

### Scoring Rules:
- Can only score once per battlefield per turn (447)
- Each score earns 1 point (448.1)
- **Final Point** special rules (448.1.b):
  - Via Hold: always scores (448.1.b.1)
  - Via Conquer: must have scored EVERY battlefield this turn (448.1.b.2)
  - If Conquer but didn't score all battlefields: draw a card instead

### Victory:
- 1v1 Duel: 8 points to win (462.3)
- Points from Burn Out (opponent gets 1) don't follow Final Point rules (448.1.a.1)

## Cleanup Steps (322)
Called MANY times: after state transitions, phase transitions, chain items added/removed, 
game objects enter/leave board, status changes, moves complete (319.1-319.7)

Steps in order:
1. Check victory score (322.1)
2. Handle board state:
   - 2a. Deathknell triggers for dying units (322.3)
   - 2b. Kill units with lethal damage (322.4)
3. Assign/remove Attacker/Defender designations if combat (322.5)
4. Uncontrolled battlefields with no units become Uncontrolled (322.6)
5. Recall gear at battlefields, remove invalid hidden cards (322.7)
6. Stage Showdowns at uncontrolled contested battlefields (322.8)
7. Stage Combats at contested battlefields with opposing units (322.10)
8. Finalize pending chain items (322.12)
9. If Neutral Open + Showdowns staged → Turn Player picks one, begin (322.13)
10. If Neutral Open + Combats staged → Turn Player picks one, begin (322.14)

## Movement (423-431)
- Standard Move = Discretionary Action, costs exhausting the unit(s) (407.3)
- Can move: Base → Battlefield, Battlefield → Base
- With Ganking: Battlefield → Battlefield
- Cannot move to battlefield with units from 2 other players (143.4.a.1)
- Moving to uncontrolled battlefield → Contested → Showdown (429.1)
- Moving to opponent-controlled battlefield → Contested → Combat (430.1)
- Move is instantaneous, cannot be reacted to (424.3.c)
