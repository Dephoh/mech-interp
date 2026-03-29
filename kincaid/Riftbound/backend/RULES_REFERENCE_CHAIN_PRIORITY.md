# Riftbound Chain & Priority - Rules Reference
> Source of truth: backend/data/rules/11_chain.txt, backend/data/rules/13_process_of_play.txt

## Chain Basics (327-330)
- Chain is a Non-Board Zone that temporarily exists when a card is played or ability activated
- Only ONE chain can exist at a time (329.1)
- Chain existing = Closed State (330.1)
- No chain = Open State (330.2)
- In Closed State: by default NOTHING can be played EXCEPT cards/abilities with Reaction keyword (330.1.a-b)

## Playing a Card Process (346-356)
1. **Remove from zone, put on chain** → becomes Pending → Closes State (351)
2. **Make choices** — targets, location for units, etc. (352)
   - Units: choose valid Location (base or controlled battlefield) (352.2)
   - Spells: choose targets as specified (352.4)
   - CRITICAL: "In order to put a spell or ability on the chain, valid choices must be made for all targets" (352.7)
3. **Determine Total Cost** — base cost + additional costs - discounts (353)
4. **Pay costs** — energy, power, non-standard costs (354)
   - CRITICAL (354.1.a): "During this step, the spell's controller can use activated abilities with the Reaction tag that Add resources"
   - This means rune tapping happens DURING cost payment, not before
5. **Check legality** — targets still valid, no illegal state (355)
6. **Finalize** (356):
   - Permanents (units/gear): leave chain, enter board immediately (356.2)
     - Units enter EXHAUSTED at chosen location (356.2.c)
     - Gear enters READY at player's Base (356.2.d)
   - Spells: stay on chain as Finalized Chain Item (356.3)
     - Other players can play Reactions before resolution

## Chain Resolution Steps (331-336)
1. **Finalize** — all Pending items complete playing steps (333)
   - Add abilities resolve IMMEDIATELY when finalized (333.1.c) — they do NOT wait on chain
2. **Execute** — Active Player can: play Reaction card, activate ability, or Pass Priority (334)
3. **Pass** — if ALL players passed without adding to chain → resolve (335)
4. **Resolve** — newest item resolves, effects execute (336)
   - If chain empty after → Open State
   - If chain not empty → controller of newest item gets priority

## Key Rule: Add Abilities Resolve Immediately (333.1.c, 416.2)
- Abilities that Add resources (energy/power) finalize and resolve IMMEDIATELY
- They do NOT linger on the chain
- Priority and Focus do NOT pass from Add abilities being finalized (416.2.a)
- This means: tapping a rune for energy is instant, not a chain item

## Rune Tapping Flow (from rules 160, 416)
Basic Rune has two abilities:
1. `[E]: [Reaction] — Add [1]` = Exhaust this rune → add 1 Energy to pool (instant)
2. `Recycle this: [Reaction] — Add [C]` = Recycle this rune → add 1 Power of its domain (instant)

Both are Reaction-timed, so they can be used:
- During Action Phase (open state)
- During cost payment of another card (354.1.a)
- During Closed States (because Reaction)
- During Showdowns (because Reaction grants Action)

## Target Validation on Play (352.5-352.16)
- Targets must be valid when the spell is FINALIZED on the chain (352.7)
- If all targets become invalid by resolution, spell still resolves but does nothing (356.3.e.10)
- Spell CANNOT be put on chain without valid targets unless it chooses "any number" or "up to" (352.13)
- "If all of a card's instructions are impossible, it is still played and resolved, but nothing happens" (055.1)

## Showdown Flow (337-345)
- Player who applied Contested status gets Focus first (341)
- Focus holder can: play Action/Reaction spell, activate ability, or Pass (344)
- If all players pass in sequence → Showdown closes (344.3.a)
- When chain resolves during showdown → Focus passes to next player (343)
