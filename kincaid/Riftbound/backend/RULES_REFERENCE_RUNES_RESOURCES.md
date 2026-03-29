# Riftbound Rune, Resource & Mulligan - Rules Reference
> Source of truth: backend/data/rules/05_runes.txt, backend/data/rules/08_costs.txt, backend/data/rules/16_stun_buff_channel.txt

## Rune System (156-164)

### Basic Runes (160)
Six types: Fury [R], Calm [G], Mind [B], Body [O], Chaos [P], Order [Y]
Each has TWO abilities:
1. `[E]: [Reaction] — Add [1]` → Exhaust rune, add 1 Energy (instant)
2. `Recycle this: [Reaction] — Add [C]` → Recycle rune to bottom of Rune Deck, add 1 Power of its Domain (instant)

### Rune Pool (161-164)
- Conceptual collection of available Energy and Power (162)
- Players must ADD resources to pool BEFORE spending them (162.2)
- **Pool empties at TWO times per turn** (163):
  1. End of Draw Phase (315.4.d)
  2. End of Expiration Step / End of Turn (317.3.b)

### Add Action (416)
- "Add" means putting resources into Rune Pool (416.1)
- Add abilities FINALIZE IMMEDIATELY (416.2) — do NOT stay on chain
- Priority/Focus do NOT pass from Add abilities (416.2.a)
- Reaction-timed Add abilities can be used DURING cost payment (416.3)

### Channel Action (417)
- Taking runes from top of Rune Deck and putting on board (417.1)
- Runes enter READY by default (unless specified otherwise)
- This is just placing them — does NOT add resources to pool

### Cost Payment Flow
When a player wants to play a card:
1. Declare card → goes on chain as Pending
2. Make choices (targets, location)
3. Determine total cost
4. **Pay costs** (step 354):
   - Player can use Reaction-timed Add abilities HERE (354.1.a)
   - This means: exhaust runes for Energy, recycle runes for Power
   - These Add abilities resolve instantly (416.2)
5. Check legality
6. Finalize

### Energy vs Power
- **Energy**: Generic resource, no domain. Pays numeric cost (159.1)
- **Power**: Domain-specific resource. Pays power symbols (159.2)
  - Power of specific domain pays only that domain's cost
  - Universal Power [A] pays any domain (134.2.e.5)
  - [C] = "Power of this card's Domain" (134.2.e.6)

### Practical Example: Playing Yordle Scout (costs 1 Energy + 1 Calm Power)
1. Player declares playing Yordle Scout
2. During cost payment, player:
   - Exhausts any rune → Add [1] → pool now has 1 Energy
   - Recycles a Calm Rune → Add [G] → pool now has 1 Calm Power
3. Spend 1 Energy + 1 Calm Power from pool
4. Card is finalized

## Burn Out (418)
- Triggered when attempting to move cards from Main Deck beyond remaining count
- Steps:
  1. Do as much of the action as possible
  2. Recycle entire Trash into Main Deck (randomized)
  3. Choose an opponent to gain 1 point
  4. Complete the original action
- Can repeat if deck+trash both empty → opponent gets points until they win

## Mulligan (116-117)
1. Each player draws 4 (116)
2. In turn order, each player performs Mulligan (117):
   - Choose up to 2 cards to set aside (117.1)
   - Draw as many as they set aside (117.2)
   - **RECYCLE** the set-aside cards (117.3)
     - This means: put them on bottom of Main Deck (403.1.a)
     - NOT trash, NOT remove from game

## Recycle Action (403)
- Take cards from a zone → put on bottom of corresponding deck (403.1)
- Main Deck cards → Main Deck bottom
- Runes → Rune Deck bottom
- If 2+ cards recycled to Main Deck simultaneously → random order (403.5)
- If 2+ cards recycled to Rune Deck simultaneously → owner's chosen order (403.5.a)
