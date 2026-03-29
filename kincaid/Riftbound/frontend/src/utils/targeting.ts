/**
 * Shared targeting validation logic.
 * Determines whether a card qualifies as a valid target for a given target type.
 */
export function isValidTarget(
  card: { card_type?: string; controller_id?: string; owner_id?: string },
  targetType: string,
  yourPlayerId: string,
): boolean {
  const isYours = card.controller_id === yourPlayerId || card.owner_id === yourPlayerId;
  switch (targetType) {
    case "enemy_unit": return card.card_type === "unit" && !isYours;
    case "friendly_unit": return card.card_type === "unit" && isYours;
    case "unit": return card.card_type === "unit";
    case "enemy_gear": return card.card_type === "gear" && !isYours;
    case "friendly_gear": return card.card_type === "gear" && isYours;
    case "gear": return card.card_type === "gear";
    case "permanent": return card.card_type === "unit" || card.card_type === "gear";
    case "enemy_permanent": return (card.card_type === "unit" || card.card_type === "gear") && !isYours;
    case "friendly_permanent": return (card.card_type === "unit" || card.card_type === "gear") && isYours;
    default: return true;
  }
}
