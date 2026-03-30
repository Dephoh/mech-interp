/**
 * Centralised card-image URL builder.
 *
 * Reads `VITE_CARD_IMAGE_BASE` from the Vite env at runtime,
 * defaulting to `/card-images/` when the variable is not set.
 */
export function getCardImageUrl(cardId: string): string {
  const base = import.meta.env.VITE_CARD_IMAGE_BASE ?? "/card-images/";
  return `${base}${cardId}.png`;
}
