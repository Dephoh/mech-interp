import { createPortal } from "react-dom";
import { useUIStore } from "../../store/uiStore";
import { getCardImageUrl } from "../../utils/cardImages";

const PORTAL_W = 275;
const PORTAL_H = 385;
const GAP = 12;

function cardImageUrl(cardId: string | undefined): string | null {
  if (!cardId) return null;
  return getCardImageUrl(cardId);
}

export function CardHoverPortal() {
  const hoverPortal = useUIStore((s) => s.hoverPortal);

  if (!hoverPortal) return null;
  const { card, rect, zone } = hoverPortal;
  if (card.facedown) return null;

  const vw = window.innerWidth;
  const vh = window.innerHeight;

  // Horizontal: prefer right of card, flip left if too close to right edge
  let left: number;
  if (rect.right + GAP + PORTAL_W < vw) {
    left = rect.right + GAP;
  } else if (rect.left - GAP - PORTAL_W > 0) {
    left = rect.left - GAP - PORTAL_W;
  } else {
    left = Math.max(4, vw - PORTAL_W - 4);
  }

  // Vertical: for hand zone, show above the card; otherwise center vertically on card
  let top: number;
  if (zone === "hand") {
    top = rect.top - PORTAL_H - GAP;
  } else {
    top = rect.top + rect.height / 2 - PORTAL_H / 2;
  }

  // Clamp to viewport
  top = Math.max(4, Math.min(top, vh - PORTAL_H - 4));

  const imgSrc = cardImageUrl(card.card_id);

  const portal = (
    <div
      className="card-hover-portal"
      style={{ left, top }}
    >
      {imgSrc ? (
        <img src={imgSrc} alt={card.name ?? ""} />
      ) : (
        <div className="card-hover-portal__missing">
          <span>{card.card_id ?? "?"}</span>
        </div>
      )}
    </div>
  );

  return createPortal(portal, document.body);
}
