import { useRef, useState } from "react";
import type { CardView as CardViewType } from "../../ws/messageTypes";
import { useUIStore } from "../../store/uiStore";
import { getCardImageUrl } from "../../utils/cardImages";

interface Props {
  card: CardViewType;
  onClick?: () => void;
  compact?: boolean;
  highlighted?: boolean;
  dimmed?: boolean;
  zone?: "hand" | "battlefield" | "base" | "other";
}

const DOMAIN_COLORS: Record<string, string> = {
  fury: "#e53935",
  calm: "#43a047",
  mind: "#1e88e5",
  body: "#ef6c00",
  chaos: "#8e24aa",
  order: "#fdd835",
};

function cardImageUrl(cardId: string | undefined): string | null {
  if (!cardId) return null;
  return getCardImageUrl(cardId);
}

export function CardViewComponent({ card, onClick, compact, highlighted, dimmed, zone = "other" }: Props) {
  const selectCard = useUIStore((s) => s.selectCard);
  const selectedCardId = useUIStore((s) => s.selectedCardId);
  const setHoverPortal = useUIStore((s) => s.setHoverPortal);
  const clearHoverPortal = useUIStore((s) => s.clearHoverPortal);
  const ref = useRef<HTMLDivElement>(null);
  const [imgError, setImgError] = useState(false);

  if (card.facedown) {
    return (
      <div className="card card--facedown" onClick={onClick}>
        <div className="card__back" />
      </div>
    );
  }

  const domain = card.domains?.[0];
  const borderColor = domain ? DOMAIN_COLORS[domain] ?? "#244060" : "#244060";
  const isSelected = selectedCardId === card.instance_id;
  const imgSrc = cardImageUrl(card.card_id);

  function handleClick() {
    selectCard(isSelected ? null : card.instance_id);
    onClick?.();
  }

  function handleDragStart(e: React.DragEvent) {
    if (!isSelected) {
      selectCard(card.instance_id);
      onClick?.();
    }
    e.dataTransfer.setData("text/plain", card.instance_id);
    e.dataTransfer.effectAllowed = "move";
    clearHoverPortal();
  }

  function handleMouseEnter() {
    if (ref.current) {
      setHoverPortal(card, ref.current.getBoundingClientRect(), zone);
    }
  }

  function handleMouseLeave() {
    clearHoverPortal();
  }

  return (
    <div
      ref={ref}
      className={[
        "card",
        compact && "card--compact",
        isSelected && "card--selected",
        highlighted && "card--highlighted",
        card.exhausted && "card--exhausted",
        card.stunned && "card--stunned",
        card.buff_counter && "card--buffed",
        dimmed && "card--dimmed",
      ].filter(Boolean).join(" ")}
      style={{ borderColor }}
      onClick={handleClick}
      draggable
      onDragStart={handleDragStart}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {imgSrc && !imgError ? (
        <img
          className="card__art"
          src={imgSrc}
          alt={card.name ?? ""}
          onError={() => setImgError(true)}
        />
      ) : (
        <div className="card__art-missing">
          <span>{card.card_id ?? "?"}</span>
        </div>
      )}
      {/* Buff badge */}
      {card.buff_counter && (
        <span className="buff-badge">&#x25B2;</span>
      )}
      {/* Stun badge */}
      {card.stunned && (
        <span className="stun-badge">&#x2744;</span>
      )}
      {/* Damage badge */}
      {(card.damage ?? 0) > 0 && (
        <span className="damage-badge">{card.damage}</span>
      )}
      {/* Might display when effective differs from base */}
      {card.card_type === "unit" && card.effective_might != null && card.base_might != null && card.effective_might !== card.base_might && (
        <span className="might-badge">{card.effective_might}/{card.base_might}</span>
      )}
    </div>
  );
}
