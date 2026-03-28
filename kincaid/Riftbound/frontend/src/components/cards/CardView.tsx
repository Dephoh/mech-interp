import { useState } from "react";
import type { CardView as CardViewType } from "../../ws/messageTypes";
import { useUIStore } from "../../store/uiStore";

interface Props {
  card: CardViewType;
  onClick?: () => void;
  compact?: boolean;
}

const DOMAIN_COLORS: Record<string, string> = {
  fury: "#d32f2f",
  calm: "#388e3c",
  mind: "#1976d2",
  body: "#e65100",
  chaos: "#7b1fa2",
  order: "#f9a825",
};

function cardImageUrl(cardId: string | undefined): string | null {
  if (!cardId) return null;
  return `/card-images/${cardId}.png`;
}

export function CardViewComponent({ card, onClick, compact }: Props) {
  const selectCard = useUIStore((s) => s.selectCard);
  const selectedCardId = useUIStore((s) => s.selectedCardId);
  const [imgError, setImgError] = useState(false);

  if (card.facedown) {
    return (
      <div className="card card--facedown" onClick={onClick}>
        <div className="card__back">?</div>
      </div>
    );
  }

  const domain = card.domains?.[0];
  const borderColor = domain ? DOMAIN_COLORS[domain] ?? "#666" : "#666";
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
  }

  return (
    <div
      className={`card ${compact ? "card--compact" : ""} ${isSelected ? "card--selected" : ""} ${card.exhausted ? "card--exhausted" : ""}`}
      style={{ borderColor }}
      onClick={handleClick}
      draggable
      onDragStart={handleDragStart}
    >
      {imgSrc && !imgError ? (
        <img
          className={`card__art ${compact ? "card__art--compact" : ""}`}
          src={imgSrc}
          alt={card.name ?? ""}
          onError={() => setImgError(true)}
        />
      ) : (
        <div className={`card__art-missing ${compact ? "card__art-missing--compact" : ""}`} style={{ height: '100%', marginBottom: 0 }}>
          <span>NO ART</span>
          <span className="card__art-missing-id">{card.card_id ?? "?"}</span>
          {!compact && (
            <div className="card__stats" style={{marginTop: 8, textAlign: 'center'}}>
              <div>{card.name}</div>
              {card.cost_energy != null && <div>{card.cost_energy}E</div>}
              {card.card_type === "unit" && <div>{card.effective_might} Might</div>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
