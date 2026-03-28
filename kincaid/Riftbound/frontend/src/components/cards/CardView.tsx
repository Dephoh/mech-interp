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

export function CardViewComponent({ card, onClick, compact }: Props) {
  const selectCard = useUIStore((s) => s.selectCard);
  const selectedCardId = useUIStore((s) => s.selectedCardId);

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

  if (compact) {
    return (
      <div
        className={`card card--compact ${isSelected ? "card--selected" : ""} ${card.exhausted ? "card--exhausted" : ""}`}
        style={{ borderColor }}
        onClick={handleClick}
        draggable
        onDragStart={handleDragStart}
        title={card.text ?? ""}
      >
        <span className="card__name">{card.name}</span>
        {card.card_type === "unit" && (
          <span className="card__might">
            {card.damage ? `${card.effective_might! - card.damage}/${card.effective_might}` : card.effective_might}
          </span>
        )}
      </div>
    );
  }

  return (
    <div
      className={`card ${isSelected ? "card--selected" : ""} ${card.exhausted ? "card--exhausted" : ""}`}
      style={{ borderColor }}
      onClick={handleClick}
      draggable
      onDragStart={handleDragStart}
    >
      <div className="card__header">
        <span className="card__name">{card.name}</span>
        {card.cost_energy != null && (
          <span className="card__cost">{card.cost_energy}E</span>
        )}
      </div>
      <div className="card__type">{card.card_type}</div>
      {card.card_type === "unit" && (
        <div className="card__stats">
          <span>Might: {card.effective_might}</span>
          {(card.damage ?? 0) > 0 && <span className="card__damage"> DMG: {card.damage}</span>}
          {card.stunned && <span className="card__stunned"> STUNNED</span>}
        </div>
      )}
      {card.keywords && card.keywords.length > 0 && (
        <div className="card__keywords">
          {card.keywords.map((k) => (
            <span key={k.keyword} className="keyword-badge">
              {k.keyword}
              {k.value > 0 ? ` [${k.value}]` : ""}
            </span>
          ))}
        </div>
      )}
      {card.text && <div className="card__text">{card.text}</div>}
    </div>
  );
}
