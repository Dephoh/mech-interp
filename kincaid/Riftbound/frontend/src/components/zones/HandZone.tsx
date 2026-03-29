import type { CardView } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface Props {
  cards: CardView[];
  isOwn: boolean;
  onCardClick?: (instanceId: string) => void;
  /** Optional callback: returns true if the card should be dimmed (unplayable). */
  isCardDimmed?: (card: CardView) => boolean;
}

export function HandZone({ cards, isOwn, onCardClick, isCardDimmed }: Props) {
  return (
    <div className={`zone zone--hand ${isOwn ? "zone--own" : "zone--opponent"}`}>
      <div className="zone__label">
        Hand ({cards.length})
      </div>
      <div className="zone__cards">
        {cards.map((card) => (
          <CardViewComponent
            key={card.instance_id}
            card={card}
            zone="hand"
            onClick={() => isOwn && onCardClick?.(card.instance_id)}
            dimmed={isOwn && isCardDimmed?.(card)}
          />
        ))}
        {cards.length === 0 && <span className="zone__empty">Empty</span>}
      </div>
    </div>
  );
}
