import type { CardView } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface Props {
  cards: CardView[];
  isOwn: boolean;
  onCardClick?: (instanceId: string) => void;
}

export function HandZone({ cards, isOwn, onCardClick }: Props) {
  return (
    <div className={`zone zone--hand ${isOwn ? "zone--own" : "zone--opponent"}`}>
      <div className="zone__label">Hand ({cards.length})</div>
      <div className="zone__cards">
        {cards.map((card) => (
          <CardViewComponent
            key={card.instance_id}
            card={card}
            onClick={() => isOwn && onCardClick?.(card.instance_id)}
          />
        ))}
      </div>
    </div>
  );
}
