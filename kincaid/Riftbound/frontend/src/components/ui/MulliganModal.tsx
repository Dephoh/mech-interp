import { useState } from "react";
import type { CardView, ClientMessage } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface Props {
  hand: CardView[];
  send: (msg: ClientMessage) => void;
}

export function MulliganModal({ hand, send }: Props) {
  const [keepIndices, setKeepIndices] = useState<Set<number>>(new Set());
  const [submitted, setSubmitted] = useState(false);

  function toggleCard(index: number) {
    if (submitted) return;
    setKeepIndices((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  }

  function handleKeepAll() {
    if (submitted) return;
    setKeepIndices(new Set(hand.map((_, i) => i)));
  }

  function handleSubmit() {
    if (submitted) return;
    setSubmitted(true);
    send({
      type: "MULLIGAN_CHOICE",
      keep_indices: Array.from(keepIndices).sort(),
    });
  }

  return (
    <div className="mulligan-overlay">
      <div className="mulligan-modal">
        <h2>Mulligan</h2>
        <p>Click cards to keep them. Unselected cards will be shuffled back and redrawn.</p>

        <div className="mulligan-cards">
          {hand.map((card, i) => (
            <div
              key={card.instance_id}
              className={`mulligan-card ${keepIndices.has(i) ? "mulligan-card--keep" : "mulligan-card--toss"}`}
              onClick={() => toggleCard(i)}
            >
              <CardViewComponent card={card} />
              <div className="mulligan-label">
                {keepIndices.has(i) ? "KEEP" : "TOSS"}
              </div>
            </div>
          ))}
        </div>

        <div className="mulligan-actions">
          <button onClick={handleKeepAll} disabled={submitted}>Keep All</button>
          <button onClick={() => !submitted && setKeepIndices(new Set())} disabled={submitted}>Toss All</button>
          <button className="mulligan-submit" onClick={handleSubmit} disabled={submitted}>
            {submitted
              ? "Waiting for opponent..."
              : `Confirm (${keepIndices.size} kept, ${hand.length - keepIndices.size} redrawn)`}
          </button>
        </div>
      </div>
    </div>
  );
}
