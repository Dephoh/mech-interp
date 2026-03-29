import { useState } from "react";
import type { CardView, ClientMessage } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface Props {
  hand: CardView[];
  send: (msg: ClientMessage) => void;
}

export function MulliganModal({ hand, send }: Props) {
  // Start with all cards kept; player clicks to toss (max 2 per rule 117.1)
  const [tossIndices, setTossIndices] = useState<Set<number>>(new Set());
  const [submitted, setSubmitted] = useState(false);

  function toggleCard(index: number) {
    if (submitted) return;
    setTossIndices((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else if (next.size < 2) {
        next.add(index);
      }
      return next;
    });
  }

  function handleKeepAll() {
    if (submitted) return;
    setTossIndices(new Set());
  }

  function handleSubmit() {
    if (submitted) return;
    setSubmitted(true);
    // Send keep_indices (all indices NOT in tossIndices)
    const keepIndices = hand
      .map((_, i) => i)
      .filter((i) => !tossIndices.has(i));
    send({
      type: "MULLIGAN_CHOICE",
      keep_indices: keepIndices,
    });
  }

  const tossCount = tossIndices.size;

  return (
    <div className="mulligan-overlay">
      <div className="mulligan-modal">
        <h2>Mulligan</h2>
        <p>Click up to 2 cards to set aside and redraw. (Rule 117)</p>

        <div className="mulligan-cards">
          {hand.map((card, i) => (
            <div
              key={card.instance_id}
              className={`mulligan-card ${tossIndices.has(i) ? "mulligan-card--toss" : "mulligan-card--keep"}`}
              onClick={() => toggleCard(i)}
            >
              <CardViewComponent card={card} />
              <div className="mulligan-label">
                {tossIndices.has(i) ? "TOSS" : "KEEP"}
              </div>
            </div>
          ))}
        </div>

        <div className="mulligan-actions">
          <button onClick={handleKeepAll} disabled={submitted}>Keep All</button>
          <button className="mulligan-submit" onClick={handleSubmit} disabled={submitted}>
            {submitted
              ? "Waiting for opponent..."
              : tossCount === 0
                ? "Keep All"
                : `Set aside ${tossCount} and redraw`}
          </button>
        </div>
      </div>
    </div>
  );
}
