import { useState } from "react";
import type { CardView, ClientMessage } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface Props {
  targets: CardView[];
  totalDamage: number;
  send: (msg: ClientMessage) => void;
}

export function DamageAssignmentModal({ targets, totalDamage, send }: Props) {
  const [assignments, setAssignments] = useState<Record<string, number>>(() => {
    const init: Record<string, number> = {};
    for (const t of targets) init[t.instance_id] = 0;
    return init;
  });
  const [submitted, setSubmitted] = useState(false);

  const assigned = Object.values(assignments).reduce((a, b) => a + b, 0);
  const remaining = totalDamage - assigned;

  function adjust(id: string, delta: number) {
    if (submitted) return;
    setAssignments((prev) => {
      const cur = prev[id] ?? 0;
      const next = Math.max(0, cur + delta);
      // Don't allow over-assignment
      const newTotal = assigned - cur + next;
      if (newTotal > totalDamage) return prev;
      return { ...prev, [id]: next };
    });
  }

  function handleSubmit() {
    if (submitted || remaining !== 0) return;
    setSubmitted(true);
    send({ type: "ASSIGN_DAMAGE", assignments });
  }

  return (
    <div className="mulligan-overlay">
      <div className="mulligan-modal">
        <h2>Assign Combat Damage</h2>
        <p>
          Distribute <strong>{totalDamage}</strong> damage across enemy units.
          {remaining > 0 && <span> ({remaining} remaining)</span>}
        </p>

        <div className="damage-targets">
          {targets.map((t) => (
            <div key={t.instance_id} className="damage-target">
              <CardViewComponent card={t} compact />
              <div className="damage-target__info">
                <span className="damage-target__name">{t.name}</span>
                <span className="damage-target__might">
                  {t.effective_might} Might
                  {(t.damage ?? 0) > 0 && ` (${t.damage} dmg)`}
                </span>
              </div>
              <div className="damage-target__controls">
                <button onClick={() => adjust(t.instance_id, -1)} disabled={submitted || (assignments[t.instance_id] ?? 0) <= 0}>
                  -
                </button>
                <span className="damage-target__value">{assignments[t.instance_id] ?? 0}</span>
                <button onClick={() => adjust(t.instance_id, 1)} disabled={submitted || remaining <= 0}>
                  +
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="mulligan-actions">
          <button className="mulligan-submit" onClick={handleSubmit} disabled={submitted || remaining !== 0}>
            {submitted ? "Waiting..." : `Confirm Damage (${assigned}/${totalDamage})`}
          </button>
        </div>
      </div>
    </div>
  );
}
