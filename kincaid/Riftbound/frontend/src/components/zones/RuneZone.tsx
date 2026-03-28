import type { CardView } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface Props {
  runes: CardView[];
  runePool: { energy: number; power: Record<string, number> };
  runeDeckCount: number;
  onExhaust?: (instanceId: string) => void;
  onRecycle?: (instanceId: string) => void;
}

export function RuneZone({ runes, runePool, runeDeckCount, onExhaust, onRecycle }: Props) {
  const powerEntries = Object.entries(runePool.power).filter(([, v]) => v > 0);

  return (
    <div className="zone zone--runes">
      <div className="zone__label">Runes</div>
      <div className="rune-pool">
        <span className="rune-pool__energy">Energy: {runePool.energy}</span>
        {powerEntries.map(([domain, amount]) => (
          <span key={domain} className={`rune-pool__power rune-pool__power--${domain}`}>
            {domain}: {amount}
          </span>
        ))}
        <span className="rune-pool__deck">Deck: {runeDeckCount}</span>
      </div>
      <div className="zone__cards">
        {runes.map((r) => (
          <div key={r.instance_id} className="rune-card-wrapper">
            <CardViewComponent card={r} compact />
            <div className="rune-actions">
              {!r.exhausted && (
                <button className="btn-sm" onClick={() => onExhaust?.(r.instance_id)} title="Exhaust for Energy">
                  E
                </button>
              )}
              <button className="btn-sm" onClick={() => onRecycle?.(r.instance_id)} title="Recycle for Power">
                P
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
