import { useState } from "react";
import type { CardView } from "../../ws/messageTypes";

interface Props {
  runes: CardView[];
  runePool: { energy: number; power: Record<string, number> };
  runeDeckCount: number;
  onExhaust?: (instanceId: string) => void;
  onRecycle?: (instanceId: string) => void;
}

const DOMAIN_COLORS: Record<string, string> = {
  fury: "#d32f2f",
  calm: "#388e3c",
  mind: "#1976d2",
  body: "#e65100",
  chaos: "#7b1fa2",
  order: "#f9a825",
};

function RuneChip({ rune, onExhaust, onRecycle }: {
  rune: CardView;
  onExhaust?: (id: string) => void;
  onRecycle?: (id: string) => void;
}) {
  const [hoverTimer, setHoverTimer] = useState<ReturnType<typeof setTimeout> | null>(null);
  const [expanded, setExpanded] = useState(false);

  const domain = rune.domains?.[0];
  const color = domain ? DOMAIN_COLORS[domain] ?? "#888" : "#888";

  function handleMouseEnter() {
    const timer = setTimeout(() => setExpanded(true), 800);
    setHoverTimer(timer);
  }

  function handleMouseLeave() {
    if (hoverTimer) clearTimeout(hoverTimer);
    setHoverTimer(null);
    setExpanded(false);
  }

  return (
    <div
      className={`rune-chip ${rune.exhausted ? "rune-chip--exhausted" : ""}`}
      style={{ borderColor: color }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <span className="rune-chip__name" style={{ color }}>
        {rune.name ?? "Rune"}
      </span>
      <div className="rune-chip__buttons">
        {!rune.exhausted && onExhaust && (
          <button
            className="rune-btn rune-btn--energy"
            onClick={(e) => { e.stopPropagation(); onExhaust(rune.instance_id); }}
            title="Exhaust for Energy"
          >
            E
          </button>
        )}
        {onRecycle && (
          <button
            className="rune-btn rune-btn--power"
            onClick={(e) => { e.stopPropagation(); onRecycle(rune.instance_id); }}
            title="Recycle for Power"
            style={{ borderColor: color, color }}
          >
            P
          </button>
        )}
      </div>
      {expanded && rune.text && (
        <div className="rune-chip__tooltip">{rune.text}</div>
      )}
    </div>
  );
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
      <div className="rune-chip-list">
        {runes.map((r) => (
          <RuneChip
            key={r.instance_id}
            rune={r}
            onExhaust={onExhaust}
            onRecycle={onRecycle}
          />
        ))}
        {runes.length === 0 && <span className="zone__empty">No runes</span>}
      </div>
    </div>
  );
}
