import type { CardView } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface Props {
  units: CardView[];
  gear: CardView[];
  label: string;
  onUnitClick?: (instanceId: string) => void;
  onDropUnit?: () => void;
}

export function BaseZone({ units, gear, label, onUnitClick, onDropUnit }: Props) {
  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    onDropUnit?.();
  }

  return (
    <div 
      className="zone zone--base"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={() => onDropUnit?.()}
    >
      <div className="zone__label">{label} Base</div>
      <div className="zone__section">
        <span className="zone__section-label">Units</span>
        <div className="zone__cards">
          {units.map((u) => (
            <CardViewComponent
              key={u.instance_id}
              card={u}
              compact
              onClick={() => onUnitClick?.(u.instance_id)}
            />
          ))}
          {units.length === 0 && <span className="zone__empty">No units</span>}
        </div>
      </div>
      {gear.length > 0 && (
        <div className="zone__section">
          <span className="zone__section-label">Gear</span>
          <div className="zone__cards">
            {gear.map((g) => (
              <CardViewComponent key={g.instance_id} card={g} compact />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
