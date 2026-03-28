import type { BattlefieldView } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface Props {
  battlefield: BattlefieldView;
  yourPlayerId: string;
  onUnitClick?: (instanceId: string) => void;
  onDropUnit?: (battlefieldId: string) => void;
}

export function BattlefieldZone({ battlefield, yourPlayerId, onUnitClick, onDropUnit }: Props) {
  const isControlled = battlefield.control_status === "controlled";
  const isYours = battlefield.controller_id === yourPlayerId;
  const isContested = battlefield.contested_by !== null;

  let statusLabel = "Uncontrolled";
  if (isControlled) {
    statusLabel = isYours ? "You control" : "Opponent controls";
  }
  if (isContested) {
    statusLabel += " (Contested)";
  }
  if (battlefield.combat_staged) {
    statusLabel += " [COMBAT]";
  }
  if (battlefield.showdown_staged) {
    statusLabel += " [SHOWDOWN]";
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    onDropUnit?.(battlefield.battlefield_id);
  }

  function handleZoneClick() {
    onDropUnit?.(battlefield.battlefield_id);
  }

  return (
    <div
      className={`zone zone--battlefield ${isControlled ? (isYours ? "bf--yours" : "bf--theirs") : "bf--neutral"}`}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={handleZoneClick}
    >
      <div className="zone__label">
        {battlefield.name ?? battlefield.battlefield_id}
        <span className="bf__status">{statusLabel}</span>
      </div>
      <div className="zone__cards">
        {battlefield.units.map((u) => (
          <CardViewComponent
            key={u.instance_id}
            card={u}
            compact
            onClick={() => onUnitClick?.(u.instance_id)}
          />
        ))}
        {battlefield.units.length === 0 && <span className="zone__empty">No units</span>}
      </div>
      {battlefield.facedown_card && (
        <div className="bf__facedown">
          <CardViewComponent card={battlefield.facedown_card} compact />
        </div>
      )}
    </div>
  );
}
