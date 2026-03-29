import { isValidTarget } from "../../utils/targeting";
import type { BattlefieldView, CardView } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface TargetingContext {
  targetType: string;
  yourPlayerId: string;
}

interface Props {
  battlefield: BattlefieldView;
  yourPlayerId: string;
  onUnitClick?: (instanceId: string) => void;
  onDropUnit?: (battlefieldId: string) => void;
  highlightIds?: string[];
  targeting?: TargetingContext;
}

function mightTotal(units: CardView[]): number {
  return units.reduce((sum, u) => sum + (u.effective_might ?? u.base_might ?? 0), 0);
}

export function BattlefieldZone({ battlefield, yourPlayerId, onUnitClick, onDropUnit, highlightIds, targeting }: Props) {
  const isControlled = battlefield.control_status === "controlled";
  const isYours = battlefield.controller_id === yourPlayerId;
  const isContested = battlefield.contested_by !== null;

  let statusLabel = "Uncontrolled";
  let statusClass = "";
  if (isControlled) {
    statusLabel = isYours ? "You control" : "Opponent controls";
  }
  if (isContested) {
    statusLabel += " [Contested]";
  }
  if (battlefield.combat_staged) {
    statusLabel += " [COMBAT]";
    statusClass = "bf__status--combat";
  }
  if (battlefield.showdown_staged) {
    statusLabel += " [SHOWDOWN]";
    statusClass = "bf__status--showdown";
  }

  const zoneClass = [
    "zone zone--battlefield",
    isControlled ? (isYours ? "bf--yours" : "bf--theirs") : "bf--neutral",
    isContested && "bf--contested",
  ].filter(Boolean).join(" ");

  // Split units by ownership
  const yourUnits = battlefield.units.filter(
    (u) => u.controller_id === yourPlayerId || u.owner_id === yourPlayerId
  );
  const oppUnits = battlefield.units.filter(
    (u) => u.controller_id !== yourPlayerId && u.owner_id !== yourPlayerId
  );

  const yourMight = mightTotal(yourUnits);
  const oppMight = mightTotal(oppUnits);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    onDropUnit?.(battlefield.battlefield_id);
  }

  function handleZoneClick() {
    onDropUnit?.(battlefield.battlefield_id);
  }

  function renderUnit(u: CardView) {
    const dimmed = targeting && !isValidTarget(u, targeting.targetType, targeting.yourPlayerId);
    return (
      <CardViewComponent
        key={u.instance_id}
        card={u}
        compact
        zone="battlefield"
        highlighted={highlightIds?.includes(u.instance_id)}
        dimmed={dimmed}
        onClick={() => onUnitClick?.(u.instance_id)}
      />
    );
  }

  const hasUnits = battlefield.units.length > 0;

  return (
    <div
      className={zoneClass}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      onClick={handleZoneClick}
    >
      <div className="zone__label">
        {battlefield.name ?? battlefield.battlefield_id}
        <span className={`bf__status ${statusClass}`}>{statusLabel}</span>
      </div>

      {hasUnits ? (
        <div className="bf__sides">
          {/* Opponent side (top) */}
          <div className="bf__side bf__side--opp">
            {oppUnits.length > 0 && (
              <span className="bf__might bf__might--opp">{oppMight}</span>
            )}
            <div className="bf__side-cards">
              {oppUnits.map(renderUnit)}
              {oppUnits.length === 0 && <span className="zone__empty">---</span>}
            </div>
          </div>

          {/* Divider */}
          <div className="bf__divider">
            <span className="bf__vs">vs</span>
          </div>

          {/* Your side (bottom) */}
          <div className="bf__side bf__side--you">
            <div className="bf__side-cards">
              {yourUnits.map(renderUnit)}
              {yourUnits.length === 0 && <span className="zone__empty">---</span>}
            </div>
            {yourUnits.length > 0 && (
              <span className="bf__might bf__might--you">{yourMight}</span>
            )}
          </div>
        </div>
      ) : (
        <div className="bf__sides">
          <span className="zone__empty">No units</span>
        </div>
      )}

      {battlefield.facedown_card && (
        <div className="bf__facedown">
          <CardViewComponent card={battlefield.facedown_card} compact zone="battlefield" />
        </div>
      )}
    </div>
  );
}
