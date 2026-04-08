import { isValidTarget } from "../../utils/targeting";
import type { CardView } from "../../ws/messageTypes";
import { CardViewComponent } from "../cards/CardView";

interface TargetingContext {
  targetType: string;
  yourPlayerId: string;
}

interface Props {
  units: CardView[];
  gear: CardView[];
  label: string;
  onUnitClick?: (instanceId: string) => void;
  onGearClick?: (instanceId: string) => void;
  onDropUnit?: () => void;
  highlightIds?: string[];
  targeting?: TargetingContext;
  activateAbility?: (instanceId: string, abilityId: string) => void;
}

export function BaseZone({ units, gear, label, onUnitClick, onGearClick, onDropUnit, highlightIds, targeting, activateAbility }: Props) {
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
          {units.map((u) => {
            const dimmed = targeting && !isValidTarget(u, targeting.targetType, targeting.yourPlayerId);
            const attachedGear = gear.filter((g) => g.attached_to === u.instance_id);
            return (
              <div key={u.instance_id} className="unit-with-gear">
                <CardViewComponent
                  card={u}
                  compact
                  zone="base"
                  highlighted={highlightIds?.includes(u.instance_id)}
                  dimmed={dimmed}
                  onClick={() => onUnitClick?.(u.instance_id)}
                />
                {attachedGear.length > 0 && (
                  <div className="attached-gear-row">
                    {attachedGear.map((g) => (
                      <div key={g.instance_id} className="attached-gear-pip" title={g.name ?? "Gear"}>
                        {(g.name ?? "G")[0]}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
          {units.length === 0 && <span className="zone__empty">No units</span>}
        </div>
      </div>
      {gear.length > 0 && (
        <div className="zone__section">
          <span className="zone__section-label">Gear</span>
          <div className="zone__cards">
            {gear.map((g) => {
              const dimmed = targeting && !isValidTarget(g, targeting.targetType, targeting.yourPlayerId);
              const activatedAbility = g.abilities?.find(ab => ab.ability_type === "activated");
              return (
                <div key={g.instance_id} className="gear-card-wrapper">
                  <CardViewComponent
                    card={g}
                    compact
                    zone="base"
                    highlighted={highlightIds?.includes(g.instance_id)}
                    dimmed={dimmed}
                    onClick={targeting ? () => onGearClick?.(g.instance_id) : undefined}
                  />
                  {g.attached_to && (
                    <span className="gear-attached-label">
                      Equipped
                    </span>
                  )}
                  {activateAbility && activatedAbility && (
                    <button
                      className="btn-activate"
                      onClick={(e) => { e.stopPropagation(); activateAbility(g.instance_id, activatedAbility.ability_id); }}
                      title={activatedAbility.text}
                    >
                      {activatedAbility.text.toLowerCase().includes("equip") ? "Equip" : "Act"}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
