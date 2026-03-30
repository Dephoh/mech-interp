import { useState } from "react";
import { useGameStore } from "../../store/gameStore";
import { useUIStore } from "../../store/uiStore";
import { getCardImageUrl } from "../../utils/cardImages";
import type { CardView, ClientMessage } from "../../ws/messageTypes";

const DOMAIN_COLORS: Record<string, string> = {
  fury: "#e53935", calm: "#43a047", mind: "#1e88e5",
  body: "#ef6c00", chaos: "#8e24aa", order: "#fdd835",
};

function CardPreview({ card }: { card: CardView }) {
  const [imgError, setImgError] = useState(false);
  const imgSrc = card.card_id ? getCardImageUrl(card.card_id) : null;

  return (
    <div className="sidebar__preview">
      <div className="preview-card-img">
        {imgSrc && !imgError ? (
          <img
            src={imgSrc}
            alt={card.name ?? ""}
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="card__art-missing">
            <span>NO ART</span>
            <span>{card.card_id ?? "?"}</span>
          </div>
        )}
      </div>

      <div className="preview-info">
        <div className="preview-name">{card.name ?? "Unknown"}</div>
        <div className="preview-type">
          {card.card_type ?? ""}
          {card.domains?.length ? (
            <> &mdash; {card.domains.map((d) => (
              <span key={d} style={{ color: DOMAIN_COLORS[d] ?? "#888", fontWeight: 600 }}>
                {d}
              </span>
            )).reduce((prev, curr, i) => i === 0 ? [curr] : [...prev, " ", curr], [] as React.ReactNode[])}</>
          ) : null}
        </div>

        <div className="preview-stats">
          {card.cost_energy != null && (
            <div className="preview-stat">
              <span className="preview-stat-label">Cost:</span>
              <span className="preview-stat-value" style={{ color: "#ffc107" }}>{card.cost_energy}E</span>
              {card.cost_power && Object.entries(card.cost_power).map(([d, v]) => (
                v > 0 && <span key={d} className="preview-stat-value" style={{ color: DOMAIN_COLORS[d] }}>+{v}{d[0].toUpperCase()}</span>
              ))}
            </div>
          )}
          {card.card_type === "unit" && (
            <div className="preview-stat">
              <span className="preview-stat-label">Might:</span>
              <span className="preview-stat-value" style={{ color: "#ef5350" }}>{card.effective_might ?? card.base_might ?? "?"}</span>
              {(card.damage ?? 0) > 0 && (
                <span className="preview-stat-value" style={{ color: "#e53935" }}>({card.damage} dmg)</span>
              )}
            </div>
          )}
        </div>

        {card.keywords && card.keywords.length > 0 && (
          <div className="preview-keywords">
            {card.keywords.map((kw, i) => (
              <span key={i} className="keyword-badge">{kw.keyword}{kw.value ? ` ${kw.value}` : ""}</span>
            ))}
          </div>
        )}

        {card.text && (
          <div className="preview-text">{card.text}</div>
        )}
      </div>
    </div>
  );
}

interface SidebarProps {
  send?: (msg: ClientMessage) => void;
}

export function Sidebar({ send }: SidebarProps) {
  const gs = useGameStore((s) => s.gameState);
  const logs = useGameStore((s) => s.gameLogs);
  const roomId = useGameStore((s) => s.roomId);
  const previewCard = useUIStore((s) => s.previewCard);

  if (!gs) return null;

  const isYourTurn = gs.is_your_turn ?? (gs.turn_player_id === gs.you.player_id);
  const hasPriority = gs.has_priority ?? (gs.active_player_id === gs.you.player_id);

  function handlePassPriority() {
    send?.({ type: "PASS_PRIORITY" });
  }

  return (
    <div className="sidebar">
      {/* Turn info header */}
      <div className="sidebar__header">
        <div className="sidebar__turn-info">
          <div className="sidebar__turn-number">Turn {gs.turn_number}</div>
          <div className={`sidebar__turn-who ${isYourTurn ? "sidebar__turn-who--you" : "sidebar__turn-who--opp"}`}>
            {isYourTurn ? "Your turn" : `${gs.opponent.display_name}'s turn`}
          </div>
        </div>
        <div>
          {roomId && <div className="sidebar__room">ROOM: {roomId.slice(0, 6).toUpperCase()}</div>}
        </div>
      </div>

      {/* Phase bar */}
      <div className="sidebar__phase-bar">
        <span className="phase-badge">{gs.phase.replace(/_/g, " ")}</span>
        <span className="turn-state-label">{gs.turn_state.replace(/_/g, " ")}</span>
        {hasPriority && <span className="priority-badge">Priority</span>}
      </div>

      {/* Game log */}
      <div className="sidebar__log">
        <div className="sidebar__log-header">Game Log</div>
        <div className="log-entries">
          {logs.slice(-30).reverse().map((msg, i) => (
            <div key={i} className="log-entry">{msg}</div>
          ))}
        </div>
      </div>

      {/* Showdown indicator */}
      {gs.showdown && (
        <div className="sidebar__indicator">
          <strong>Showdown</strong>
          <div>Focus: {gs.showdown.focus_player_id === gs.you.player_id ? "You" : gs.opponent.display_name}</div>
          {gs.showdown.is_combat_showdown && <div>Combat Showdown</div>}
        </div>
      )}

      {/* Combat indicator */}
      {gs.combat && (
        <div className="sidebar__indicator">
          <strong>Combat</strong>
          <div>Phase: {gs.combat.phase.replace(/_/g, " ")}</div>
          <div>{gs.combat.attacker_id === gs.you.player_id ? "You attack" : "You defend"}</div>
        </div>
      )}

      {/* Chain section */}
      <div className="sidebar__chain">
        <div className="sidebar__chain-header">
          <span className="sidebar__chain-title">Chain ({gs.chain.items.length})</span>
        </div>

        {gs.chain.items.map((item) => (
          <div key={item.item_id} className="chain-item">
            {item.card?.name ?? item.source_name ?? "Ability"} &mdash; {item.controller_id === gs.you.player_id ? "You" : "Opp"}
          </div>
        ))}

        {!gs.chain.is_empty && hasPriority && (
          <>
            <div className="chain-status chain-status--approved">
              {gs.opponent.display_name} Approved
            </div>
            <button className="btn-resolve" onClick={handlePassPriority}>
              Resolve (S)
            </button>
          </>
        )}

        {gs.chain.is_empty && (
          <div className="chain-status">No active chain</div>
        )}
      </div>

      {/* Card preview */}
      {previewCard ? (
        <CardPreview card={previewCard} />
      ) : (
        <div className="sidebar__preview">
          <div className="preview-empty">Hover a card to preview</div>
        </div>
      )}
    </div>
  );
}
