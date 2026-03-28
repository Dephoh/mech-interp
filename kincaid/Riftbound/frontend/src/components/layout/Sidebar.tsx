import { useGameStore } from "../../store/gameStore";

export function Sidebar() {
  const gs = useGameStore((s) => s.gameState);
  const logs = useGameStore((s) => s.gameLogs);

  if (!gs) return null;

  const isYourTurn = gs.turn_player_id === gs.you.player_id;
  const hasAction = gs.active_player_id === gs.you.player_id;

  return (
    <div className="sidebar">
      <div className="sidebar__scores">
        <div className="score-row">
          <span>{gs.you.display_name}</span>
          <span className="score-val">{gs.you.score} / 8</span>
        </div>
        <div className="score-row">
          <span>{gs.opponent.display_name}</span>
          <span className="score-val">{gs.opponent.score} / 8</span>
        </div>
      </div>

      <div className="sidebar__phase">
        <div>Turn {gs.turn_number}</div>
        <div className="phase-label">{gs.phase.replace(/_/g, " ").toUpperCase()}</div>
        <div className="turn-state">{gs.turn_state.replace(/_/g, " ")}</div>
        {isYourTurn && <div className="your-turn">Your Turn</div>}
        {hasAction && <div className="has-action">You have priority</div>}
      </div>

      {gs.showdown && (
        <div className="sidebar__showdown">
          <strong>Showdown</strong>
          <div>Focus: {gs.showdown.focus_player_id === gs.you.player_id ? "You" : "Opponent"}</div>
        </div>
      )}

      {gs.combat && (
        <div className="sidebar__combat">
          <strong>Combat</strong>
          <div>Phase: {gs.combat.phase}</div>
          <div>
            {gs.combat.attacker_id === gs.you.player_id ? "You attack" : "You defend"}
          </div>
        </div>
      )}

      <div className="sidebar__chain">
        <strong>Chain ({gs.chain.items.length})</strong>
        {gs.chain.items.map((item) => (
          <div key={item.item_id} className="chain-item">
            {item.card?.name ?? item.source_name ?? "Ability"} — {item.controller_id === gs.you.player_id ? "You" : "Opp"}
          </div>
        ))}
      </div>

      <div className="sidebar__log">
        <strong>Game Log</strong>
        <div className="log-entries">
          {logs.slice(-30).map((msg, i) => (
            <div key={i} className="log-entry">{msg}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
