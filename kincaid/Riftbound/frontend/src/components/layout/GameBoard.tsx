import { useGameStore } from "../../store/gameStore";
import { useUIStore } from "../../store/uiStore";
import type { ClientMessage } from "../../ws/messageTypes";
import { MulliganModal } from "../ui/MulliganModal";
import { BaseZone } from "../zones/BaseZone";
import { BattlefieldZone } from "../zones/BattlefieldZone";
import { HandZone } from "../zones/HandZone";
import { RuneZone } from "../zones/RuneZone";
import { Sidebar } from "./Sidebar";

interface Props {
  send: (msg: ClientMessage) => void;
}

export function GameBoard({ send }: Props) {
  const gs = useGameStore((s) => s.gameState);
  const gameOver = useGameStore((s) => s.gameOver);
  const winnerId = useGameStore((s) => s.winnerId);
  const lastError = useGameStore((s) => s.lastError);
  const selectedCardId = useUIStore((s) => s.selectedCardId);
  const selectCard = useUIStore((s) => s.selectCard);

  if (!gs) {
    return <div className="board-loading">Waiting for game state...</div>;
  }

  const you = gs.you;
  const opp = gs.opponent;
  const battlefields = Object.values(gs.battlefields);

  // Server-provided flags (rules-accurate)
  const hasPriority = gs.has_priority ?? (gs.active_player_id === you.player_id);
  const isYourTurn = gs.is_your_turn ?? (gs.turn_player_id === you.player_id);
  const canPlay = gs.can_play_cards ?? false;
  const isMulligan = gs.phase === "setup_mulligan";

  // During mulligan, show the mulligan modal
  if (isMulligan) {
    return (
      <div className="game-layout">
        <MulliganModal hand={you.hand} send={send} />
        <Sidebar />
      </div>
    );
  }

  function handlePlayCard(instanceId: string) {
    if (!canPlay) return;
    send({ type: "PLAY_CARD", instance_id: instanceId, targets: [] });
    selectCard(null);
  }

  function handleMoveUnit(battlefieldId: string) {
    if (!selectedCardId) return;
    send({
      type: "MOVE_UNIT",
      instance_id: selectedCardId,
      destination: { zone: "battlefield", id: battlefieldId },
    });
    selectCard(null);
  }

  function handleMoveToBase() {
    if (!selectedCardId) return;
    send({
      type: "MOVE_UNIT",
      instance_id: selectedCardId,
      destination: { zone: "base" },
    });
    selectCard(null);
  }

  function handleAdvancePhase() {
    send({ type: "ADVANCE_PHASE" });
  }

  function handlePassPriority() {
    send({ type: "PASS_PRIORITY" });
  }

  function handlePassFocus() {
    send({ type: "PASS_FOCUS" });
  }

  function handleExhaustRune(instanceId: string) {
    send({ type: "EXHAUST_RUNE", instance_id: instanceId });
  }

  function handleRecycleRune(instanceId: string) {
    send({ type: "RECYCLE_RUNE", instance_id: instanceId });
  }

  function handleConcede() {
    send({ type: "CONCEDE" });
  }

  // Pool display
  const pool = you.rune_pool;
  const poolParts: string[] = [];
  if (pool.energy > 0) poolParts.push(`${pool.energy}E`);
  for (const [dom, amt] of Object.entries(pool.power || {})) {
    if (amt > 0) poolParts.push(`${amt} ${dom}`);
  }
  const poolStr = poolParts.length > 0 ? poolParts.join(", ") : "empty";

  return (
    <div className="game-layout">
      <div className="game-board">
        {gameOver && (
          <div className="game-over-banner">
            {winnerId === you.player_id ? "You Win!" : "You Lose!"}
          </div>
        )}

        {lastError && (
          <div className="error-banner" onClick={() => useGameStore.getState().setError(null)}>
            {lastError}
          </div>
        )}

        {/* Status bar: whose turn, phase, pool */}
        <div className="status-bar">
          <span className="status-turn">
            {isYourTurn ? `⚔ Your Turn` : `⏳ ${opp.display_name}'s Turn`}
          </span>
          <span className="status-phase">
            Phase: {gs.phase.replace(/_/g, " ")}
          </span>
          <span className="status-pool">
            Pool: {poolStr}
          </span>
          <span className="status-score">
            You: {you.score} | {opp.display_name}: {opp.score}
          </span>
        </div>

        {/* Opponent's area */}
        <div className="board-row board-row--opponent">
          <HandZone cards={opp.hand} isOwn={false} />
          <BaseZone units={opp.base_units} gear={opp.base_gear} label={opp.display_name} />
        </div>

        {/* Battlefields (shared center) */}
        <div className="board-row board-row--battlefields">
          {battlefields.map((bf) => (
            <BattlefieldZone
              key={bf.battlefield_id}
              battlefield={bf}
              yourPlayerId={you.player_id}
              onUnitClick={(id) => selectCard(id)}
              onDropUnit={handleMoveUnit}
            />
          ))}
        </div>

        {/* Your area */}
        <div className="board-row board-row--you">
          <HandZone
            cards={you.hand}
            isOwn
            onCardClick={canPlay ? handlePlayCard : undefined}
          />
          <BaseZone
            units={you.base_units}
            gear={you.base_gear}
            label={you.display_name}
            onUnitClick={(id) => selectCard(id)}
            onDropUnit={handleMoveToBase}
          />
          <RuneZone
            runes={you.rune_board}
            runePool={you.rune_pool}
            runeDeckCount={you.rune_deck_count}
            onExhaust={handleExhaustRune}
            onRecycle={handleRecycleRune}
          />
        </div>

        {/* Action buttons */}
        <div className="action-bar">
          {/* Chain indicator */}
          {!gs.chain.is_empty && (
            <div className={`chain-indicator ${hasPriority ? 'chain-indicator--active' : ''}`}>
              ⚡ Chain active ({gs.chain.items.length} item{gs.chain.items.length !== 1 ? 's' : ''})
              {hasPriority
                ? ' — You have priority'
                : ` — Waiting for ${opp.display_name}`
              }
            </div>
          )}

          {/* End Phase (only turn player, only when chain empty) */}
          {isYourTurn && gs.chain.is_empty && !gs.showdown && (
            <button onClick={handleAdvancePhase}>End Phase</button>
          )}

          {/* Pass Priority (only when you have priority and chain exists) */}
          {hasPriority && !gs.chain.is_empty && (
            <button className="btn-priority" onClick={handlePassPriority}>
              ▶ Pass Priority
            </button>
          )}

          {/* Pass Focus (Showdown) */}
          {gs.showdown && gs.showdown.focus_player_id === you.player_id && (
            <button onClick={handlePassFocus}>Pass Focus</button>
          )}

          {selectedCardId && (
            <button onClick={() => selectCard(null)}>Deselect</button>
          )}
          <button className="btn-danger" onClick={handleConcede}>Concede</button>
        </div>
      </div>

      <Sidebar />
    </div>
  );
}

