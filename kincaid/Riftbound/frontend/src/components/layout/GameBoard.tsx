import { useCallback, useEffect, useState } from "react";
import { useGameStore } from "../../store/gameStore";
import { useTestLabStore } from "../../store/testlabStore";
import { useUIStore } from "../../store/uiStore";
import { isValidTarget } from "../../utils/targeting";
import type { CardView, ClientMessage } from "../../ws/messageTypes";
import { TestLabToolbar } from "../testlab/TestLabToolbar";
import { AccelerateModal } from "../ui/AccelerateModal";
import { ChoiceModal } from "../ui/ChoiceModal";
import { DamageAssignmentModal } from "../ui/DamageAssignmentModal";
import { MulliganModal } from "../ui/MulliganModal";
import { BaseZone } from "../zones/BaseZone";
import { BattlefieldZone } from "../zones/BattlefieldZone";
import { HandZone } from "../zones/HandZone";
import { RuneZone } from "../zones/RuneZone";
import { Sidebar } from "./Sidebar";

interface Props {
  send: (msg: ClientMessage) => void;
}

/* ── Score Tracker (left column) ───────────── */
function ScoreTracker({ yourScore, oppScore, yourName, oppName }: {
  yourScore: number; oppScore: number; yourName: string; oppName: string;
}) {
  const pips = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  return (
    <div className="score-tracker">
      {/* Opponent score: 0 at top, 8 at bottom */}
      <div className="score-section">
        <span className="score-name">{oppName}</span>
        {pips.map((n) => (
          <div
            key={`opp-${n}`}
            className={`score-pip ${
              n === oppScore ? "score-pip--opp-active" :
              n < oppScore ? "score-pip--opp-filled" : ""
            }`}
          >
            {n}
          </div>
        ))}
      </div>

      <div className="score-divider" />

      {/* Your score: 8 at top, 0 at bottom */}
      <div className="score-section score-section--you">
        {[...pips].reverse().map((n) => (
          <div
            key={`you-${n}`}
            className={`score-pip ${
              n === yourScore ? "score-pip--active" :
              n < yourScore ? "score-pip--filled" : ""
            }`}
          >
            {n}
          </div>
        ))}
        <span className="score-name">{yourName}</span>
      </div>
    </div>
  );
}

/* ── Floating Resources ────────────────────── */
function FloatingResources({ pool }: { pool: { energy: number; power: Record<string, number> } }) {
  const DOMAIN_SHORT: Record<string, string> = {
    fury: "R", calm: "G", mind: "B", body: "O", chaos: "P", order: "Y",
  };
  const DOMAIN_COLORS: Record<string, string> = {
    fury: "#e53935", calm: "#43a047", mind: "#1e88e5",
    body: "#ef6c00", chaos: "#8e24aa", order: "#fdd835",
  };
  const powerEntries = Object.entries(pool.power || {}).filter(([, v]) => v > 0);

  return (
    <div className="floating-resources">
      <span className="floating-label">Floating</span>
      <div className="floating-row">
        <span>Energy</span>
        <span className="energy-val">{pool.energy}</span>
      </div>
      {powerEntries.length > 0 && (
        <div className="floating-row">
          <span>Power</span>
          {powerEntries.map(([dom, amt]) => (
            <span key={dom} className="power-val" style={{ color: DOMAIN_COLORS[dom] }}>
              {amt}{DOMAIN_SHORT[dom]}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Card Thumbnail (Legend / Champion) ─────── */
function CardThumb({ card, label }: { card: CardView | null; label: string }) {
  const setPreviewCard = useUIStore((s) => s.setPreviewCard);
  const [imgErr, setImgErr] = useState(false);
  if (!card) return null;
  const src = card.card_id ? `/card-images/${card.card_id}.png` : null;
  return (
    <div
      className="card-thumb"
      title={`${label}: ${card.name ?? "?"}`}
      onMouseEnter={() => setPreviewCard(card)}
      onMouseLeave={() => setPreviewCard(null)}
    >
      {src && !imgErr ? (
        <img src={src} alt={card.name ?? label} onError={() => setImgErr(true)} />
      ) : (
        <span className="card-thumb__fallback">{label[0]}</span>
      )}
    </div>
  );
}

/* ── Deck Icon ─────────────────────────────── */
function DeckIcon({ count, isRune }: { count: number; isRune?: boolean }) {
  return (
    <div className="deck-icon">
      <div className={`card-back ${isRune ? "card-back--rune" : ""}`} />
      <span className="deck-count">{count}</span>
    </div>
  );
}

export function GameBoard({ send }: Props) {
  const gs = useGameStore((s) => s.gameState);
  const gameOver = useGameStore((s) => s.gameOver);
  const winnerId = useGameStore((s) => s.winnerId);
  const lastError = useGameStore((s) => s.lastError);
  const isTestLab = useTestLabStore((s) => s.isTestLab);
  const selectedCardId = useUIStore((s) => s.selectedCardId);
  const selectCard = useUIStore((s) => s.selectCard);
  const targeting = useUIStore((s) => s.targeting);
  const enterTargeting = useUIStore((s) => s.enterTargeting);
  const enterAbilityTargeting = useUIStore((s) => s.enterAbilityTargeting);
  const addTarget = useUIStore((s) => s.addTarget);
  const cancelTargeting = useUIStore((s) => s.cancelTargeting);
  const pendingChoice = useUIStore((s) => s.pendingChoice);
  const clearPendingChoice = useUIStore((s) => s.clearPendingChoice);
  const showDamageModal = useUIStore((s) => s.showDamageModal);
  const setDamageModal = useUIStore((s) => s.setDamageModal);
  const pendingAccelerate = useUIStore((s) => s.pendingAccelerate);
  const setPendingAccelerate = useUIStore((s) => s.setPendingAccelerate);
  const clearPendingAccelerate = useUIStore((s) => s.clearPendingAccelerate);

  // ESC to cancel targeting
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && targeting) cancelTargeting();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [targeting, cancelTargeting]);

  // Auto-confirm when enough targets selected
  const handleConfirmTargets = useCallback(() => {
    if (!targeting) return;
    if (targeting.actionType === "activate_ability") {
      send({
        type: "ACTIVATE_ABILITY",
        source_id: targeting.sourceId!,
        ability_id: targeting.abilityId!,
        targets: targeting.selectedTargets,
      });
    } else {
      send({
        type: "PLAY_CARD",
        instance_id: targeting.cardId,
        targets: targeting.selectedTargets,
      });
    }
    cancelTargeting();
  }, [targeting, send, cancelTargeting]);

  useEffect(() => {
    if (targeting && targeting.selectedTargets.length >= targeting.targetsNeeded) {
      handleConfirmTargets();
    }
  }, [targeting?.selectedTargets.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // Trigger damage assignment modal when combat requires it
  useEffect(() => {
    if (!gs?.combat) return;
    const combat = gs.combat;
    const youId = gs.you.player_id;
    const isAttacker = combat.attacker_id === youId;
    const isDefender = combat.defender_id === youId;
    const needsAssignment =
      combat.phase === "damage_assignment" &&
      ((isAttacker && !combat.attacker_assigned) ||
       (isDefender && !combat.defender_assigned));
    if (needsAssignment && !showDamageModal) {
      setDamageModal(true);
    } else if (!needsAssignment && showDamageModal) {
      setDamageModal(false);
    }
  }, [gs?.combat, gs?.you.player_id, showDamageModal, setDamageModal]);

  if (!gs) {
    return <div className="board-loading">Waiting for game state...</div>;
  }

  const you = gs.you;
  const opp = gs.opponent;
  const battlefields = Object.values(gs.battlefields);

  const hasPriority = gs.has_priority ?? (gs.active_player_id === you.player_id);
  const isYourTurn = gs.is_your_turn ?? (gs.turn_player_id === you.player_id);
  const canPlay = gs.can_play_cards ?? false;
  const isMulligan = gs.phase === "setup_mulligan";

  if (isMulligan) {
    return (
      <div className="game-layout">
        <ScoreTracker yourScore={you.score} oppScore={opp.score} yourName={you.display_name} oppName={opp.display_name} />
        <MulliganModal hand={you.hand} send={send} />
        <Sidebar send={send} />
      </div>
    );
  }

  function handleTargetClick(card: CardView) {
    if (!targeting) return;
    if (isValidTarget(card, targeting.targetType, you.player_id)) {
      addTarget(card.instance_id);
    }
  }

  /** Check if a card has the accelerate keyword and prompt the player. */
  function checkAccelerate(card: CardView): boolean {
    if (!card.keywords) return false;
    return card.keywords.some((k) => k.keyword === "accelerate");
  }

  /**
   * Determine if a hand card should be dimmed (visually unplayable).
   * Rules 732/739: Action cards can play in Open states and Showdowns;
   * Reaction cards can also play in Closed states.
   * Cards without either keyword can only play in Neutral Open.
   */
  function isHandCardDimmed(card: CardView): boolean {
    if (!canPlay) return true;
    const kws = card.keywords ?? [];
    const hasReaction = kws.some((k) => k.keyword === "reaction");
    const hasAction = kws.some((k) => k.keyword === "action");
    const turnState = gs.turn_state ?? "";
    if (turnState === "neutral_closed" || turnState === "showdown_closed") {
      return !hasReaction;
    }
    if (turnState === "showdown_open") {
      return !(hasAction || hasReaction);
    }
    // neutral_open: all cards playable for turn player (canPlay handles turn check)
    return false;
  }

  function handlePlayCard(instanceId: string) {
    if (!canPlay) {
      const phase = gs?.phase ?? "unknown";
      useGameStore.getState().setError(
        `Can't play cards right now (phase: ${phase}, ${isYourTurn ? "your turn" : "opponent's turn"})`
      );
      return;
    }
    const card = you.hand.find((c) => c.instance_id === instanceId);
    if (!card) return;
    if (card.abilities) {
      const needsTargets = card.abilities.find((ab) => ab.targets_required > 0);
      if (needsTargets) {
        enterTargeting(instanceId, needsTargets.targets_required, needsTargets.target_type);
        return;
      }
    }
    // Rule 352.2: Units choose a destination (base or controlled battlefield)
    if (card.card_type === "unit") {
      const controlled = battlefields.filter(
        (bf) => bf.control_status === "controlled" && bf.controller_id === you.player_id
      );
      if (controlled.length > 0) {
        // Player has controlled battlefields — select card and let them click a destination
        selectCard(instanceId);
        return;
      }
      // No controlled battlefields — auto-send to base
      if (checkAccelerate(card)) {
        setPendingAccelerate({
          cardId: instanceId,
          cardName: card.name ?? "this unit",
          domain: card.domains?.[0] ?? "any",
          destination: { zone: "base" },
        });
        return;
      }
      send({
        type: "PLAY_CARD",
        instance_id: instanceId,
        targets: [],
        destination: { zone: "base" },
      });
      selectCard(null);
      return;
    }
    send({ type: "PLAY_CARD", instance_id: instanceId, targets: [] });
    selectCard(null);
  }

  function handlePlayUnitToBattlefield(battlefieldId: string) {
    if (!selectedCardId) return;
    const inHand = you.hand.find((c) => c.instance_id === selectedCardId);
    if (inHand && inHand.card_type === "unit") {
      if (checkAccelerate(inHand)) {
        setPendingAccelerate({
          cardId: selectedCardId,
          cardName: inHand.name ?? "this unit",
          domain: inHand.domains?.[0] ?? "any",
          destination: { zone: "battlefield", id: battlefieldId },
        });
        return;
      }
      send({
        type: "PLAY_CARD",
        instance_id: selectedCardId,
        targets: [],
        destination: { zone: "battlefield", id: battlefieldId },
      });
      selectCard(null);
      return;
    }
    // Not a hand unit — treat as move
    handleMoveUnit(battlefieldId);
  }

  function handlePlayUnitToBase() {
    if (!selectedCardId) return;
    const inHand = you.hand.find((c) => c.instance_id === selectedCardId);
    if (inHand && inHand.card_type === "unit") {
      if (checkAccelerate(inHand)) {
        setPendingAccelerate({
          cardId: selectedCardId,
          cardName: inHand.name ?? "this unit",
          domain: inHand.domains?.[0] ?? "any",
          destination: { zone: "base" },
        });
        return;
      }
      send({
        type: "PLAY_CARD",
        instance_id: selectedCardId,
        targets: [],
        destination: { zone: "base" },
      });
      selectCard(null);
      return;
    }
    // Not a hand unit — treat as move
    handleMoveToBase();
  }

  function handleAccelerateChoice(payAccelerate: boolean) {
    if (!pendingAccelerate) return;
    send({
      type: "PLAY_CARD",
      instance_id: pendingAccelerate.cardId,
      targets: [],
      destination: pendingAccelerate.destination,
      pay_accelerate: payAccelerate,
    });
    selectCard(null);
    clearPendingAccelerate();
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

  function handleActivateAbility(sourceId: string, abilityId: string) {
    // Find the ability on the source card
    const allCards = [...you.base_gear, ...you.base_units, ...you.rune_board];
    const source = allCards.find((c) => c.instance_id === sourceId);
    const ability = source?.abilities?.find((ab) => ab.ability_id === abilityId);
    if (ability && ability.targets_required > 0) {
      enterAbilityTargeting(sourceId, abilityId, ability.targets_required, ability.target_type);
      return;
    }
    // No targets needed — send directly
    send({ type: "ACTIVATE_ABILITY", source_id: sourceId, ability_id: abilityId, targets: [] });
  }

  function handleAdvancePhase() { send({ type: "ADVANCE_PHASE" }); }
  function handlePassPriority() { send({ type: "PASS_PRIORITY" }); }
  function handlePassFocus() { send({ type: "PASS_FOCUS" }); }
  function handleExhaustRune(instanceId: string) { send({ type: "EXHAUST_RUNE", instance_id: instanceId }); }
  function handleRecycleRune(instanceId: string) { send({ type: "RECYCLE_RUNE", instance_id: instanceId }); }
  function handleConcede() { send({ type: "CONCEDE" }); }

  const hasBaseUnitsOpp = opp.base_units.length > 0 || opp.base_gear.length > 0;
  const hasBaseUnitsYou = you.base_units.length > 0 || you.base_gear.length > 0;
  const isPlacingUnit = selectedCardId && you.hand.some(
    (c) => c.instance_id === selectedCardId && c.card_type === "unit"
  );

  return (
    <div className={`game-layout ${isTestLab ? "game-layout--testlab" : ""}`}>
      {/* Score Tracker */}
      <ScoreTracker
        yourScore={you.score}
        oppScore={opp.score}
        yourName={you.display_name}
        oppName={opp.display_name}
      />

      {/* Main Board */}
      <div className="main-board">
        {isTestLab && <TestLabToolbar />}

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

        {targeting && (
          <div className="targeting-banner">
            Select {targeting.targetsNeeded} target(s)
            {targeting.targetType ? ` (${targeting.targetType.replace(/_/g, " ")})` : ""}
            {" "}&mdash; {targeting.selectedTargets.length}/{targeting.targetsNeeded} selected
            <button disabled={targeting.selectedTargets.length < targeting.targetsNeeded} onClick={handleConfirmTargets}>Confirm</button>
            <button onClick={cancelTargeting}>Cancel</button>
          </div>
        )}

        {isPlacingUnit && (
          <div className="targeting-banner">
            Choose destination: click a Battlefield you control or your Base
            <button onClick={() => { handlePlayUnitToBase(); }}>Send to Base</button>
            <button onClick={() => selectCard(null)}>Cancel</button>
          </div>
        )}

        {/* Opponent Strip */}
        <div className="player-strip">
          <div className="strip-identity">
            <CardThumb card={opp.legend} label="Legend" />
            <CardThumb card={opp.champion} label="Champion" />
            <DeckIcon count={opp.main_deck_count} />
            <DeckIcon count={opp.rune_deck_count} isRune />
          </div>

          <div className="strip-hand">
            <HandZone cards={opp.hand} isOwn={false} />
          </div>

          <FloatingResources pool={opp.rune_pool} />

          <div className="opp-runes-mini">
            {opp.rune_board.map((r) => (
              <div
                key={r.instance_id}
                className={`card-back card-back--rune opp-rune-card ${r.exhausted ? 'rune-chip--exhausted' : ''}`}
                title={r.name ?? "Rune"}
              />
            ))}
          </div>
        </div>

        {/* Opponent Base — always visible */}
        <div className="base-row base-row--opp">
          <BaseZone
            units={opp.base_units}
            gear={opp.base_gear}
            label={opp.display_name}
            onUnitClick={targeting
              ? (id) => { const c = opp.base_units.find(u => u.instance_id === id); if (c) handleTargetClick(c); }
              : undefined}
            highlightIds={targeting?.selectedTargets}
            targeting={targeting ? { targetType: targeting.targetType, yourPlayerId: you.player_id } : undefined}
          />
        </div>

        {/* Battlefields */}
        <div className="battlefield-area">
          {battlefields.map((bf) => (
            <BattlefieldZone
              key={bf.battlefield_id}
              battlefield={bf}
              yourPlayerId={you.player_id}
              onUnitClick={targeting
                ? (id) => { const c = bf.units.find(u => u.instance_id === id); if (c) handleTargetClick(c); }
                : (id) => selectCard(id)}
              onDropUnit={targeting ? undefined : handlePlayUnitToBattlefield}
              highlightIds={targeting?.selectedTargets}
              targeting={targeting ? { targetType: targeting.targetType, yourPlayerId: you.player_id } : undefined}
            />
          ))}
        </div>

        {/* Your Base — always visible */}
        <div className="base-row base-row--you">
          <BaseZone
            units={you.base_units}
            gear={you.base_gear}
            label={you.display_name}
            onUnitClick={targeting
              ? (id) => { const c = you.base_units.find(u => u.instance_id === id); if (c) handleTargetClick(c); }
              : (id) => selectCard(id)}
            onDropUnit={targeting ? undefined : handlePlayUnitToBase}
            highlightIds={targeting?.selectedTargets}
            targeting={targeting ? { targetType: targeting.targetType, yourPlayerId: you.player_id } : undefined}
            activateAbility={targeting ? undefined : handleActivateAbility}
          />
        </div>

        {/* Your Strip */}
        <div className="player-strip player-strip--you">
          <div className="strip-identity">
            <CardThumb card={you.legend} label="Legend" />
            <CardThumb card={you.champion} label="Champion" />
            <DeckIcon count={you.main_deck_count} />
          </div>

          <FloatingResources pool={you.rune_pool} />

          <div className="strip-hand">
            <HandZone
              cards={you.hand}
              isOwn
              onCardClick={targeting ? undefined : handlePlayCard}
              isCardDimmed={isHandCardDimmed}
            />
          </div>

          <div className="strip-runes">
            <RuneZone
              runes={you.rune_board}
              runePool={you.rune_pool}
              runeDeckCount={you.rune_deck_count}
              onExhaust={handleExhaustRune}
              onRecycle={handleRecycleRune}
            />
          </div>
        </div>

        {/* Action Bar */}
        <div className="action-bar">
          {!gs.chain.is_empty && (
            <div className={`chain-indicator ${hasPriority ? 'chain-indicator--active' : ''}`}>
              Chain active ({gs.chain.items.length})
              {hasPriority ? ' — You have priority' : ` — Waiting for ${opp.display_name}`}
            </div>
          )}

          {isYourTurn && gs.chain.is_empty && !gs.showdown && (
            <button onClick={handleAdvancePhase}>End Phase</button>
          )}

          {hasPriority && !gs.chain.is_empty && (
            <button className="btn-priority" onClick={handlePassPriority}>Pass Priority</button>
          )}

          {gs.showdown && gs.showdown.focus_player_id === you.player_id && (
            <button onClick={handlePassFocus}>Pass Focus</button>
          )}

          {selectedCardId && (
            <button onClick={() => selectCard(null)}>Deselect</button>
          )}
          <button className="btn-danger" onClick={handleConcede}>Concede</button>
        </div>

        {/* Choice Modal */}
        {pendingChoice && (
          <ChoiceModal
            prompt={pendingChoice.prompt}
            options={pendingChoice.options}
            minChoices={pendingChoice.minChoices}
            maxChoices={pendingChoice.maxChoices}
            send={send}
            onDone={clearPendingChoice}
          />
        )}

        {/* Accelerate Modal */}
        {pendingAccelerate && (
          <AccelerateModal
            cardName={pendingAccelerate.cardName}
            domain={pendingAccelerate.domain}
            onConfirm={handleAccelerateChoice}
          />
        )}

        {/* Damage Assignment Modal */}
        {showDamageModal && gs.combat && (() => {
          const combat = gs.combat!;
          const bf = gs.battlefields[combat.battlefield_id];
          const isAttacker = combat.attacker_id === you.player_id;
          // Targets: opponent's units at the active combat battlefield
          const targets = bf
            ? bf.units.filter((u) => {
                const isYourUnit = u.controller_id === you.player_id || u.owner_id === you.player_id;
                return isAttacker ? !isYourUnit : isYourUnit;
              })
            : [];
          // Your total damage: sum of effective_might of your units at this battlefield
          const yourUnits = bf
            ? bf.units.filter((u) => {
                const isYourUnit = u.controller_id === you.player_id || u.owner_id === you.player_id;
                return isAttacker ? isYourUnit : !isYourUnit;
              })
            : [];
          const totalDamage = yourUnits.reduce((sum, u) => sum + (u.effective_might ?? 0), 0);
          return (
            <DamageAssignmentModal
              targets={targets}
              totalDamage={totalDamage}
              send={send}
            />
          );
        })()}
      </div>

      {/* Sidebar */}
      <Sidebar send={send} />
    </div>
  );
}
