import { useCallback, useEffect, useState } from "react";
import "./App.css";
import { CardHoverPortal } from "./components/cards/CardHoverPortal";
import { GameBoard } from "./components/layout/GameBoard";
import { LobbyPage } from "./lobby/LobbyPage";
import { useGameStore } from "./store/gameStore";
import { useTestLabStore } from "./store/testlabStore";
import { useUIStore } from "./store/uiStore";
import type { ClientMessage, ServerMessage } from "./ws/messageTypes";
import { useWebSocket } from "./ws/useWebSocket";

// Starter deck using real card IDs from card_definitions.json
// Mix of domains so rune power is useful. All units/spells have effect_ir.
function makeStarterDeck() {
  return {
    legend_id: "unl-230-star-219",       // Bashful Bloom
    champion_id: "ogn-097-298",          // Blastcone Fae (2E 2M, cheap unit as placeholder)
    main_deck: [
      // Units (24) — spread across domains
      ...Array(3).fill("ogn-097-298"),   // Blastcone Fae        2E 2M mind
      ...Array(3).fill("sfd-091-221"),   // Buhru Captain         3E 3M body
      ...Array(3).fill("sfd-061-221"),   // Aspiring Engineer     3E 3M mind
      ...Array(3).fill("ogn-056-298"),   // Adaptatron            4E 3M calm
      ...Array(3).fill("ogs-010-024"),   // Annie, Stubborn       4E 3M chaos
      ...Array(3).fill("unl-121-219"),   // Bewitching Spirit     3E 2M chaos
      ...Array(3).fill("sfd-177a-221"),  // Azir, Sovereign       4E 4M order
      ...Array(3).fill("unl-132-219"),   // Angler Beast          5E 5M chaos
      // Spells (12) — all have effect_ir
      ...Array(3).fill("sfd-011-221"),   // Angle Shot            2E fury  (deal damage)
      ...Array(3).fill("ogn-127-298"),   // Cannon Barrage        2E body  (deal damage)
      ...Array(3).fill("sfd-001-221"),   // Against the Odds      2E fury  (give might)
      ...Array(3).fill("sfd-162-221"),   // Blood Money           2E order
      // Gear (4)
      ...Array(2).fill("sfd-169-221"),   // Altar of Memories     2E
      ...Array(2).fill("ogn-124-298"),   // Arena Bar             3E
    ],
    rune_deck: [
      // 2 of each domain = 12 runes
      ...Array(2).fill("ogn-007a-298"),  // Fury Rune
      ...Array(2).fill("ogn-042a-298"),  // Calm Rune
      ...Array(2).fill("ogn-089a-298"),  // Mind Rune
      ...Array(2).fill("ogn-126a-298"),  // Body Rune
      ...Array(2).fill("ogn-166a-298"),  // Chaos Rune
      ...Array(2).fill("ogn-214a-298"),  // Order Rune
    ],
    battlefields: [
      "unl-205-219",   // Abandoned Hall
      "unl-206-219",   // Altar of Blood
      "ogn-275-298",   // Altar to Unity
    ],
  };
}

const API_BASE =
  import.meta.env.VITE_API_URL ??
  `${window.location.protocol}//${window.location.host}`;

function App() {
  const [inGame, setInGame] = useState(false);
  const roomId = useGameStore((s) => s.roomId);
  const waiting = useGameStore((s) => s.waiting);  // reactive subscription
  const setRoom = useGameStore((s) => s.setRoom);
  const setPlayer = useGameStore((s) => s.setPlayer);
  const setWaiting = useGameStore((s) => s.setWaiting);
  const setReconnectToken = useGameStore((s) => s.setReconnectToken);
  const setOpponentDisconnected = useGameStore((s) => s.setOpponentDisconnected);
  const updateState = useGameStore((s) => s.updateState);
  const addLogs = useGameStore((s) => s.addLogs);
  const setGameOver = useGameStore((s) => s.setGameOver);
  const setError = useGameStore((s) => s.setError);

  const setTestLab = useTestLabStore((s) => s.setTestLab);
  const setScenarios = useTestLabStore((s) => s.setScenarios);
  const setCurrentIndex = useTestLabStore((s) => s.setCurrentIndex);
  const setCurrentScenario = useTestLabStore((s) => s.setCurrentScenario);
  const setTestLabRoomId = useTestLabStore((s) => s.setRoomId);
  const setTestLabLoading = useTestLabStore((s) => s.setLoading);

  const handleMessage = useCallback((msg: ServerMessage) => {
    switch (msg.type) {
      case "ROOM_JOINED":
        // player_id comes from GAME_STARTED; here just clear waiting on slot 0
        setPlayer(msg.room_id, msg.player_slot);
        // Store reconnect token for use during reconnection
        if (msg.reconnect_token) {
          setReconnectToken(msg.reconnect_token);
        }
        break;
      case "RECONNECT_SUCCESS":
        // Restore player identity after reconnection
        setPlayer(msg.your_player_id, msg.player_slot);
        setOpponentDisconnected(false);
        break;
      case "PLAYER_DISCONNECTED":
        setOpponentDisconnected(true);
        addLogs(["Opponent disconnected. Waiting for reconnection..."]);
        break;
      case "PLAYER_RECONNECTED":
        setOpponentDisconnected(false);
        addLogs(["Opponent reconnected."]);
        break;
      case "GAME_STARTED":
        setPlayer(msg.your_player_id, useGameStore.getState().playerSlot ?? 0);
        break;
      case "STATE_UPDATE":
        updateState(msg.state);
        break;
      case "GAME_LOG":
        addLogs(msg.entries.map((e) => e.message));
        break;
      case "GAME_OVER":
        setGameOver(msg.winner_id, msg.reason);
        break;
      case "ACTION_REJECTED":
        setError(msg.reason);
        break;
      case "WAITING_FOR_OPPONENT":
        setWaiting(true);
        break;
      case "ERROR":
        setError(msg.message);
        break;
      case "CHOICE_REQUIRED":
        useUIStore.getState().setPendingChoice({
          prompt: msg.prompt,
          options: msg.options ?? [],
          target: msg.target ?? null,
          minChoices: msg.min_choices ?? 1,
          maxChoices: msg.max_choices ?? 1,
          controllerId: msg.controller_id,
        });
        break;
      case "TESTLAB_RESET":
      case "TESTLAB_SCENARIO_CHANGED":
        break;
    }
  }, [setPlayer, setReconnectToken, setOpponentDisconnected, updateState, addLogs, setGameOver, setError, setWaiting]);

  const { status, connect, send, reconnectAttempt, wasConnected, justReconnected } = useWebSocket(handleMessage);

  const handleJoin = useCallback(
    (rid: string, playerName: string) => {
      setRoom(rid);
      setInGame(true);
      // Pass rid directly — avoids stale closure on roomId state
      const joinMsg: ClientMessage = {
        type: "JOIN_ROOM" as const,
        player_name: playerName,
        deck: makeStarterDeck(),
      };
      connect(rid, joinMsg);
    },
    [setRoom, connect]
  );

  const handleSandbox = useCallback(
    (rid: string) => {
      setRoom(rid);
      setInGame(true);
      // Sandbox: no deck needed, game already created on server
      const joinMsg: ClientMessage = {
        type: "JOIN_ROOM" as const,
        player_name: "Sandbox Player",
      };
      connect(rid, joinMsg);
    },
    [setRoom, connect]
  );

  const handleTestLab = useCallback(async () => {
    setTestLabLoading(true);
    try {
      // Retry fetching scenarios — server may still be booting
      let scenarios: any[] = [];
      for (let attempt = 0; attempt < 15; attempt++) {
        try {
          const res = await fetch(`${API_BASE}/testlab/scenarios`);
          if (res.ok) {
            scenarios = await res.json();
            break;
          }
        } catch {
          // Server not ready yet
        }
        await new Promise((r) => setTimeout(r, 1000));
      }
      if (!scenarios.length) {
        setError("Failed to load test lab scenarios. Is the server running?");
        return;
      }

      setScenarios(scenarios);
      setTestLab(true);

      const first = scenarios[0];
      const loadRes = await fetch(`${API_BASE}/testlab/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario_id: first.scenario_id }),
      });
      const loadData = await loadRes.json();

      setTestLabRoomId(loadData.room_id);
      setCurrentIndex(0);
      setCurrentScenario(first);
      setRoom(loadData.room_id);
      setInGame(true);

      const joinMsg: ClientMessage = {
        type: "JOIN_ROOM" as const,
        player_name: "Test Lab",
      };
      connect(loadData.room_id, joinMsg);
    } finally {
      setTestLabLoading(false);
    }
  }, [setRoom, connect, setTestLab, setScenarios, setCurrentIndex, setCurrentScenario, setTestLabRoomId, setTestLabLoading, setError]);

  // Auto-detect ?mode=testlab in URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("mode") === "testlab") {
      handleTestLab();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const testLabLoading = useTestLabStore((s) => s.loading);

  if (!inGame && testLabLoading) {
    return (
      <div className="waiting-screen">
        <h2>Card Test Lab</h2>
        <p>Loading scenarios...</p>
      </div>
    );
  }

  if (!inGame) {
    return <LobbyPage onJoin={handleJoin} onSandbox={handleSandbox} onTestLab={handleTestLab} />;
  }

  if (waiting || status === "connecting") {
    return (
      <div className="waiting-screen">
        <h2>Room: {roomId}</h2>
        <p>Waiting for opponent to join...</p>
        <p className="status">Connection: {status}</p>
      </div>
    );
  }

  return (
    <>
      <GameBoard
        send={send}
        connectionStatus={status}
        reconnectAttempt={reconnectAttempt}
        wasConnected={wasConnected}
        justReconnected={justReconnected}
      />
      <CardHoverPortal />
    </>
  );
}

export default App;
