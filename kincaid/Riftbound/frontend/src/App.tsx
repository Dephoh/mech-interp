import { useCallback, useState } from "react";
import "./App.css";
import { GameBoard } from "./components/layout/GameBoard";
import { LobbyPage } from "./lobby/LobbyPage";
import { useGameStore } from "./store/gameStore";
import type { ClientMessage, ServerMessage } from "./ws/messageTypes";
import { useWebSocket } from "./ws/useWebSocket";

// Starter deck using actual card IDs from the database
function makeStarterDeck() {
  return {
    legend_id: "legend_jinx",
    champion_id: "unit_jinx_rebel",
    main_deck: [
      ...Array(4).fill("unit_hextech_striker"),
      ...Array(4).fill("unit_demacian_guard"),
      ...Array(4).fill("unit_mystic_sage"),
      ...Array(4).fill("unit_noxian_recruit"),
      ...Array(4).fill("unit_yordle_scout"),
      ...Array(4).fill("unit_stalwart_poro"),
      ...Array(3).fill("spell_arcane_bolt"),
      ...Array(3).fill("spell_quick_strike"),
      ...Array(3).fill("spell_healing_light"),
      ...Array(3).fill("spell_power_surge"),
      ...Array(2).fill("gear_hextech_core"),
      ...Array(2).fill("gear_scouts_ward"),
    ],
    rune_deck: [
      ...Array(2).fill("rune_fury"),
      ...Array(2).fill("rune_calm"),
      ...Array(2).fill("rune_mind"),
      ...Array(2).fill("rune_body"),
      ...Array(2).fill("rune_chaos"),
      ...Array(2).fill("rune_order"),
    ],
    battlefields: ["bf_summoner_rift", "bf_howling_abyss", "bf_piltover"],
  };
}

function App() {
  const [inGame, setInGame] = useState(false);
  const roomId = useGameStore((s) => s.roomId);
  const waiting = useGameStore((s) => s.waiting);  // reactive subscription
  const setRoom = useGameStore((s) => s.setRoom);
  const setPlayer = useGameStore((s) => s.setPlayer);
  const setWaiting = useGameStore((s) => s.setWaiting);
  const updateState = useGameStore((s) => s.updateState);
  const addLogs = useGameStore((s) => s.addLogs);
  const setGameOver = useGameStore((s) => s.setGameOver);
  const setError = useGameStore((s) => s.setError);

  const handleMessage = useCallback((msg: ServerMessage) => {
    switch (msg.type) {
      case "ROOM_JOINED":
        // player_id comes from GAME_STARTED; here just clear waiting on slot 0
        setPlayer(msg.room_id, msg.player_slot);
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
    }
  }, [setPlayer, updateState, addLogs, setGameOver, setError, setWaiting]);

  const { status, connect, send } = useWebSocket(handleMessage);

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

  if (!inGame) {
    return <LobbyPage onJoin={handleJoin} />;
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

  return <GameBoard send={send} />;
}

export default App;
