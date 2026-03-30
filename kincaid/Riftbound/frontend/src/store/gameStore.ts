import { create } from "zustand";
import type { ClientGameState } from "../ws/messageTypes";

interface GameStore {
  // Connection state
  roomId: string | null;
  playerId: string | null;
  playerSlot: number | null;
  waiting: boolean;
  reconnectToken: string | null;
  opponentDisconnected: boolean;

  // Game state (from server)
  gameState: ClientGameState | null;
  gameLogs: string[];
  gameOver: boolean;
  winnerId: string | null;
  lastError: string | null;

  // Actions
  setRoom: (roomId: string) => void;
  setPlayer: (playerId: string, slot: number) => void;
  setWaiting: (waiting: boolean) => void;
  setReconnectToken: (token: string) => void;
  setOpponentDisconnected: (disconnected: boolean) => void;
  updateState: (state: ClientGameState) => void;
  addLogs: (logs: string[]) => void;
  setGameOver: (winnerId: string, reason: string) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useGameStore = create<GameStore>((set) => ({
  roomId: null,
  playerId: null,
  playerSlot: null,
  waiting: false,
  reconnectToken: null,
  opponentDisconnected: false,
  gameState: null,
  gameLogs: [],
  gameOver: false,
  winnerId: null,
  lastError: null,

  setRoom: (roomId) => set({ roomId }),
  setPlayer: (playerId, slot) => set({ playerId, playerSlot: slot, waiting: false }),
  setWaiting: (waiting) => set({ waiting }),
  setReconnectToken: (token) => set({ reconnectToken: token }),
  setOpponentDisconnected: (disconnected) => set({ opponentDisconnected: disconnected }),
  updateState: (state) => set({ gameState: state }),
  addLogs: (logs) =>
    set((s) => ({ gameLogs: [...s.gameLogs.slice(-200), ...logs] })),
  setGameOver: (winnerId, _reason) => set({ gameOver: true, winnerId }),
  setError: (error) => set({ lastError: error }),
  reset: () =>
    set({
      roomId: null,
      playerId: null,
      playerSlot: null,
      waiting: false,
      reconnectToken: null,
      opponentDisconnected: false,
      gameState: null,
      gameLogs: [],
      gameOver: false,
      winnerId: null,
      lastError: null,
    }),
}));
