import { useCallback, useEffect, useRef, useState } from "react";
import { useGameStore } from "../store/gameStore";
import type { ClientMessage, ServerMessage } from "./messageTypes";

// Auto-derive WebSocket URL from the current page host so it works when
// served from any machine (LAN, ngrok, cloud) without any env config.
// Falls back to env var if set, or localhost for dev.
function getWsBase(): string {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}`;
  }
  return "ws://localhost:8000";
}
const WS_BASE = getWsBase();

/** Reconnect configuration */
const RECONNECT_BASE_DELAY = 1000;   // 1s initial delay
const RECONNECT_MAX_DELAY = 30000;   // 30s cap
const RECONNECT_MAX_ATTEMPTS = 10;

export type ConnectionStatus = "disconnected" | "connecting" | "connected";

export function useWebSocket(onMessage?: (msg: ServerMessage) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const [wasConnected, setWasConnected] = useState(false);
  const [justReconnected, setJustReconnected] = useState(false);

  // Store the last connection params for reconnection
  const lastRoomIdRef = useRef<string | null>(null);
  const lastJoinMsgRef = useRef<ClientMessage | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intentionalCloseRef = useRef(false);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  /** Clear any pending reconnect timer */
  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  /** Core connect logic (used for both initial and reconnect). */
  const connectInternal = useCallback(
    (rid: string, joinMsg: ClientMessage, isReconnect: boolean) => {
      if (!rid) return;

      // Clean up any existing socket
      if (wsRef.current) {
        intentionalCloseRef.current = true;
        wsRef.current.close();
        wsRef.current = null;
      }

      setStatus("connecting");
      intentionalCloseRef.current = false;

      const ws = new WebSocket(`${WS_BASE}/ws/${rid}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("connected");
        setReconnectAttempt(0);

        if (isReconnect) {
          // On reconnect, send RECONNECT with stored token if available
          const token = useGameStore.getState().reconnectToken;
          if (token) {
            ws.send(JSON.stringify({ type: "RECONNECT", reconnect_token: token }));
          } else {
            // Fall back to re-sending original join message
            ws.send(JSON.stringify(joinMsg));
          }
          setJustReconnected(true);
          // Clear the "Reconnected!" flash after 3 seconds
          setTimeout(() => setJustReconnected(false), 3000);
        } else {
          ws.send(JSON.stringify(joinMsg));
          setWasConnected(true);
        }
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as ServerMessage;
          onMessageRef.current?.(msg);
        } catch {
          console.error("Failed to parse WS message:", event.data);
        }
      };

      ws.onclose = () => {
        wsRef.current = null;

        if (intentionalCloseRef.current) {
          // Intentional disconnect (user called disconnect or component unmounted)
          setStatus("disconnected");
          setReconnectAttempt(0);
          setWasConnected(false);
          return;
        }

        // Unexpected close -- attempt reconnect
        setStatus("disconnected");
        setWasConnected(true);

        setReconnectAttempt((prev) => {
          const next = prev + 1;
          if (next > RECONNECT_MAX_ATTEMPTS) {
            console.warn(`WebSocket: gave up reconnecting after ${RECONNECT_MAX_ATTEMPTS} attempts`);
            return next;
          }

          const delay = Math.min(
            RECONNECT_BASE_DELAY * Math.pow(2, prev),
            RECONNECT_MAX_DELAY,
          );
          console.log(`WebSocket: reconnecting in ${delay}ms (attempt ${next}/${RECONNECT_MAX_ATTEMPTS})`);

          clearReconnectTimer();
          reconnectTimerRef.current = setTimeout(() => {
            const storedRid = lastRoomIdRef.current;
            const storedMsg = lastJoinMsgRef.current;
            if (storedRid && storedMsg) {
              connectInternal(storedRid, storedMsg, true);
            }
          }, delay);

          return next;
        });
      };

      ws.onerror = (err) => {
        console.error("WS error:", err);
        // Let onclose handle reconnection; just close the socket
        ws.close();
      };
    },
    [clearReconnectTimer],
  );

  // Accept roomId explicitly so callers don't depend on stale closures
  const connect = useCallback(
    (rid: string, joinMsg: ClientMessage) => {
      // Store params for potential reconnection
      lastRoomIdRef.current = rid;
      lastJoinMsgRef.current = joinMsg;
      setReconnectAttempt(0);
      setWasConnected(false);
      setJustReconnected(false);
      clearReconnectTimer();
      connectInternal(rid, joinMsg, false);
    },
    [connectInternal, clearReconnectTimer],
  );

  const send = useCallback((msg: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const disconnect = useCallback(() => {
    intentionalCloseRef.current = true;
    clearReconnectTimer();
    lastRoomIdRef.current = null;
    lastJoinMsgRef.current = null;
    wsRef.current?.close();
  }, [clearReconnectTimer]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      intentionalCloseRef.current = true;
      clearReconnectTimer();
      wsRef.current?.close();
    };
  }, [clearReconnectTimer]);

  return {
    status,
    connect,
    send,
    disconnect,
    /** Number of reconnect attempts so far (0 = connected or never disconnected). */
    reconnectAttempt,
    /** True if the socket was previously connected and then lost. Drives "Connection lost" UI. */
    wasConnected,
    /** True briefly after a successful reconnection. Drives "Reconnected!" UI. */
    justReconnected,
  };
}
