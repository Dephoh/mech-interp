import { useCallback, useEffect, useRef, useState } from "react";
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

export type ConnectionStatus = "disconnected" | "connecting" | "connected";

export function useWebSocket(onMessage?: (msg: ServerMessage) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");

  // Accept roomId explicitly so callers don't depend on stale closures
  const connect = useCallback(
    (rid: string, joinMsg: ClientMessage) => {
      if (!rid) return;
      setStatus("connecting");

      const ws = new WebSocket(`${WS_BASE}/ws/${rid}`);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("connected");
        ws.send(JSON.stringify(joinMsg));
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as ServerMessage;
          onMessage?.(msg);
        } catch {
          console.error("Failed to parse WS message:", event.data);
        }
      };

      ws.onclose = () => {
        setStatus("disconnected");
        wsRef.current = null;
      };

      ws.onerror = (err) => {
        console.error("WS error:", err);
        ws.close();
      };
    },
    []
  );

  const send = useCallback((msg: ClientMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return { status, connect, send, disconnect };
}
