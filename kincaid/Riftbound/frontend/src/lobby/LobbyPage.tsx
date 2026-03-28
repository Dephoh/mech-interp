import { useState } from "react";

interface LobbyPageProps {
  onJoin: (roomId: string, playerName: string) => void;
}

export function LobbyPage({ onJoin }: LobbyPageProps) {
  const [roomId, setRoomId] = useState("");
  const [playerName, setPlayerName] = useState("");
  const [creating, setCreating] = useState(false);

  // Derive API base from current page host so it works on any machine.
  const API_BASE = import.meta.env.VITE_API_URL ?? `${window.location.protocol}//${window.location.host}`;

  async function handleCreate() {
    setCreating(true);
    try {
      const res = await fetch(`${API_BASE}/rooms`, { method: "POST" });
      const data = await res.json();
      setRoomId(data.room_id);
    } finally {
      setCreating(false);
    }
  }

  function handleJoin() {
    if (!roomId.trim() || !playerName.trim()) return;
    onJoin(roomId.trim(), playerName.trim());
  }

  return (
    <div className="lobby">
      <h1>Riftbound Simulator</h1>

      <div className="lobby-form">
        <div>
          <label>Player Name</label>
          <input
            value={playerName}
            onChange={(e) => setPlayerName(e.target.value)}
            placeholder="Enter your name"
          />
        </div>

        <div>
          <label>Room ID</label>
          <div className="room-row">
            <input
              value={roomId}
              onChange={(e) => setRoomId(e.target.value)}
              placeholder="Enter room ID or create one"
            />
            <button onClick={handleCreate} disabled={creating}>
              {creating ? "Creating..." : "New Room"}
            </button>
          </div>
        </div>

        {roomId && (
          <div className="room-link">
            Share this Room ID with your opponent: <strong>{roomId}</strong>
          </div>
        )}

        <button className="join-btn" onClick={handleJoin} disabled={!roomId || !playerName}>
          Join Game
        </button>
      </div>
    </div>
  );
}
