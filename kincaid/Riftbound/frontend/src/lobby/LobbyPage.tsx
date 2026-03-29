import { useState } from "react";

interface LobbyPageProps {
  onJoin: (roomId: string, playerName: string) => void;
  onSandbox: (roomId: string) => void;
  onTestLab: () => void;
}

export function LobbyPage({ onJoin, onSandbox, onTestLab }: LobbyPageProps) {
  const [roomId, setRoomId] = useState("");
  const [playerName, setPlayerName] = useState("");
  const [creating, setCreating] = useState(false);
  const [creatingSandbox, setCreatingSandbox] = useState(false);

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

  async function handleSandbox() {
    setCreatingSandbox(true);
    try {
      const res = await fetch(`${API_BASE}/sandbox`, { method: "POST" });
      const data = await res.json();
      onSandbox(data.room_id);
    } finally {
      setCreatingSandbox(false);
    }
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

        <hr style={{ margin: "1rem 0", borderColor: "#444" }} />

        <button
          className="join-btn sandbox-btn"
          onClick={handleSandbox}
          disabled={creatingSandbox}
        >
          {creatingSandbox ? "Creating..." : "Sandbox (Test Mode)"}
        </button>
        <p style={{ fontSize: "0.8rem", color: "#888", marginTop: "0.25rem" }}>
          Pre-built game with boosted resources. Open two tabs to play both sides.
        </p>

        <button
          className="join-btn testlab-btn"
          onClick={onTestLab}
        >
          Card Test Lab
        </button>
        <p style={{ fontSize: "0.8rem", color: "#888", marginTop: "0.25rem" }}>
          Test every card with pre-built scenarios, infinite resources, and reset controls.
        </p>
      </div>
    </div>
  );
}
