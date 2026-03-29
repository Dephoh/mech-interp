import { useTestLabStore } from "../../store/testlabStore";
import { useGameStore } from "../../store/gameStore";
import { ScenarioBrowser } from "./ScenarioBrowser";

const API_BASE =
  import.meta.env.VITE_API_URL ??
  `${window.location.protocol}//${window.location.host}`;

export function TestLabToolbar() {
  const scenarios = useTestLabStore((s) => s.scenarios);
  const currentIndex = useTestLabStore((s) => s.currentIndex);
  const currentScenario = useTestLabStore((s) => s.currentScenario);
  const roomId = useTestLabStore((s) => s.roomId);
  const loading = useTestLabStore((s) => s.loading);
  const showBrowser = useTestLabStore((s) => s.showBrowser);
  const setCurrentIndex = useTestLabStore((s) => s.setCurrentIndex);
  const setCurrentScenario = useTestLabStore((s) => s.setCurrentScenario);
  const setRoomId = useTestLabStore((s) => s.setRoomId);
  const setLoading = useTestLabStore((s) => s.setLoading);
  const setShowBrowser = useTestLabStore((s) => s.setShowBrowser);
  const updateState = useGameStore((s) => s.updateState);

  async function switchScenario(index: number) {
    if (index < 0 || index >= scenarios.length || !roomId) return;
    const scenario = scenarios[index];
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/testlab/switch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room_id: roomId, scenario_id: scenario.scenario_id }),
      });
      const data = await res.json();
      if (data.room_id) setRoomId(data.room_id);
      if (data.state) updateState(data.state);
      setCurrentIndex(index);
      setCurrentScenario(scenario);
    } finally {
      setLoading(false);
    }
  }

  async function handleReset() {
    if (!roomId || !currentScenario) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/testlab/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room_id: roomId, scenario_id: currentScenario.scenario_id }),
      });
      const data = await res.json();
      if (data.room_id) setRoomId(data.room_id);
      if (data.state) updateState(data.state);
    } finally {
      setLoading(false);
    }
  }

  function handleBrowseSelect(index: number) {
    setShowBrowser(false);
    switchScenario(index);
  }

  return (
    <>
      <div className="testlab-toolbar">
        <div className="testlab-toolbar__row">
          <button
            className="testlab-toolbar__nav-btn"
            disabled={loading || currentIndex <= 0}
            onClick={() => switchScenario(currentIndex - 1)}
          >
            &laquo; Prev
          </button>

          <span className="testlab-toolbar__title">
            Scenario {currentIndex + 1}/{scenarios.length}:{" "}
            &ldquo;{currentScenario?.name ?? "..."}&rdquo;
          </span>

          <button
            className="testlab-toolbar__nav-btn"
            disabled={loading || currentIndex >= scenarios.length - 1}
            onClick={() => switchScenario(currentIndex + 1)}
          >
            Next &raquo;
          </button>
        </div>

        <div className="testlab-toolbar__row">
          <button
            className="testlab-toolbar__action-btn"
            onClick={() => setShowBrowser(true)}
            disabled={loading}
          >
            Browse
          </button>
          <button
            className="testlab-toolbar__action-btn"
            onClick={handleReset}
            disabled={loading}
          >
            Reset
          </button>
          <button
            className="testlab-toolbar__action-btn"
            onClick={() => { window.location.href = "/"; }}
          >
            Back to Lobby
          </button>
        </div>

        {currentScenario?.expected_behavior && (
          <div className="testlab-toolbar__expected">
            Try: {currentScenario.expected_behavior}
          </div>
        )}
      </div>

      {showBrowser && (
        <ScenarioBrowser
          onClose={() => setShowBrowser(false)}
          onSelect={handleBrowseSelect}
        />
      )}
    </>
  );
}
