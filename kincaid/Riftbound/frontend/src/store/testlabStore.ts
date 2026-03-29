import { create } from "zustand";

export interface ScenarioMeta {
  scenario_id: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  expected_behavior: string;
}

interface TestLabStore {
  isTestLab: boolean;
  scenarios: ScenarioMeta[];
  currentIndex: number;
  currentScenario: ScenarioMeta | null;
  roomId: string | null;
  loading: boolean;
  showBrowser: boolean;

  setTestLab: (val: boolean) => void;
  setScenarios: (s: ScenarioMeta[]) => void;
  setCurrentIndex: (i: number) => void;
  setCurrentScenario: (s: ScenarioMeta | null) => void;
  setRoomId: (id: string | null) => void;
  setLoading: (l: boolean) => void;
  setShowBrowser: (show: boolean) => void;
}

export const useTestLabStore = create<TestLabStore>((set) => ({
  isTestLab: false,
  scenarios: [],
  currentIndex: 0,
  currentScenario: null,
  roomId: null,
  loading: false,
  showBrowser: false,

  setTestLab: (val) => set({ isTestLab: val }),
  setScenarios: (s) => set({ scenarios: s }),
  setCurrentIndex: (i) => set({ currentIndex: i }),
  setCurrentScenario: (s) => set({ currentScenario: s }),
  setRoomId: (id) => set({ roomId: id }),
  setLoading: (l) => set({ loading: l }),
  setShowBrowser: (show) => set({ showBrowser: show }),
}));
