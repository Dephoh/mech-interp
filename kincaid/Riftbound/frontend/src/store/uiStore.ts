import { create } from "zustand";

interface UIStore {
  selectedCardId: string | null;
  dragSourceId: string | null;
  showMulliganModal: boolean;
  showDamageModal: boolean;
  contextMenu: { x: number; y: number; cardId: string } | null;

  selectCard: (id: string | null) => void;
  setDragSource: (id: string | null) => void;
  setMulliganModal: (show: boolean) => void;
  setDamageModal: (show: boolean) => void;
  openContextMenu: (x: number, y: number, cardId: string) => void;
  closeContextMenu: () => void;
}

export const useUIStore = create<UIStore>((set) => ({
  selectedCardId: null,
  dragSourceId: null,
  showMulliganModal: false,
  showDamageModal: false,
  contextMenu: null,

  selectCard: (id) => set({ selectedCardId: id }),
  setDragSource: (id) => set({ dragSourceId: id }),
  setMulliganModal: (show) => set({ showMulliganModal: show }),
  setDamageModal: (show) => set({ showDamageModal: show }),
  openContextMenu: (x, y, cardId) => set({ contextMenu: { x, y, cardId } }),
  closeContextMenu: () => set({ contextMenu: null }),
}));
