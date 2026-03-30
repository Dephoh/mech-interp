import { create } from "zustand";
import type { CardView } from "../ws/messageTypes";

interface TargetingState {
  cardId: string;
  targetsNeeded: number;
  targetType: string;
  selectedTargets: string[];
  actionType: "play_card" | "activate_ability";
  sourceId?: string;
  abilityId?: string;
}

interface PendingChoice {
  prompt: string;
  options: { label: string; effect_ir?: unknown }[];
  target: unknown | null;
  minChoices: number;
  maxChoices: number;
  controllerId: string;
}

interface PendingAccelerate {
  cardId: string;
  cardName: string;
  domain: string;
  destination?: { zone: string; id?: string };
}

interface HoverPortal {
  card: CardView;
  rect: DOMRect;
  zone: "hand" | "battlefield" | "base" | "other";
}

interface UIStore {
  selectedCardId: string | null;
  dragSourceId: string | null;
  showMulliganModal: boolean;
  showDamageModal: boolean;
  targeting: TargetingState | null;
  pendingChoice: PendingChoice | null;
  previewCard: CardView | null;
  hoverPortal: HoverPortal | null;
  pendingAccelerate: PendingAccelerate | null;

  selectCard: (id: string | null) => void;
  setDragSource: (id: string | null) => void;
  setMulliganModal: (show: boolean) => void;
  setDamageModal: (show: boolean) => void;
  enterTargeting: (cardId: string, targetsNeeded: number, targetType: string) => void;
  enterAbilityTargeting: (sourceId: string, abilityId: string, targetsNeeded: number, targetType: string) => void;
  addTarget: (instanceId: string) => void;
  cancelTargeting: () => void;
  setPendingChoice: (choice: PendingChoice) => void;
  clearPendingChoice: () => void;
  setPreviewCard: (card: CardView | null) => void;
  setHoverPortal: (card: CardView, rect: DOMRect, zone: HoverPortal["zone"]) => void;
  clearHoverPortal: () => void;
  setPendingAccelerate: (data: PendingAccelerate) => void;
  clearPendingAccelerate: () => void;
}

export const useUIStore = create<UIStore>((set) => ({
  selectedCardId: null,
  dragSourceId: null,
  showMulliganModal: false,
  showDamageModal: false,
  targeting: null,
  pendingChoice: null,
  previewCard: null,
  hoverPortal: null,
  pendingAccelerate: null,

  selectCard: (id) => set({ selectedCardId: id }),
  setDragSource: (id) => set({ dragSourceId: id }),
  setMulliganModal: (show) => set({ showMulliganModal: show }),
  setDamageModal: (show) => set({ showDamageModal: show }),
  enterTargeting: (cardId, targetsNeeded, targetType) =>
    set({ targeting: { cardId, targetsNeeded, targetType, selectedTargets: [], actionType: "play_card" }, selectedCardId: null }),
  enterAbilityTargeting: (sourceId, abilityId, targetsNeeded, targetType) =>
    set({ targeting: { cardId: sourceId, targetsNeeded, targetType, selectedTargets: [], actionType: "activate_ability", sourceId, abilityId }, selectedCardId: null }),
  addTarget: (instanceId) =>
    set((s) => {
      if (!s.targeting) return {};
      const already = s.targeting.selectedTargets.includes(instanceId);
      const next = already
        ? s.targeting.selectedTargets.filter((id) => id !== instanceId)
        : [...s.targeting.selectedTargets, instanceId];
      return { targeting: { ...s.targeting, selectedTargets: next } };
    }),
  cancelTargeting: () => set({ targeting: null }),
  setPendingChoice: (choice) => set({ pendingChoice: choice }),
  clearPendingChoice: () => set({ pendingChoice: null }),
  setPreviewCard: (card) => set({ previewCard: card }),
  setHoverPortal: (card, rect, zone) => set({ hoverPortal: { card, rect, zone }, previewCard: card }),
  clearHoverPortal: () => set({ hoverPortal: null, previewCard: null }),
  setPendingAccelerate: (data) => set({ pendingAccelerate: data }),
  clearPendingAccelerate: () => set({ pendingAccelerate: null }),
}));
