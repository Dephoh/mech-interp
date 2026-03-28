// TypeScript types mirroring the backend protocol

// ---------------------------------------------------------------------------
// Client → Server
// ---------------------------------------------------------------------------

export interface DeckPayload {
  legend_id: string;
  champion_id: string;
  main_deck: string[];
  rune_deck: string[];
  battlefields: string[];
}

export interface JoinRoomMsg {
  type: "JOIN_ROOM";
  player_name: string;
  deck: DeckPayload;
}

export interface MulliganChoiceMsg {
  type: "MULLIGAN_CHOICE";
  keep_indices: number[];
}

export interface AdvancePhaseMsg {
  type: "ADVANCE_PHASE";
}

export interface PassPriorityMsg {
  type: "PASS_PRIORITY";
}

export interface PassFocusMsg {
  type: "PASS_FOCUS";
}

export interface PlayCardMsg {
  type: "PLAY_CARD";
  instance_id: string;
  targets: string[];
  pay_accelerate?: boolean;
}

export interface ActivateAbilityMsg {
  type: "ACTIVATE_ABILITY";
  source_id: string;
  ability_id: string;
  targets: string[];
}

export interface MoveUnitMsg {
  type: "MOVE_UNIT";
  instance_id: string;
  destination: { zone: string; id?: string };
}

export interface ExhaustRuneMsg {
  type: "EXHAUST_RUNE";
  instance_id: string;
}

export interface RecycleRuneMsg {
  type: "RECYCLE_RUNE";
  instance_id: string;
}

export interface AssignDamageMsg {
  type: "ASSIGN_DAMAGE";
  assignments: Record<string, number>;
}

export interface ConcedeMsg {
  type: "CONCEDE";
}

export type ClientMessage =
  | JoinRoomMsg
  | MulliganChoiceMsg
  | AdvancePhaseMsg
  | PassPriorityMsg
  | PassFocusMsg
  | PlayCardMsg
  | ActivateAbilityMsg
  | MoveUnitMsg
  | ExhaustRuneMsg
  | RecycleRuneMsg
  | AssignDamageMsg
  | ConcedeMsg;

// ---------------------------------------------------------------------------
// Server → Client
// ---------------------------------------------------------------------------

export interface RoomJoinedMsg {
  type: "ROOM_JOINED";
  player_slot: number;
  room_id: string;
}

export interface GameStartedMsg {
  type: "GAME_STARTED";
  your_player_id: string;
}

export interface StateUpdateMsg {
  type: "STATE_UPDATE";
  state: ClientGameState;
}

export interface ActionRejectedMsg {
  type: "ACTION_REJECTED";
  reason: string;
}

export interface GameLogMsg {
  type: "GAME_LOG";
  entries: { message: string }[];
}

export interface GameOverMsg {
  type: "GAME_OVER";
  winner_id: string;
  reason: string;
}

export interface WaitingForOpponentMsg {
  type: "WAITING_FOR_OPPONENT";
}

export interface ErrorMsg {
  type: "ERROR";
  message: string;
}

export type ServerMessage =
  | RoomJoinedMsg
  | GameStartedMsg
  | StateUpdateMsg
  | ActionRejectedMsg
  | GameLogMsg
  | GameOverMsg
  | WaitingForOpponentMsg
  | ErrorMsg;

// ---------------------------------------------------------------------------
// Game State (client view)
// ---------------------------------------------------------------------------

export interface ClientGameState {
  game_id: string;
  phase: string;
  turn_number: number;
  turn_player_id: string;
  active_player_id: string;
  has_priority: boolean;
  is_your_turn: boolean;
  can_play_cards: boolean;
  game_over: boolean;
  winner_id: string | null;
  you: PlayerView;
  opponent: PlayerView;
  battlefields: Record<string, BattlefieldView>;
  chain: ChainView;
  combat: CombatView | null;
  showdown: ShowdownView | null;
  turn_state: string;
  log: string[];
}

export interface PlayerView {
  player_id: string;
  display_name: string;
  score: number;
  main_deck_count: number;
  rune_deck_count: number;
  trash_count: number;
  hand_count: number;
  hand: CardView[];
  rune_pool: { energy: number; power: Record<string, number> };
  legend: CardView | null;
  champion: CardView | null;
  base_units: CardView[];
  base_gear: CardView[];
  rune_board: CardView[];
  trash: CardView[];
}

export interface CardView {
  instance_id: string;
  card_id?: string;
  name?: string;
  card_type?: string;
  owner_id?: string;
  controller_id?: string;
  zone?: string;
  domains?: string[];
  cost_energy?: number;
  cost_power?: Record<string, number>;
  text?: string;
  base_might?: number;
  effective_might?: number;
  combat_might?: number;
  damage?: number;
  exhausted?: boolean;
  stunned?: boolean;
  buff_counter?: boolean;
  combat_role?: string;
  keywords?: { keyword: string; value: number }[];
  abilities?: { ability_id: string; ability_type: string; timing: string; text: string }[];
  facedown?: boolean;
}

export interface BattlefieldView {
  battlefield_id: string;
  name?: string;
  card_id?: string;
  control_status: string;
  controller_id: string | null;
  contested_by: string | null;
  showdown_staged: boolean;
  combat_staged: boolean;
  units: CardView[];
  facedown_card: CardView | null;
}

export interface ChainView {
  items: ChainItemView[];
  is_empty: boolean;
}

export interface ChainItemView {
  item_id: string;
  controller_id: string;
  card?: CardView;
  ability_id?: string;
  source_name?: string;
}

export interface CombatView {
  battlefield_id: string;
  attacker_id: string;
  defender_id: string;
  phase: string;
  attacker_assigned: boolean;
  defender_assigned: boolean;
}

export interface ShowdownView {
  battlefield_id: string;
  initiator_id: string;
  focus_player_id: string;
  is_combat_showdown: boolean;
}
