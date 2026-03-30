"""All enumerations used by the Riftbound game engine."""

from __future__ import annotations

from enum import Enum, auto


class Phase(str, Enum):
    """Phases of a turn, plus combat sub-phases."""

    SETUP_MULLIGAN = "setup_mulligan"
    AWAKEN = "awaken"
    BEGINNING = "beginning"
    CHANNEL = "channel"
    DRAW = "draw"
    ACTION = "action"
    END_OF_TURN = "end_of_turn"
    # Combat sub-phases (pause the ACTION phase)
    SHOWDOWN = "showdown"
    COMBAT_DAMAGE = "combat_damage"
    COMBAT_RESOLUTION = "combat_resolution"
    # Terminal
    GAME_OVER = "game_over"


class ZoneType(str, Enum):
    BASE = "base"
    BATTLEFIELD = "battlefield"
    HAND = "hand"
    MAIN_DECK = "main_deck"
    RUNE_DECK = "rune_deck"
    RUNE_BOARD = "rune_board"
    LEGEND_ZONE = "legend_zone"
    CHAMPION_ZONE = "champion_zone"
    TRASH = "trash"
    BANISHMENT = "banishment"
    FACEDOWN_ZONE = "facedown_zone"
    CHAIN = "chain"


class CardType(str, Enum):
    UNIT = "unit"
    GEAR = "gear"
    SPELL = "spell"
    RUNE = "rune"
    LEGEND = "legend"
    BATTLEFIELD = "battlefield"


class Domain(str, Enum):
    FURY = "fury"      # Red  [R]
    CALM = "calm"      # Green [G]
    MIND = "mind"      # Blue [B]
    BODY = "body"      # Orange [O]
    CHAOS = "chaos"    # Purple [P]
    ORDER = "order"    # Yellow [Y]
    ANY = "any"        # Any domain [A]


class Keyword(str, Enum):
    ACCELERATE = "accelerate"
    ACTION = "action"
    AMBUSH = "ambush"
    ASSAULT = "assault"
    BACKLINE = "backline"
    DEATHKNELL = "deathknell"
    DEFLECT = "deflect"
    EQUIP = "equip"
    GANKING = "ganking"
    HIDDEN = "hidden"
    HUNT = "hunt"
    LEGION = "legion"
    LEVEL = "level"
    MIGHTY = "mighty"
    PREDICT = "predict"
    QUICK_DRAW = "quick_draw"
    REACTION = "reaction"
    REPEAT = "repeat"
    SHIELD = "shield"
    TANK = "tank"
    TEMPORARY = "temporary"
    UNIQUE = "unique"
    VISION = "vision"
    WEAPONMASTER = "weaponmaster"


class ControlStatus(str, Enum):
    UNCONTROLLED = "uncontrolled"
    CONTROLLED = "controlled"  # check controller_id on BattlefieldState
    CONTESTED = "contested"


class TurnState(str, Enum):
    """Combined turn state (Neutral/Showdown x Open/Closed)."""

    NEUTRAL_OPEN = "neutral_open"
    NEUTRAL_CLOSED = "neutral_closed"
    SHOWDOWN_OPEN = "showdown_open"
    SHOWDOWN_CLOSED = "showdown_closed"


class CombatRole(str, Enum):
    ATTACKER = "attacker"
    DEFENDER = "defender"
    NONE = "none"


class AbilityType(str, Enum):
    ACTIVATED = "activated"
    TRIGGERED = "triggered"
    PASSIVE = "passive"
    REPLACEMENT = "replacement"


class ActionType(str, Enum):
    """All client action types."""

    JOIN_ROOM = "join_room"
    MULLIGAN_CHOICE = "mulligan_choice"
    ADVANCE_PHASE = "advance_phase"
    PASS_PRIORITY = "pass_priority"
    PASS_FOCUS = "pass_focus"
    PLAY_CARD = "play_card"
    ACTIVATE_ABILITY = "activate_ability"
    MOVE_UNIT = "move_unit"
    EXHAUST_RUNE = "exhaust_rune"
    RECYCLE_RUNE = "recycle_rune"
    HIDE_CARD = "hide_card"
    ASSIGN_DAMAGE = "assign_damage"
    CHANNEL_RUNE = "channel_rune"
    CONCEDE = "concede"
    SUBMIT_CHOICE = "submit_choice"


class GameMode(str, Enum):
    """Sanctioned modes of play (rules 462-466)."""

    STANDARD_1V1 = "standard_1v1"   # 2 players, 2 battlefields, victory=8
    FFA3 = "ffa3"                   # 3 players, 3 battlefields, victory=8
    FFA4 = "ffa4"                   # 4 players, 3 battlefields, victory=8
    TWO_VS_TWO = "2v2"             # 4 players, 2 teams, 3 battlefields, victory=11


class SuperType(str, Enum):
    CHAMPION = "champion"
    SIGNATURE = "signature"
    TOKEN = "token"
