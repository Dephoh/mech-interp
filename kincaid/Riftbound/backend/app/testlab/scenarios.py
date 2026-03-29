"""Scenario definition dataclass for testlab."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScenarioDef:
    scenario_id: str          # "keyword_deathknell"
    name: str                 # "Deathknell Units"
    description: str          # What this tests
    category: str             # "keyword", "ir_type", "card_type", "individual"
    tags: list[str]           # ["deathknell", "triggered"]
    expected_behavior: str    # Human instructions for what to try
    p1_hand: list[str]        # card_ids in your hand
    p1_base_units: list[str]  # card_ids in your base
    p1_bf_units: dict[int, list[str]]  # {bf_index: [card_ids]} on battlefields
    p2_bf_units: dict[int, list[str]]  # enemy units as targets
    energy: int = 99
    power: dict[str, int] = field(default_factory=lambda: {
        "fury": 99, "calm": 99, "mind": 99, "body": 99, "chaos": 99, "order": 99,
    })
