"""Card database: loads card definitions from JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from .card_types import CardDefinition


class CardDB:
    """Registry of all card definitions, loaded from JSON files."""

    _registry: dict[str, CardDefinition] = {}

    @classmethod
    def load_all(cls, data_dir: str | Path) -> None:
        data_dir = Path(data_dir)
        cls._registry.clear()
        for json_file in sorted(data_dir.glob("*.json")):
            with open(json_file, encoding="utf-8") as f:
                cards = json.load(f)
            if isinstance(cards, dict):
                cards = [cards]
            for raw in cards:
                defn = CardDefinition.from_dict(raw)
                cls._registry[defn.card_id] = defn

    @classmethod
    def get(cls, card_id: str) -> CardDefinition:
        return cls._registry[card_id]

    @classmethod
    def get_or_none(cls, card_id: str) -> CardDefinition | None:
        return cls._registry.get(card_id)

    @classmethod
    def all_cards(cls) -> dict[str, CardDefinition]:
        return dict(cls._registry)

    @classmethod
    def cards_of_type(cls, card_type: str) -> list[CardDefinition]:
        return [c for c in cls._registry.values() if c.card_type.value == card_type]
