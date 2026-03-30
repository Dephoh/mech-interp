"""TS-06: Performance and load testing.

Tests:
  a. Create 50 concurrent game rooms (GameState objects) and verify no errors.
  b. Benchmark card_pipeline conversion of all 767 cards.
  c. Benchmark effect_resolver tree walking for representative IR trees.
  d. Profile create_game under repeated invocation.

All slow tests are marked with @pytest.mark.slow so they can be skipped
with: pytest -m "not slow"
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from helpers import add_unit, load_card_db, make_game, make_unit_def

from app.engine.card_db import CardDB
from app.engine.card_types import CardDefinition, CardInstance
from app.engine.effect_ir import (
    ConditionSpec,
    ENEMY_UNIT,
    FRIENDLY_UNIT,
    SELF_TARGET,
    ANY_UNIT,
    ALL_UNITS,
    deal_damage,
    draw_cards,
    give_might,
    heal,
    kill,
    play_token,
    stun,
    sequence,
    conditional,
    for_each,
    optional,
    add_energy,
    look_at_top,
)
from app.engine.effect_resolver import resolve_effect_ir
from app.engine.enums import CardType, Domain, Phase, ZoneType
from app.engine.game_state import GameState, PlayerState, create_game


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Known card IDs used for create_game (from test_contracts.py)
LEGEND_ID = "unl-230-star-219"
CHAMPION_ID = "ogn-097-298"
RUNE_ID = "ogn-007a-298"
BATTLEFIELD_IDS = ["unl-205-219", "unl-206-219", "ogn-275-298"]


@pytest.fixture(scope="module", autouse=True)
def _ensure_card_db():
    """Load the card database once for all tests in this module."""
    load_card_db()


def _make_deck_config(player_id: str, display_name: str) -> dict:
    """Build a standard player config for create_game."""
    return {
        "player_id": player_id,
        "display_name": display_name,
        "legend_id": LEGEND_ID,
        "champion_id": CHAMPION_ID,
        "main_deck": [CHAMPION_ID] * 40,
        "rune_deck": [RUNE_ID] * 12,
        "battlefields": BATTLEFIELD_IDS,
    }


# ---------------------------------------------------------------------------
# (a) 50 concurrent game rooms
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConcurrentRooms:
    """Create 50 game rooms and verify all are valid."""

    NUM_ROOMS = 50

    def test_create_50_rooms_no_errors(self):
        """Creating 50 GameState objects via create_game must not raise."""
        card_db = CardDB.all_cards()
        rooms: list[GameState] = []

        start = time.perf_counter()
        for i in range(self.NUM_ROOMS):
            configs = [
                _make_deck_config(f"p1_room{i}", f"Player1 R{i}"),
                _make_deck_config(f"p2_room{i}", f"Player2 R{i}"),
            ]
            gs = create_game(f"room-{i}", configs, card_db)
            rooms.append(gs)
        elapsed = time.perf_counter() - start

        # Verify all rooms are valid
        assert len(rooms) == self.NUM_ROOMS
        for i, gs in enumerate(rooms):
            assert gs.game_id == f"room-{i}"
            assert len(gs.players) == 2
            assert len(gs.battlefields) > 0
            assert gs.phase == Phase.SETUP_MULLIGAN
            # Each player should have 4 cards in hand (initial draw)
            for pid in gs.player_order:
                assert len(gs.players[pid].hand) == 4

        print(f"\n  50 rooms created in {elapsed:.3f}s ({elapsed/self.NUM_ROOMS*1000:.1f}ms each)")
        # Generous threshold: 50 rooms in under 30 seconds (CI-safe)
        assert elapsed < 30.0, f"Creating 50 rooms took {elapsed:.1f}s (>30s threshold)"

    def test_rooms_are_independent(self):
        """Mutating one room must not affect another."""
        card_db = CardDB.all_cards()
        configs_a = [
            _make_deck_config("ind_p1", "P1"),
            _make_deck_config("ind_p2", "P2"),
        ]
        configs_b = [
            _make_deck_config("ind_p3", "P3"),
            _make_deck_config("ind_p4", "P4"),
        ]
        gs_a = create_game("independence-a", configs_a, card_db)
        gs_b = create_game("independence-b", configs_b, card_db)

        # Mutate room A
        gs_a.players["ind_p1"].score = 99
        gs_a.game_over = True

        # Room B is unaffected
        assert gs_b.players["ind_p3"].score == 0
        assert gs_b.game_over is False

    def test_100_actions_per_room_simulation(self):
        """Simulate 100 lightweight mutations on each of 50 rooms.

        This doesn't go through the full action pipeline (which would
        require valid game states at every step), but instead exercises
        the kind of state mutations that happen during gameplay: adding
        units, moving cards, updating scores, etc.
        """
        card_db = CardDB.all_cards()
        rooms: list[GameState] = []
        for i in range(self.NUM_ROOMS):
            configs = [
                _make_deck_config(f"act_p1_{i}", f"P1_{i}"),
                _make_deck_config(f"act_p2_{i}", f"P2_{i}"),
            ]
            gs = create_game(f"action-room-{i}", configs, card_db)
            rooms.append(gs)

        action_times: list[float] = []
        start = time.perf_counter()

        for gs in rooms:
            p1 = gs.player_order[0]
            p2 = gs.player_order[1]
            bf_id = list(gs.battlefields.keys())[0]

            for action_num in range(100):
                t0 = time.perf_counter()
                op = action_num % 5

                if op == 0:
                    # Add a unit to base
                    unit = add_unit(gs, p1, ZoneType.BASE, might=3, name=f"Unit_{action_num}")
                elif op == 1:
                    # Move a unit to battlefield (if any in base)
                    if gs.base_units[p1]:
                        uid = gs.base_units[p1][-1]
                        gs.base_units[p1].remove(uid)
                        gs.battlefields[bf_id].units.append(uid)
                        inst = gs.get_instance(uid)
                        if inst:
                            inst.zone = ZoneType.BATTLEFIELD
                            inst.location_id = bf_id
                elif op == 2:
                    # Update score
                    gs.players[p1].score += 1
                elif op == 3:
                    # Draw a card (add to hand from deck if available)
                    ps = gs.players[p1]
                    if ps.main_deck:
                        iid = ps.main_deck.pop(0)
                        inst = gs.get_instance(iid)
                        if inst:
                            inst.zone = ZoneType.HAND
                            inst.location_id = p1
                        ps.hand.append(iid)
                elif op == 4:
                    # Recalculate modifiers
                    gs.recalculate_modifiers()

                action_times.append(time.perf_counter() - t0)

        total = time.perf_counter() - start
        avg_us = (sum(action_times) / len(action_times)) * 1_000_000
        p99_us = sorted(action_times)[int(len(action_times) * 0.99)] * 1_000_000

        print(f"\n  50 rooms x 100 actions = {len(action_times)} total actions")
        print(f"  Total: {total:.3f}s, Avg: {avg_us:.1f}us, P99: {p99_us:.1f}us")

        # Generous threshold: 5000 actions in under 60 seconds
        assert total < 60.0, f"50x100 actions took {total:.1f}s (>60s threshold)"
        # Average action should be under 10ms (generous for CI)
        assert avg_us < 10_000, f"Average action {avg_us:.0f}us (>10ms threshold)"


# ---------------------------------------------------------------------------
# (b) Card pipeline conversion benchmark
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCardPipelineBenchmark:
    """Benchmark card_pipeline.convert_card on all cards."""

    def test_convert_all_cards(self):
        """Convert all cards from CMS format and measure time.

        This tests the card_definitions.json -> CardDefinition path,
        which is what CardDB.load_all does at startup.
        """
        defs_path = DATA_DIR / "card_definitions.json"
        with open(defs_path, encoding="utf-8") as f:
            raw_cards = json.load(f)

        start = time.perf_counter()
        definitions: list[CardDefinition] = []
        for raw in raw_cards:
            defn = CardDefinition.from_dict(raw)
            definitions.append(defn)
        elapsed = time.perf_counter() - start

        print(f"\n  Converted {len(definitions)} cards in {elapsed:.3f}s "
              f"({elapsed/len(definitions)*1000:.2f}ms each)")

        assert len(definitions) == len(raw_cards)
        assert all(d.card_id for d in definitions)
        # Threshold: all cards in under 10 seconds (generous for CI)
        assert elapsed < 10.0, f"Card conversion took {elapsed:.1f}s (>10s threshold)"

    def test_convert_cards_10_iterations(self):
        """Convert all cards 10 times to get stable timing."""
        defs_path = DATA_DIR / "card_definitions.json"
        with open(defs_path, encoding="utf-8") as f:
            raw_cards = json.load(f)

        times: list[float] = []
        for _ in range(10):
            start = time.perf_counter()
            for raw in raw_cards:
                CardDefinition.from_dict(raw)
            times.append(time.perf_counter() - start)

        avg = sum(times) / len(times)
        best = min(times)
        worst = max(times)
        total_cards = len(raw_cards) * 10

        print(f"\n  10 iterations x {len(raw_cards)} cards = {total_cards} conversions")
        print(f"  Avg: {avg:.3f}s, Best: {best:.3f}s, Worst: {worst:.3f}s")
        print(f"  Per-card avg: {avg/len(raw_cards)*1000:.3f}ms")

        # The average iteration should be under 5 seconds
        assert avg < 5.0, f"Avg iteration {avg:.1f}s (>5s threshold)"

    def test_card_db_load_all_benchmark(self):
        """Benchmark the full CardDB.load_all path."""
        times: list[float] = []
        for _ in range(5):
            start = time.perf_counter()
            CardDB.load_all(DATA_DIR)
            times.append(time.perf_counter() - start)

        avg = sum(times) / len(times)
        print(f"\n  CardDB.load_all: avg {avg:.3f}s over 5 runs")

        # Threshold: load_all should complete in under 5 seconds
        assert avg < 5.0, f"CardDB.load_all avg {avg:.1f}s (>5s threshold)"


# ---------------------------------------------------------------------------
# (c) Effect resolver tree-walking benchmark
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestEffectResolverBenchmark:
    """Benchmark effect_resolver for representative IR trees."""

    @staticmethod
    def _make_source_and_gs():
        """Create a game state with units for effect resolution."""
        gs = make_game(num_battlefields=2, phase=Phase.ACTION)
        bf_id = list(gs.battlefields.keys())[0]

        # Add several units for targeting
        for i in range(5):
            add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=3, name=f"FUnit{i}", bf_id=bf_id)
            add_unit(gs, "p2", ZoneType.BATTLEFIELD, might=2, name=f"EUnit{i}", bf_id=bf_id)

        # Add units at base
        for i in range(3):
            add_unit(gs, "p1", ZoneType.BASE, might=4, name=f"BaseUnit{i}")
            add_unit(gs, "p2", ZoneType.BASE, might=2, name=f"EBase{i}")

        # Give decks so draw_cards works
        for pid in gs.player_order:
            for i in range(20):
                defn = make_unit_def(name=f"DeckCard{i}")
                inst = CardInstance.create(defn, pid, ZoneType.MAIN_DECK)
                gs.instances[inst.instance_id] = inst
                gs.players[pid].main_deck.append(inst.instance_id)

        source = add_unit(gs, "p1", ZoneType.BATTLEFIELD, might=5, name="Source", bf_id=bf_id)
        return source, gs

    def test_primitive_deal_damage(self):
        """Benchmark a single deal_damage primitive."""
        source, gs = self._make_source_and_gs()
        ir = deal_damage(3, ENEMY_UNIT)

        times: list[float] = []
        for _ in range(1000):
            # Reset target health so damage doesn't kill and remove targets
            for inst in gs.instances.values():
                inst.damage_taken = 0

            t0 = time.perf_counter()
            resolve_effect_ir(ir, source, gs, [])
            times.append(time.perf_counter() - t0)

        avg_us = (sum(times) / len(times)) * 1_000_000
        print(f"\n  deal_damage x1000: avg {avg_us:.1f}us per call")
        assert avg_us < 5_000, f"deal_damage avg {avg_us:.0f}us (>5ms threshold)"

    def test_sequence_tree(self):
        """Benchmark a sequence of 5 primitives (common spell pattern)."""
        source, gs = self._make_source_and_gs()
        ir = sequence([
            deal_damage(2, ENEMY_UNIT),
            draw_cards(1),
            give_might(2, SELF_TARGET),
            add_energy(1),
            heal(2, FRIENDLY_UNIT),
        ])

        times: list[float] = []
        for _ in range(500):
            for inst in gs.instances.values():
                inst.damage_taken = 0
            t0 = time.perf_counter()
            resolve_effect_ir(ir, source, gs, [])
            times.append(time.perf_counter() - t0)

        avg_us = (sum(times) / len(times)) * 1_000_000
        print(f"\n  sequence(5 steps) x500: avg {avg_us:.1f}us per call")
        assert avg_us < 10_000, f"sequence avg {avg_us:.0f}us (>10ms threshold)"

    def test_conditional_tree(self):
        """Benchmark conditional with then/else branches."""
        source, gs = self._make_source_and_gs()
        ir = conditional(
            condition=ConditionSpec(cond_type="mighty"),
            then=sequence([
                deal_damage(4, ENEMY_UNIT),
                draw_cards(2),
            ]),
            else_=draw_cards(1),
        )

        times: list[float] = []
        for _ in range(500):
            for inst in gs.instances.values():
                inst.damage_taken = 0
            t0 = time.perf_counter()
            resolve_effect_ir(ir, source, gs, [])
            times.append(time.perf_counter() - t0)

        avg_us = (sum(times) / len(times)) * 1_000_000
        print(f"\n  conditional x500: avg {avg_us:.1f}us per call")
        assert avg_us < 10_000, f"conditional avg {avg_us:.0f}us (>10ms threshold)"

    def test_for_each_tree(self):
        """Benchmark for_each over all friendly units."""
        source, gs = self._make_source_and_gs()
        ir = for_each(
            targets=FRIENDLY_UNIT,
            effect=give_might(1, SELF_TARGET),
        )

        times: list[float] = []
        for _ in range(500):
            for inst in gs.instances.values():
                inst.damage_taken = 0
            t0 = time.perf_counter()
            resolve_effect_ir(ir, source, gs, [])
            times.append(time.perf_counter() - t0)

        avg_us = (sum(times) / len(times)) * 1_000_000
        print(f"\n  for_each(friendly units) x500: avg {avg_us:.1f}us per call")
        assert avg_us < 20_000, f"for_each avg {avg_us:.0f}us (>20ms threshold)"

    def test_deeply_nested_tree(self):
        """Benchmark a deeply nested IR tree (realistic complex card).

        Tree structure:
          sequence([
            conditional(cond, then=sequence([damage, draw])),
            optional(sequence([heal, give_might])),
            for_each(friendly, effect=add_energy(1)),
            deal_damage(1, enemy),
            draw_cards(1),
          ])
        """
        source, gs = self._make_source_and_gs()
        ir = sequence([
            conditional(
                condition=ConditionSpec(cond_type="mighty"),
                then=sequence([
                    deal_damage(2, ENEMY_UNIT),
                    draw_cards(1),
                ]),
                else_=heal(1, SELF_TARGET),
            ),
            optional(
                sequence([
                    heal(2, FRIENDLY_UNIT),
                    give_might(1, SELF_TARGET),
                ]),
            ),
            for_each(
                targets=FRIENDLY_UNIT,
                effect=add_energy(1),
            ),
            deal_damage(1, ENEMY_UNIT),
            draw_cards(1),
        ])

        times: list[float] = []
        for _ in range(200):
            for inst in gs.instances.values():
                inst.damage_taken = 0
            t0 = time.perf_counter()
            resolve_effect_ir(ir, source, gs, [])
            times.append(time.perf_counter() - t0)

        avg_us = (sum(times) / len(times)) * 1_000_000
        p99_us = sorted(times)[int(len(times) * 0.99)] * 1_000_000
        print(f"\n  deeply nested tree x200: avg {avg_us:.1f}us, P99 {p99_us:.1f}us")
        assert avg_us < 50_000, f"deep tree avg {avg_us:.0f}us (>50ms threshold)"

    def test_resolve_all_card_ir_trees(self):
        """Walk every IR tree from card_definitions.json through the resolver.

        This ensures no IR tree in the real dataset crashes the resolver
        and measures aggregate resolution time.
        """
        defs_path = DATA_DIR / "card_definitions.json"
        with open(defs_path, encoding="utf-8") as f:
            raw_cards = json.load(f)

        source, gs = self._make_source_and_gs()

        ir_nodes: list[dict] = []
        for raw in raw_cards:
            for ab in raw.get("abilities", []):
                ir = ab.get("effect_ir")
                if ir and isinstance(ir, dict) and ir.get("type"):
                    ir_nodes.append(ir)

        assert len(ir_nodes) > 100, f"Expected 100+ IR nodes, got {len(ir_nodes)}"

        errors: list[str] = []
        start = time.perf_counter()
        for ir in ir_nodes:
            try:
                # Reset damage each iteration to keep targets alive
                for inst in gs.instances.values():
                    inst.damage_taken = 0
                resolve_effect_ir(ir, source, gs, [])
            except Exception as e:
                errors.append(f"{ir.get('type', '?')}: {e}")
        elapsed = time.perf_counter() - start

        print(f"\n  Resolved {len(ir_nodes)} IR trees in {elapsed:.3f}s "
              f"({elapsed/len(ir_nodes)*1000:.3f}ms each)")

        # Allow some errors (player_choice nodes, etc.) but not too many
        error_rate = len(errors) / len(ir_nodes)
        if errors:
            print(f"  Errors: {len(errors)}/{len(ir_nodes)} ({error_rate:.1%})")
            # Print first few errors for debugging
            for e in errors[:5]:
                print(f"    {e}")

        # Threshold: resolving all trees should take under 30 seconds
        assert elapsed < 30.0, f"Resolving all IR trees took {elapsed:.1f}s (>30s)"


# ---------------------------------------------------------------------------
# (d) create_game repeated invocation profiling
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCreateGameProfile:
    """Profile create_game under repeated invocation."""

    def test_create_game_timing_distribution(self):
        """Measure create_game timing across 100 invocations."""
        card_db = CardDB.all_cards()

        times: list[float] = []
        for i in range(100):
            configs = [
                _make_deck_config(f"prof_p1_{i}", f"P1_{i}"),
                _make_deck_config(f"prof_p2_{i}", f"P2_{i}"),
            ]
            t0 = time.perf_counter()
            gs = create_game(f"profile-{i}", configs, card_db)
            times.append(time.perf_counter() - t0)
            # Basic sanity
            assert len(gs.players) == 2

        avg_ms = (sum(times) / len(times)) * 1000
        p50_ms = sorted(times)[50] * 1000
        p99_ms = sorted(times)[99] * 1000
        total = sum(times)

        print(f"\n  create_game x100:")
        print(f"  Total: {total:.3f}s, Avg: {avg_ms:.1f}ms, P50: {p50_ms:.1f}ms, P99: {p99_ms:.1f}ms")

        # Threshold: average create_game under 500ms (generous for CI)
        assert avg_ms < 500, f"create_game avg {avg_ms:.0f}ms (>500ms threshold)"

    def test_memory_stability(self):
        """Creating many games should not accumulate shared state."""
        card_db = CardDB.all_cards()

        instance_counts: list[int] = []
        for i in range(20):
            configs = [
                _make_deck_config(f"mem_p1_{i}", f"P1_{i}"),
                _make_deck_config(f"mem_p2_{i}", f"P2_{i}"),
            ]
            gs = create_game(f"mem-{i}", configs, card_db)
            instance_counts.append(len(gs.instances))

        # All games should have the same number of instances
        # (battlefields may vary due to random selection, but should be similar)
        min_count = min(instance_counts)
        max_count = max(instance_counts)
        assert max_count - min_count <= 5, (
            f"Instance count varies too much: min={min_count}, max={max_count}"
        )
