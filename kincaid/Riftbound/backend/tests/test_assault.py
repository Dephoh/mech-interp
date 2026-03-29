"""Tests for the Assault keyword (rules 733-734).

Assault N means "While attacking, I have +N Might."
- If Assault has no value, N defaults to 1 (rule 733.1.b.3).
- The bonus only applies while the unit has the Attacker combat role.
- The bonus is removed when combat ends.
- Multiple Assault sources stack (rule 733.2).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.engine.card_types import CardInstance, KeywordInstance
from app.engine.enums import CombatRole, Keyword, ZoneType
from app.engine.combat import start_combat, resolve_combat, apply_combat_damage
from helpers import add_unit, make_game


def _assault_keywords(value: int = 0) -> tuple[KeywordInstance, ...]:
    """Return a tuple with a single Assault keyword instance."""
    return (KeywordInstance(keyword=Keyword.ASSAULT, value=value),)


def _setup_two_player_battlefield(gs, p1_units, p2_units, contested_by="p1"):
    """
    Place units on a shared battlefield for both players, mark it as
    contested so start_combat will assign p1 as attacker.

    p1_units / p2_units: list of (might, name, keywords) tuples.
    Returns (bf_id, p1_instances, p2_instances).
    """
    bf_id = list(gs.battlefields.keys())[0]
    bf = gs.battlefields[bf_id]
    bf.contested_by = contested_by

    p1_insts = []
    for might, name, kws in p1_units:
        u = add_unit(gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id,
                     might=might, name=name, keywords=kws)
        p1_insts.append(u)

    p2_insts = []
    for might, name, kws in p2_units:
        u = add_unit(gs, "p2", ZoneType.BATTLEFIELD, bf_id=bf_id,
                     might=might, name=name, keywords=kws)
        p2_insts.append(u)

    return bf_id, p1_insts, p2_insts


class TestAssaultUnitGainsMightInCombat:
    """Test 1: Assault unit gains +N might when assigned as attacker during combat."""

    def test_assault_unit_gains_might_in_combat(self):
        gs = make_game()

        # P1 has an assault unit (base 3, assault +1), P2 has a vanilla defender
        bf_id, [attacker], [defender] = _setup_two_player_battlefield(
            gs,
            p1_units=[(3, "Assault Striker", _assault_keywords(0))],
            p2_units=[(5, "Tough Defender", ())],
        )

        assert attacker.effective_might == 3  # no bonus before combat

        # Start combat: p1 attacks, p2 defends
        start_combat(gs, bf_id)

        # Attacker should now have ATTACKER role
        assert attacker.combat_role == CombatRole.ATTACKER
        # Effective might = base 3 + assault 1 = 4
        assert attacker.effective_might == 4, (
            f"Expected 4 (base 3 + assault 1), got {attacker.effective_might}"
        )
        assert attacker.combat_might == 4


class TestAssaultBonusNotAppliedWhenDefending:
    """Test 2: Assault bonus does NOT apply when the unit is a defender."""

    def test_assault_bonus_not_applied_when_defending(self):
        gs = make_game()

        # P2 has the assault unit.  P1 attacks, so P2's assault unit defends.
        bf_id, [p1_unit], [p2_assault] = _setup_two_player_battlefield(
            gs,
            p1_units=[(3, "P1 Attacker", ())],
            p2_units=[(4, "P2 Assault Defender", _assault_keywords(0))],
            contested_by="p1",
        )

        start_combat(gs, bf_id)

        # P2's assault unit should be a DEFENDER — no assault bonus
        assert p2_assault.combat_role == CombatRole.DEFENDER
        assert p2_assault.effective_might == 4, (
            f"Assault should NOT apply to defenders: expected 4, "
            f"got {p2_assault.effective_might}"
        )


class TestAssaultBonusRemovedAfterCombat:
    """Test 3: After combat resolves, the assault unit reverts to base might."""

    def test_assault_bonus_removed_after_combat(self):
        gs = make_game()

        # Both sides survive: attacker (5 might w/ assault) vs defender (10 might).
        # Assault unit (base 4, assault +1 = eff 5) attacks, defender has 10 might.
        # Attacker deals 5 < 10, so defender survives. Defender deals 10 >= 5,
        # so attacker dies. But we need the attacker to survive too to test
        # role removal. Use high-might attacker so both survive.
        bf_id, [attacker], [defender] = _setup_two_player_battlefield(
            gs,
            p1_units=[(10, "Beefy Assault", _assault_keywords(0))],
            p2_units=[(10, "Beefy Defender", ())],
        )

        start_combat(gs, bf_id)

        # During combat: assault bonus active
        assert attacker.combat_role == CombatRole.ATTACKER
        assert attacker.effective_might == 11  # 10 + 1

        # Close the showdown so combat proceeds to damage
        # Simulate both players passing in the showdown
        from app.engine.combat import pass_focus
        pass_focus(gs, "p1")
        pass_focus(gs, "p2")

        # After combat resolves, roles are cleared
        assert attacker.combat_role == CombatRole.NONE, (
            f"Expected NONE after combat, got {attacker.combat_role}"
        )
        # Might should be back to base (no assault bonus)
        assert attacker.effective_might == 10, (
            f"Expected base 10 after combat, got {attacker.effective_might}"
        )


class TestAssaultStacksWithBuffs:
    """Test 4: Assault stacks with temporary might buffs."""

    def test_assault_stacks_with_buffs(self):
        gs = make_game()

        bf_id, [attacker], [defender] = _setup_two_player_battlefield(
            gs,
            p1_units=[(3, "Buffed Assault", _assault_keywords(0))],
            p2_units=[(10, "Wall", ())],
        )

        # Apply a +2 temporary buff
        attacker.might_modifiers.append(2)

        # Before combat: base 3 + buff 2 = 5 (no assault yet)
        assert attacker.effective_might == 5

        start_combat(gs, bf_id)

        # During combat as attacker: base 3 + buff 2 + assault 1 = 6
        assert attacker.combat_role == CombatRole.ATTACKER
        assert attacker.effective_might == 6, (
            f"Expected 6 (base 3 + buff 2 + assault 1), got {attacker.effective_might}"
        )


class TestAssaultDamageDealtCorrectly:
    """Test 5: Assault unit deals combat_might damage, not base might."""

    def test_assault_damage_dealt_correctly(self):
        gs = make_game()

        # Assault unit: base 3 + assault 1 = effective 4 while attacking
        # Defender: might 4.
        # Attacker deals 4 damage (enough to kill defender with might 4).
        bf_id, [attacker], [defender] = _setup_two_player_battlefield(
            gs,
            p1_units=[(3, "Assault Fighter", _assault_keywords(0))],
            p2_units=[(4, "Defender", ())],
        )

        start_combat(gs, bf_id)

        assert attacker.combat_role == CombatRole.ATTACKER
        assert attacker.combat_might == 4  # 3 base + 1 assault

        # Fast-forward past showdown: simulate both players passing
        from app.engine.combat import pass_focus
        pass_focus(gs, "p1")
        # pass_focus for p2 triggers close_showdown -> execute_combat_damage
        pass_focus(gs, "p2")

        # After combat, defender should have taken 4 damage.
        # The defender has might 4, so 4 damage is lethal => destroyed => in trash.
        assert defender.zone == ZoneType.TRASH, (
            f"Expected defender in TRASH (lethal damage from assault), "
            f"got zone={defender.zone}"
        )


class TestAssaultValueZeroTreatedAsOne:
    """Test 6: Assault with value=0 in data is treated as +1 (rule 733.1.b.3)."""

    def test_assault_value_zero_treated_as_one(self):
        gs = make_game()
        bf_id = list(gs.battlefields.keys())[0]

        unit = add_unit(
            gs, "p1", ZoneType.BATTLEFIELD, bf_id=bf_id,
            might=3, name="Bare Assault",
            keywords=_assault_keywords(0),
        )

        # keyword_value should interpret 0 as 1 for Assault
        assert unit.keyword_value(Keyword.ASSAULT) == 1, (
            f"Bare Assault should default to value 1, "
            f"got {unit.keyword_value(Keyword.ASSAULT)}"
        )

        # As attacker, effective_might = 3 + 1 = 4, not 3 + 0 = 3
        unit.combat_role = CombatRole.ATTACKER
        assert unit.effective_might == 4, (
            f"Expected 4 (base 3 + assault 1), got {unit.effective_might}"
        )
