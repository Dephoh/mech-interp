"""Tests for keyword mechanics."""

import pytest
from helpers import make_game, add_unit

from app.engine.card_types import CardInstance, KeywordInstance
from app.engine.enums import CombatRole, Domain, Keyword, ZoneType
from app.engine.keywords import (
    apply_accelerate,
    can_play_in_state,
    check_lethal_before_next,
    check_tank_ordering,
    get_accelerate_cost,
)
from helpers import make_unit_def


class TestAssaultShield:
    def test_assault_adds_might_as_attacker(self):
        defn = make_unit_def(
            might=3,
            keywords=(KeywordInstance(keyword=Keyword.ASSAULT, value=2),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        inst.combat_role = CombatRole.ATTACKER
        assert inst.effective_might == 5  # 3 + 2

    def test_assault_no_bonus_as_defender(self):
        defn = make_unit_def(
            might=3,
            keywords=(KeywordInstance(keyword=Keyword.ASSAULT, value=2),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        inst.combat_role = CombatRole.DEFENDER
        assert inst.effective_might == 3

    def test_shield_adds_might_as_defender(self):
        defn = make_unit_def(
            might=3,
            keywords=(KeywordInstance(keyword=Keyword.SHIELD, value=2),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        inst.combat_role = CombatRole.DEFENDER
        assert inst.effective_might == 5

    def test_shield_no_bonus_as_attacker(self):
        defn = make_unit_def(
            might=3,
            keywords=(KeywordInstance(keyword=Keyword.SHIELD, value=2),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        inst.combat_role = CombatRole.ATTACKER
        assert inst.effective_might == 3


class TestStunned:
    def test_stunned_combat_might_zero(self):
        defn = make_unit_def(might=5)
        inst = CardInstance.create(defn, "p1", ZoneType.BATTLEFIELD)
        inst.stunned = True
        assert inst.combat_might == 0
        assert inst.effective_might == 5  # still has might, just can't deal damage


class TestAccelerate:
    def test_accelerate_unexhausts_unit(self):
        defn = make_unit_def(
            keywords=(KeywordInstance(keyword=Keyword.ACCELERATE),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BASE)
        inst.exhausted = True
        apply_accelerate(inst, paid=True)
        assert not inst.exhausted

    def test_accelerate_not_paid(self):
        defn = make_unit_def(
            keywords=(KeywordInstance(keyword=Keyword.ACCELERATE),),
        )
        inst = CardInstance.create(defn, "p1", ZoneType.BASE)
        inst.exhausted = True
        apply_accelerate(inst, paid=False)
        assert inst.exhausted

    def test_get_accelerate_cost(self):
        defn = make_unit_def(domains=(Domain.FURY,))
        inst = CardInstance.create(defn, "p1", ZoneType.BASE)
        energy, power = get_accelerate_cost(inst)
        assert energy == 1
        assert power == {Domain.FURY: 1}


class TestCanPlayInState:
    def test_reaction_in_closed(self):
        kws = [KeywordInstance(keyword=Keyword.REACTION)]
        assert can_play_in_state(kws, is_showdown=False, is_closed=True)

    def test_action_blocked_in_closed(self):
        kws = [KeywordInstance(keyword=Keyword.ACTION)]
        assert not can_play_in_state(kws, is_showdown=False, is_closed=True)

    def test_action_in_showdown_open(self):
        kws = [KeywordInstance(keyword=Keyword.ACTION)]
        assert can_play_in_state(kws, is_showdown=True, is_closed=False)

    def test_any_card_in_neutral_open(self):
        kws = []
        assert can_play_in_state(kws, is_showdown=False, is_closed=False)


class TestBuffCounter:
    def test_buff_counter_adds_one_might(self):
        defn = make_unit_def(might=3)
        inst = CardInstance.create(defn, "p1", ZoneType.BASE)
        assert inst.effective_might == 3
        inst.buff_counter = True
        assert inst.effective_might == 4
