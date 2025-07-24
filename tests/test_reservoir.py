import pytest
from math import isclose

from RWHmodel import Reservoir

def test_update_state_no_overflow_enough_supply():
    r = Reservoir(reservoir_cap=100.0, reservoir_stor=50.0)
    # runoff 30, demand 60 => available=80, after demand stor=20, no overflow, no deficit
    r.update_state(runoff=30.0, demand=60.0)
    assert r.reservoir_stor == pytest.approx(20.0)
    assert r.deficit == pytest.approx(0.0)
    assert r.reservoir_overflow == pytest.approx(0.0)

def test_update_state_with_overflow():
    r = Reservoir(reservoir_cap=100.0, reservoir_stor=80.0)
    # available = 80 + 50 = 130
    # after demand = 130 - 20 = 110
    # reservoir capacity = 100
    # overflow = 110 - 100 = 10
    r.update_state(runoff=50.0, demand=20.0)
    assert r.reservoir_stor == pytest.approx(100.0)
    assert r.reservoir_overflow == pytest.approx(10.0)
    assert r.deficit == pytest.approx(0.0)

def test_update_state_exact_capacity():
    r = Reservoir(reservoir_cap=100.0, reservoir_stor=60.0)
    # runoff 40, demand 0 => stor=100, no overflow, no deficit
    r.update_state(runoff=40.0, demand=0.0)
    assert r.reservoir_stor == pytest.approx(100.0)
    assert r.reservoir_overflow == pytest.approx(0.0)
    assert r.deficit == pytest.approx(0.0)

def test_update_state_deficit_some_supply():
    r = Reservoir(reservoir_cap=100.0, reservoir_stor=30.0)
    # runoff 10, demand 50 => available=40, stor=0, deficit=10
    r.update_state(runoff=10.0, demand=50.0)
    assert r.reservoir_stor == pytest.approx(0.0)
    assert r.deficit == pytest.approx(10.0)
    assert r.reservoir_overflow == pytest.approx(0.0)

def test_update_state_deficit_no_supply():
    r = Reservoir(reservoir_cap=100.0, reservoir_stor=0.0)
    # runoff 0, demand 20 => available=0, stor=0, deficit=20
    r.update_state(runoff=0.0, demand=20.0)
    assert r.reservoir_stor == pytest.approx(0.0)
    assert r.deficit == pytest.approx(20.0)
    assert r.reservoir_overflow == pytest.approx(0.0)