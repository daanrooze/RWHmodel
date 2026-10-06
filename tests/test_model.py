import os
from types import SimpleNamespace

import pandas as pd
import pytest
from RWHmodel import Model
from RWHmodel.reservoir import Reservoir

# Root folder for tests
TEST_ROOT = "C:/repos/RWHmodel/tests"

# Setup files (expected at ./tests/)
SETUP_SINGLE_FILE = "setup_single_test.toml"
SETUP_BATCH_FILE = "setup_batch_test.toml"

# Data files (expected inside ./tests/input/)
FORCING_FILE = os.path.join(TEST_ROOT, "input", "forcing_test.csv")
DEMAND_FILE = os.path.join(TEST_ROOT, "input", "demand_test.csv")

def test_model_run():
    model = Model(
        root=TEST_ROOT,
        name="test_run",
        mode="single",
        setup_fn=SETUP_SINGLE_FILE,
        forcing_fn=FORCING_FILE,
        demand_fn=DEMAND_FILE,
        demand_transform=False,
        reservoir_initial_state=0.5,
        timestep=86400,
        t_start=None,
        t_end=None,
        unit="mm",
    )

    df = model.run(save=False)

    # Basic assertions
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['reservoir_stor', 'reservoir_overflow', 'demand', 'deficit', 'deficit_timesteps']
    for col in expected_cols:
        assert col in df.columns

    # No negative storage
    assert (df['reservoir_stor'] >= 0).all()

def test_model_batch_run():
    demand_range = [0.1, 2.1, 25]

    model = Model(
        root=TEST_ROOT,
        name="test_batch",
        mode="batch",
        setup_fn=SETUP_BATCH_FILE,
        forcing_fn=FORCING_FILE,
        demand_fn=demand_range,
        demand_transform=False,
        reservoir_range=[5, 105, 25],
        reservoir_initial_state=0.5,
        timestep=86400,
        t_start=None,
        t_end=None,
        unit="mm",
    )

    model.batch_run(method="consecutive_timesteps", log=False, save=False)

    assert hasattr(model, "statistics")
    assert isinstance(model.statistics, pd.DataFrame)


@pytest.mark.parametrize(
    "runoff, expected_events",
    [
        ([5, 0, 0], [0, 0, 2]),
        ([0, 0, 5], [0, 0, 2]),
        ([0, 0, 0], [0, 0, 3]),
        ([5, 5, 5], [0, 0, 0]),
        ([0, 5, 0], [0, 1, 1]),
        ([5], [0]),
        ([0], [1]),
        ([], []),
    ],
)
def test_model_run_deficit_events(runoff, expected_events):
    index = pd.date_range("2000-01-01", periods=len(runoff))
    model = Model.__new__(Model)
    model.forcing = SimpleNamespace(
        data=pd.DataFrame({"precip": 0.0, "pet": 0.0}, index=index)
    )
    model.demand = SimpleNamespace(
        data=pd.DataFrame({"demand": 5.0}, index=index)
    )
    model.reservoir = Reservoir(reservoir_cap=1.0, reservoir_stor=0.0)
    model.runoff_source = "user"
    model.user_runoff = pd.Series(runoff, index=index, dtype=float)
    model.unit = "mm"
    model.mode = "batch"

    results = model.run(save=False)

    assert results["deficit_timesteps"].tolist() == expected_events
    assert results["deficit"].gt(0).tolist() == [flow < 5 for flow in runoff]
