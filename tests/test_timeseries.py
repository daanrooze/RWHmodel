import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import pytest

from RWHmodel.timeseries import Forcing, Demand


# ----------------------------
# Forcing tests

def test_read_forcing(tmpdir):
    forcing = Forcing(forcing_fn="tests/data/forcing_test.csv", root=str(tmpdir))
    assert isinstance(forcing.data, pd.DataFrame)
    assert all([col in forcing.data.columns for col in ["precip", "pet"]])
    assert isinstance(forcing.data.index, pd.DatetimeIndex)


def test_write_forcing(tmpdir):
    forcing = Forcing(forcing_fn="tests/data/forcing_test.csv", root=str(tmpdir))
    os.makedirs(os.path.join(tmpdir, "output", "runs"), exist_ok=True)

    # Call write without expecting a return
    forcing.write(fn_out="test_forcing")

    # Build the expected path and assert
    expected_path = os.path.join(tmpdir, "output", "runs", "test_forcing.csv")
    assert os.path.exists(expected_path)


# ----------------------------
# Demand tests

def make_dummy_forcing_df():
    """Helper to build a simple forcing DataFrame in memory."""
    # Create hourly data for 2 days
    start = pd.Timestamp("2020-01-01 00:00")
    idx = pd.date_range(start=start, periods=48, freq="h")
    data = pd.DataFrame({"precip": np.random.rand(len(idx)),
                         "pet": np.random.rand(len(idx))}, index=idx)
    return data


def test_read_demand(tmpdir):
    # Prepare a small CSV demand file
    csv_path = tmpdir.join("demand.csv")
    times = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame({"datetime": times.strftime("%d-%m-%Y %H:%M"),
                       "demand": [1.1, 2.2, 3.3, 4.4, 5.5]})
    df.to_csv(csv_path, index=False)

    demand = Demand(
        root=str(tmpdir),
        demand_fn=str(csv_path),
        forcing_fn=make_dummy_forcing_df()
    )
    assert isinstance(demand.data, pd.DataFrame)
    assert "demand" in demand.data.columns
    assert isinstance(demand.data.index, pd.DatetimeIndex)
    # yearly_demand should be a float
    assert isinstance(demand.yearly_demand, float)


def test_read_demand_from_value(tmpdir):
    # Create a forcing DataFrame to pass in
    forcing_df = make_dummy_forcing_df()
    # Pass a constant demand value
    demand = Demand(
        root=str(tmpdir),
        demand_fn=5.0,
        forcing_fn=forcing_df
    )
    assert "demand" in demand.data.columns
    assert np.allclose(demand.data["demand"], 5.0)


def test_write_demand(tmpdir):
    os.makedirs(os.path.join(tmpdir, "output", "runs"), exist_ok=True)

    csv_path = tmpdir.join("demand.csv")
    times = pd.date_range("2020-01-01", periods=5, freq="h")
    df = pd.DataFrame({"datetime": times.strftime("%d-%m-%Y %H:%M"),
                       "demand": [1, 2, 3, 4, 5]})
    df.to_csv(csv_path, index=False)

    demand = Demand(
        root=str(tmpdir),
        demand_fn=str(csv_path),
        forcing_fn=make_dummy_forcing_df()
    )
    demand.write(fn_out="test_demand")

    expected_path = os.path.join(tmpdir, "output", "runs", "test_demand.csv")
    assert os.path.exists(expected_path)


# ----------------------------
# seasonal variation test


def test_seasonal_variation_shape():
    # simple short time window
    t_start = pd.Timestamp("2020-01-01 00:00")
    t_end = t_start + pd.Timedelta(days=2)
    d = Demand(
        root=".",
        demand_fn=1.0,
        forcing_fn=make_dummy_forcing_df()
    )
    arr = d.seasonal_variation(
        yearly_demand=100.0,
        timestep=3600,
        t_start=t_start,
        t_end=t_end,
        perc_constant=0.5,
        shift=0.0,
    )
    # 3 days * 24 hours = 72 hourly values
    assert len(arr) == 72
    assert all(isinstance(x, float) for x in arr)
