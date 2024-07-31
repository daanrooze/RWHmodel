import os

import pandas as pd

from RWHmodel.timeseries import Forcing


def test_read_forcing(tmpdir):
    forcing = Forcing(forcing_fn="tests/data/forcing_test.csv", root=tmpdir)
    assert isinstance(forcing.data, pd.DataFrame)
    assert all([col in forcing.data.columns for col in ["precip", "pet"]])
    assert isinstance(forcing.data.index, pd.DatetimeIndex)


def test_write_forcing(tmpdir):
    forcing = Forcing(forcing_fn="tests/data/forcing_test.csv", root=tmpdir)
    fp = forcing.write(fn_out="test_forcing.csv")
    assert os.path.exists(fp)


def test_read_demand(tmpdir):
    pass


def test_write_demand(tmpdir):
    pass
