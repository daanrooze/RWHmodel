import pandas as pd
import numpy as np

from RWHmodel.analysis import return_period, func_system_curve, func_system_curve_inv, func_fitting


def test_return_period_shape_and_type():
    df = pd.DataFrame({
        "1": [10, 8, 6, 4, 2],
        "T_return": [1, 2, 3, 4, 5]
    })
    result = return_period(df, T_return_list=[1, 2, 3])
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 3  # Three T_return values


def test_return_period_index_mapping():
    df = pd.DataFrame({
        "1": [10, 8, 6, 4, 2],
        "T_return": [1, 2, 3, 4, 5]
    })
    result = return_period(df, T_return_list=[1, 2, 3])
    assert 1.0 in result.index or "1" in result.index.astype(str)


def test_return_period_column_match():
    df = pd.DataFrame({
        "1": [10, 8, 6, 4, 2],
        "T_return": [1, 2, 3, 4, 5]
    })
    T_list = [1, 2, 3]
    result = return_period(df, T_return_list=T_list)
    assert list(result.columns) == T_list


def test_return_period_values_numeric():
    df = pd.DataFrame({
        "1": [10, 8, 6, 4, 2],
        "T_return": [1, 2, 3, 4, 5]
    })
    result = return_period(df, T_return_list=[1, 2, 3])
    assert np.issubdtype(result.values.dtype, np.number)


def test_return_period_positive_values():
    df = pd.DataFrame({
        "1": [10, 8, 6, 4, 2],
        "T_return": [1, 2, 3, 4, 5]
    })
    result = return_period(df, T_return_list=[1, 2, 3])
    assert (result.values >= 0).all()


def test_func_system_curve_roundtrip():
    a, b, n = 10, 2, 1.5
    x = 5
    y = func_system_curve(x, a, b, n)
    x_back = func_system_curve_inv(y, a, b, n)
    assert np.isclose(x_back, x, rtol=1e-6)


def test_func_system_curve_monotonic():
    a, b, n = 10, 2, 1.5
    xs = np.linspace(0.1, 10, 50)
    ys = func_system_curve(xs, a, b, n)
    assert np.all(np.diff(ys) >= 0)


def test_func_fitting_structure():
    # Create a fake system DataFrame
    capacities = np.linspace(1, 10, 10)
    df = pd.DataFrame({
        'reservoir_cap': capacities,
        '1yr': func_system_curve(capacities, 10, 2, 1.2),
        '5yr': func_system_curve(capacities, 12, 3, 1.4),
        '10yr': func_system_curve(capacities, 14, 4, 1.6),
    })
    result = func_fitting(df)
    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {'1yr', '5yr', '10yr'}
    assert list(result.columns) == ['a', 'b', 'n']


def test_func_fitting_values_are_finite():
    capacities = np.linspace(1, 10, 10)
    df = pd.DataFrame({
        'reservoir_cap': capacities,
        '1yr': func_system_curve(capacities, 10, 2, 1.2),
        '5yr': func_system_curve(capacities, 12, 3, 1.4),
        '10yr': func_system_curve(capacities, 14, 4, 1.6),
    })
    result = func_fitting(df)
    assert np.isfinite(result.values).all()
