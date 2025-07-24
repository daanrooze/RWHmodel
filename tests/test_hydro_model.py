import numpy as np

from RWHmodel import HydroModel  # replace 'your_module' with the actual module name


def test_calc_runoff_positive_precip():
    """Test runoff and storage behavior with positive precipitation only."""
    model = HydroModel(int_cap=2)
    net_precip = np.array([0, 1, 2, 1, 3])  # all positive
    int_stor, runoff = model.calc_runoff(net_precip)

    # expected behavior: storage fills up to 2, extra becomes runoff
    expected_int_stor = np.array([0, 1, 2, 2, 2])
    expected_runoff = np.array([0, 0, 1, 1, 3])

    np.testing.assert_allclose(int_stor, expected_int_stor, err_msg="int_stor values incorrect for positive precip")
    np.testing.assert_allclose(runoff, expected_runoff, err_msg="runoff values incorrect for positive precip")

    # assert storage never exceeds capacity
    assert (int_stor <= 2).all(), "int_stor should never exceed capacity"


def test_calc_runoff_with_negative_precip():
    """Test that negative net_precip (evaporation) reduces storage but never below zero."""
    model = HydroModel(int_cap=3)
    net_precip = np.array([0, 2, -1, -5, 1])  # includes negative values
    int_stor, runoff = model.calc_runoff(net_precip)

    # step by step expectations:
    # t=0: start 0
    # t=1: +2 storage = 2
    # t=2: evap -1 => storage = 1
    # t=3: evap -5 => storage = 0 (never below zero)
    # t=4: +1 storage = 1
    expected_int_stor = np.array([0, 2, 1, 0, 1])
    expected_runoff = np.array([0, 0, 0, 0, 0])  # evaporation never causes runoff

    np.testing.assert_allclose(int_stor, expected_int_stor, err_msg="int_stor values incorrect with negative precip")
    np.testing.assert_allclose(runoff, expected_runoff, err_msg="runoff values incorrect with negative precip")

    # check storage constraints
    assert (int_stor >= 0).all(), "int_stor should never be negative"
    assert (int_stor <= 3).all(), "int_stor should never exceed capacity"


def test_calc_runoff_mixed_behavior():
    """Test a mixed case with positive and negative net_precip, ensuring storage stays valid."""
    model = HydroModel(int_cap=1.5)
    net_precip = np.array([0, 1, 1, -0.5, 2])
    int_stor, runoff = model.calc_runoff(net_precip)

    # just validate constraints rather than exact numbers:
    assert (int_stor >= 0).all(), "Storage must never go negative"
    assert (int_stor <= 1.5).all(), "Storage must not exceed capacity"
    assert (runoff >= 0).all(), "Runoff must never be negative"

    # also test shape matching
    assert int_stor.shape == net_precip.shape
    assert runoff.shape == net_precip.shape
