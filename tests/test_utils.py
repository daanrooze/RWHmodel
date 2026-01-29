import os
import pytest
import pandas as pd
import shutil

from RWHmodel import (  # replace your_module with actual module name
    convert_m3_to_mm,
    convert_mm_to_m3,
    makedir,
    check_variables,
    colloquial_date_text,
)

def test_convert_m3_to_mm():
    df = pd.DataFrame({'vol_m3': [1.0, 2.0, 3.0]})
    surface_area = 10.0
    result = convert_m3_to_mm(df.copy(), 'vol_m3', surface_area)
    expected = [(v / surface_area) * 1000 for v in [1.0, 2.0, 3.0]]
    assert all(result['vol_m3'].round(6) == pd.Series(expected).round(6))

def test_convert_mm_to_m3():
    df = pd.DataFrame({'depth_mm': [1000, 2000, 3000]})
    surface_area = 10.0
    result = convert_mm_to_m3(df.copy(), 'depth_mm', surface_area)
    expected = [(v / 1000) * surface_area for v in [1000, 2000, 3000]]
    assert all(result['depth_mm'].round(6) == pd.Series(expected).round(6))

def test_makedir_creates_directory(tmp_path):
    test_dir = tmp_path / "new_folder"
    # Directory should not exist initially
    assert not test_dir.exists()
    makedir(str(test_dir))
    assert test_dir.exists()
    assert test_dir.is_dir()

def test_makedir_existing_directory(tmp_path):
    test_dir = tmp_path / "existing_folder"
    test_dir.mkdir()
    # Should not raise or print error
    makedir(str(test_dir))
    assert test_dir.exists()

@pytest.mark.parametrize("mode,config,demand_transformation,should_raise,missing_vars", [
    ("single", {}, False, True, ['connected_srf_area', 'int_cap', 'reservoir_cap', 'reservoir_type']),
    ("single", {'connected_srf_area':1, 'int_cap':1, 'reservoir_cap':1, 'reservoir_type':'closed'}, False, False, []),
    ("single", {'connected_srf_area':1, 'int_cap':1, 'reservoir_type':'closed'}, True, True, ['reservoir_cap', 'shift', 'perc_constant']),
    ("batch", {'connected_srf_area':1, 'int_cap':1, 'reservoir_type':'closed', 'threshold':1, 'T_return_list':[1], 'shift':1, 'perc_constant':1}, False, False, []),
    ("batch", {'connected_srf_area':1, 'int_cap':1, 'reservoir_type':'closed', 'threshold':1, 'T_return_list':[1]}, True, True, ['shift', 'perc_constant']),
])
def test_check_variables(mode, config, demand_transformation, should_raise, missing_vars):
    if should_raise:
        with pytest.raises(ValueError) as e:
            check_variables(mode, config, demand_transformation)
        for var in missing_vars:
            assert var in str(e.value)
    else:
        # Should not raise
        check_variables(mode, config, demand_transformation)

@pytest.mark.parametrize("timestep,expected", [
    (365*24*3600, 'year'),
    (365*24*3600 + 1, 'year'),
    (24*3600, 'day'),
    (24*3600 + 1, 'day'),
    (3600, 'hour'),
    (3600 + 1, 'hour'),
])
def test_colloquial_date_text(timestep, expected):
    assert colloquial_date_text(timestep) == expected