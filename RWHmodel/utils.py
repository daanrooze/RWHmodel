""" Utility functions. """

import os
import pandas as pd


def convert_m3_to_mm(
        df: pd.DataFrame,
        col: str,
        surface_area: float
    ) -> pd.DataFrame:
    """
    Convert a pd.series with unit cubic meters per timestep
    to milimeters per timestep.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        name of column in cubic meters
    surface_area : float
        surface area in meters squared

    Returns
    -------
    pd.DataFrame
    """
    df.loc[:, col] = (df.loc[:, col] / surface_area) * 1000
    return df

def convert_mm_to_m3(
        df: pd.DataFrame,
        col: str,
        surface_area: float
    ) -> pd.DataFrame:
    """
    Convert a pd.series with unit milimeters per timestep
    to cubic meters per timestep.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        name of column in cubic meters
    surface_area : float
        surface area in meters squared

    Returns
    -------
    pd.DataFrame
    """
    df.loc[:, col] = (df.loc[:, col] / 1000) * surface_area
    return df

def makedir(
        path: str
    ) -> None:
    """
    Create new directory if path does not exist.
    
    Parameters
    ----------
    path : str
        Path to input and output folders.
    """
    if not os.path.exists(path):
        print(f"{path} does not exist. Making new path.")
        os.makedirs(path)

def check_variables(
        mode: str,
        config: dict,
        demand_transformation: bool
    ) -> None:
    # List of required variables
    if mode == "single":
        variables = [
            'srf_area', 'int_cap', 'reservoir_cap', 'shift', 'perc_constant'
        ]
    if mode == "batch":
        variables = [
            'srf_area', 'int_cap', 'threshold', 'T_return_list', 'shift', 'perc_constant'
        ]

    # List to store missing variables
    missing_vars = []

    # Check the common variables
    for var in variables:
        if var in ['shift', 'perc_constant']:
            # Skip these two variables for now, check them later
            continue
        if var not in config:
            missing_vars.append(var)

    # Check 'shift' and 'perc_constant' if 'demand_transformation' is True
    if demand_transformation:
        if 'shift' not in config:
            missing_vars.append('shift')
        if 'perc_constant' not in config:
            missing_vars.append('perc_constant')

    # Raise ValueError if there are any missing variables
    if missing_vars:
        missing_vars_str = ', '.join(missing_vars)
        raise ValueError(f"The following variables are missing in config: {missing_vars_str}")

def colloquial_date_text(
        timestep: int
    ) -> None:
    if timestep >= 365 * 24 * 3600:
        timestep_txt = 'year'
    elif timestep >= 24 * 3600:
        timestep_txt = 'day'
    elif timestep >= 3600:
        timestep_txt = 'hour'
    return timestep_txt