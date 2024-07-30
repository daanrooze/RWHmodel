"""Utility functions."""

import os

import pandas as pd


def convert_m3_to_mm(df: pd.DataFrame, col: str, surface_area: float) -> pd.DataFrame:
    """Convert a cubic meters pd.series to milimeters.

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
    df[col] = df[col] / surface_area
    return df


def makedir(path: str) -> None:
    """Create new directory if path does not exist."""
    if not os.path.exists(path):
        print(f"{path} does not exist. Making new path.")
        os.makedirs(path)
