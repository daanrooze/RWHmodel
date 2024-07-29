import pandas as pd
import os

def convert_m3_to_mm(df: pd.DataFrame , col: str , surface_area: float) -> pd.DataFrame:
        df[col] = df[col] / surface_area
        return df 

def makedir(path):
    """Create new directory if path does not exist"""
    if not os.path.exists(path):
        print(f"{path} does not exist. Making new path.")
        os.makedirs(path)