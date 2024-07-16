import pandas as pd

def convert_m3_to_mm(df: pd.DataFrame , col: str , surface_area: float) -> pd.DataFrame:
        df[col] = df[col] / surface_area
        return df 

