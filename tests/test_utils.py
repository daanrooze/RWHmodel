import pandas as pd

from RWHmodel.utils import convert_m3_to_mm


def test_convert_m3_to_mm():
    reservoir = pd.DataFrame({"water": [1, 2, 3]})
    df = convert_m3_to_mm(df=reservoir, col="water", surface_area=0.5)

    assert df.iloc[0].values == 2.0
