import pandas as pd

from RWHmodel.utils import convert_m3_to_mm


def test_convert_m3_to_mm():
    reservoir = pd.DataFrame({"value": [1]})
    df = convert_m3_to_mm(df=reservoir, col="value", surface_area=1)

    assert df.iloc[0].values == 1000