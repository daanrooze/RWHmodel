import pandas as pd
import numpy as np
from os.path import join
from typing import Optional, List, Union

from RWHmodel.utils import convert_m3_to_mm


class TimeSeries:
    file_formats = ["csv"]

    def __init__(self, fn: str, root: str) -> None:
        self.root = root
        self.fn = fn

    def read_timeseries(
        self,
        file_type: str,
        required_headers: List[str],
        numeric_cols: List[str],
        resample: bool = True,
        timestep: Optional[int] = None,
    ) -> pd.DataFrame:
        df = pd.read_csv(self.fn, sep=",")

        if not all(item in df.columns for item in required_headers):
            raise ValueError(
                f"Provide {file_type} file with at least the following headers: {', '.join(required_headers)} "
            )
        for col in numeric_cols:
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"{col} is not numeric")

        df["datetime"] = pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M")
        df = df.set_index("datetime")

        if resample:
            if not timestep:
                raise ValueError("timestep is needed for timeseries resample.")
            df.resample(f"{timestep}s", label="right").sum()
        return df

    def write_timeseries(
        self, df: pd.DataFrame, subdir: str, fn_out: str, file_format: str = "csv"
    ):
        if file_format not in self.file_formats:
            raise ValueError(
                f"Provide supported file format from {', '.join(self.file_formats)}"
            )

        out_path = join(self.root, "output", subdir, f"{fn_out}.{file_format}")

        if file_format == "csv":
            df.to_csv(out_path, sep=",", date_format="%d-%m-%Y %H:%M")


class Forcing(TimeSeries):
    def __init__(
        self,
        forcing_fn: str,
        timestep: Optional[int] = None,
        root: str = "./",
        resample: bool = False,
    ) -> None:
        # Call TimeSeries __init__ with super
        super().__init__(fn=forcing_fn, root=root)
        self.data = self.read_timeseries(
            file_type="csv",
            required_headers=["datetime", "precip", "pet"],
            numeric_cols=["precip", "pet"],
            resample=resample,
            timestep=timestep,
        )

    def statistics(self):
        raise NotImplementedError

    def write(self, fn_out="forcing"):
        self.write_timeseries(df=self.data, subdir="forcing", fn_out=fn_out)


class Demand(TimeSeries):
    def __init__(
        self,
        demand_fn: str,
        root: str,
        timestep: int,
        unit: str = "mm",
        area_chars: Optional[dict] = None
    ):
        #if type(demand_fn)==int:
        #    pass
        #else:
        super().__init__(fn=demand_fn, root=root)
        self.data = self.read_timeseries(
            file_type="csv",
            required_headers=["datetime", "demand"],
            numeric_cols=["demand"],
            resample=True,
            timestep=timestep,
        )

        if unit == "m3":  # Convert to mm
            if surface_area := area_chars.get("srf_area"):
                self.demand = convert_m3_to_mm(
                    df=self.demand, col="demand", surface_area=surface_area
                )
            else:
                raise ValueError("Missing surface area for converting m3 per timestep to mm per timestep")


    def write(self, fn_out):
        self.write_timeseries(df=self.demand, subdir="demand", fn_out=fn_out)


class ConstantDemand: # deprecate, move to Demand class?
    def __init__(
        self,
        timeseries_df,
        constant: Union[int, float]
    ) -> None:
        timeseries_df["demand"] = constant
        self.data = timeseries_df[["demand"]]
