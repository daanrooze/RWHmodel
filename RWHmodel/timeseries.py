import os
from os.path import join
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from RWHmodel.utils import convert_m3_to_mm

# TODO: implement option to clip dataframe based on time interval (t_start, t_end)


class TimeSeries:
    file_formats = ["csv"]

    def __init__(
        self,
        fn: str,
        root: str,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
    ) -> None:
        self.root = root
        self.fn = fn
        self.t_start = t_start
        self.t_end = t_end

    def read_timeseries(
        self,
        file_type: str,
        required_headers: List[str],
        numeric_cols: List[str],
        resample: bool = True,
        timestep: Optional[int] = None,
        # t_start: Optional[str] = None,
        # t_end: Optional[str] = None
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

        if self.t_start:
            self.t_start = pd.to_datetime(self.t_start)
        else:
            self.t_start = df.index.min()
        if self.t_end:
            self.t_end = pd.to_datetime(self.t_end)
        else:
            self.t_end = df.index.max()
        self.num_years = (self.t_end - self.t_start) / (np.timedelta64(1, "W") * 52)
        mask = (df.index > self.t_start) & (df.index <= self.t_end)
        df = df.loc[mask]

        if resample:
            if not timestep:
                raise ValueError("timestep is needed for timeseries resample.")
            df.resample(f"{timestep}s", label="right").sum()
        return df

    def write_timeseries(
        self, df: pd.DataFrame, subdir: str, fn_out: str, file_format: str = "csv"
    ) -> str:
        if file_format not in self.file_formats:
            raise ValueError(
                f"Provide supported file format from {', '.join(self.file_formats)}"
            )

        out_dir = join(self.root, "output", subdir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = join(out_dir, f"{fn_out}.{file_format}")

        if file_format == "csv":
            df.to_csv(out_path, sep=",", date_format="%d-%m-%Y %H:%M")
        else:
            raise ValueError("Only allowed to write csv files")
        return out_path


class Forcing(TimeSeries):
    def __init__(
        self,
        forcing_fn: str,
        root: str,
        timestep: Optional[int] = None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        resample: bool = False,
    ) -> None:
        # Call TimeSeries __init__ with super
        super().__init__(fn=forcing_fn, root=root, t_start=t_start, t_end=t_end)
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
        return self.write_timeseries(df=self.data, subdir="forcing", fn_out=fn_out)


class Demand(TimeSeries):
    def __init__(
        self,
        demand_fn: str,
        root: str,
        timestep: int,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        resample: bool = False,
        unit: str = "mm",
        setup_fn: Optional[dict] = None,
    ):
        super().__init__(fn=demand_fn, root=root, t_start=t_start, t_end=t_end)
        # if type(demand_fn)==int:
        #    pass
        # else:
        self.data = self.read_timeseries(
            file_type="csv",
            required_headers=["datetime", "demand"],
            numeric_cols=["demand"],
            resample=resample,
            timestep=timestep,
        )

        if unit == "m3":  # Convert to mm
            if surface_area := setup_fn.get("srf_area"):
                self.demand = convert_m3_to_mm(
                    df=self.demand, col="demand", surface_area=surface_area
                )
            else:
                raise ValueError(
                    "Missing surface area for converting m3 per timestep to mm per timestep"
                )

    def write(self, fn_out):
        self.write_timeseries(df=self.demand, subdir="demand", fn_out=fn_out)


class ConstantDemand:
    def __init__(
        self,
        forcing_fn,  # take forcing_fn as template
        constant: Union[int, float],
    ) -> None:
        forcing_fn["demand"] = constant
        self.data = forcing_fn[["demand"]]
