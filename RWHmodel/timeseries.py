import os
from os.path import join
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from RWHmodel.utils import convert_m3_to_mm


class TimeSeries:
    file_formats = ["csv"]

    def __init__(
        self,
        fn: str,
        root: str,
        timestep: Optional[int] = None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
    ) -> None:
        self.root = root
        self.fn = fn
        self.t_start = t_start
        self.t_end = t_end
        self.timestep = timestep

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

        # Overwrite self with arguments if given.
        self.timestep = self.timestep if self.timestep is not None else int((df.index[1] - df.index[0]).total_seconds())
        self.t_start = pd.to_datetime(self.t_start) if self.t_start is not None else df.index.min()
        self.t_end = pd.to_datetime(self.t_end) if self.t_end is not None else df.index.max()
        
        self.num_years = (self.t_end - self.t_start) / (np.timedelta64(1, "W") * 52)

        # Resample if timestep is not same as df_datetime (already to self). If not given, set self.timestep based on provided forcing input
        if self.timestep != int((df.index[1] - df.index[0]).total_seconds()):
            df = df.resample(f"{timestep}s", label="right").sum()
        #else:
        #    self.timestep = int((df.index[1] - df.index[0]).total_seconds())
        
        # Mask timeseries based on t_start and t_end
        mask = (df.index >= self.t_start) & (df.index <= self.t_end)
        df = df.loc[mask]
        
        # Transform data type to float
        df[df.columns[0]] = df[df.columns[0]].astype('float')
        
        return df

    def write_timeseries(
        self, df: pd.DataFrame,
        subdir: str,
        fn_out: str,
        file_format: str = "csv"
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
        # resample: bool = False,
    ) -> None:
        # Call TimeSeries __init__ with super
        super().__init__(
            fn=forcing_fn, root=root, timestep=timestep, t_start=t_start, t_end=t_end
        )
        self.data = self.read_timeseries(
            file_type="csv",
            required_headers=["datetime", "precip", "pet"],
            numeric_cols=["precip", "pet"],
            # resample=resample,
            timestep=timestep,
        )

    def statistics(self):
        raise NotImplementedError

    def write(self, fn_out="forcing"):
        return self.write_timeseries(df=self.data, subdir="forcing", fn_out=fn_out)


class Demand(TimeSeries):
    def __init__(
        self,
        root: str,
        demand_fn: str,
        forcing_fn: str,
        demand_transform: Optional[bool] = False,
        timestep: Optional[int] = None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        unit: str = "mm",
        setup_fn: Optional[dict] = None,
        perc_constant: Optional[int] = None,
        shift: Optional[int] = None
    ):
        super().__init__(
            fn=demand_fn, root=root, timestep=timestep, t_start=t_start, t_end=t_end
        )
        
        # Check if seasonal transformation can be applied
        if type(demand_fn)==str and demand_transform==True:
                    raise ValueError(
                        "Can only apply generic sinusoid seasonal variation to singular yearly demand figures."
        )
        # Set self.transform if demand timeseries transformation is True
        demand_transform
        if demand_transform:
            self.perc_constant = perc_constant
            self.shift = shift
        
        if isinstance(demand_fn, str): # Call read_timeseries function if given a timeseries file.
            self.data = self.read_timeseries(
                file_type="csv",
                required_headers=["datetime", "demand"],
                numeric_cols=["demand"],
                timestep=timestep,
            )
        
        if isinstance(demand_fn, (int, float, list)): # Use forcing timeseries to fill demand timeseries
            forcing_fn["demand"] = float(demand_fn[0]) if isinstance(demand_fn, list) else float(demand_fn)
            self.data = forcing_fn[["demand"]]
            self.fn = list if isinstance(demand_fn, list) else float
            self.num_years = (max(forcing_fn.index) - min(forcing_fn.index)) / (np.timedelta64(1, "W") * 52)
            self.timestep = int((self.data.index[1] - self.data.index[0]).total_seconds())
            self.t_start = pd.to_datetime(self.t_start) if self.t_start is not None else forcing_fn.index.min()
            self.t_end = pd.to_datetime(self.t_end) if self.t_end is not None else forcing_fn.index.max()
        
        if unit == "m3":  # Convert to mm
            if surface_area := setup_fn["srf_area"]:
                self.data = convert_m3_to_mm(
                    df=self.data, col="demand", surface_area=surface_area
                )
            else:
                raise ValueError(
                    "Missing surface area for converting m3 per timestep to mm per timestep"
                )
        self.yearly_demand = np.round(float((self.data["demand"].sum())/self.num_years), 1)
        
        if demand_transform:
            timeseries_transformed = self.seasonal_variation(
                yearly_demand = self.yearly_demand,
                perc_constant = self.perc_constant,
                shift= self.shift,
                timestep = self.timestep,
                t_start = self.t_start,
                t_end = self.t_end
            )
            self.data.loc[:, "demand"] = timeseries_transformed[:len(self.data)]
    
    def update_demand(
        self,
        update_data,
        demand_transform: Optional[bool] = False
    ) -> pd.DataFrame:
        """
        Update the demand data in the object.
        
        Args:
            update_data (pd.Series or pd.DataFrame): New demand data to replace the current demand.
        """
        if isinstance(update_data, (int, float)):
            self.data.loc[:, "demand"] = update_data
        if isinstance(update_data, str):
            self.data = self.read_timeseries(
                file_type="csv",
                required_headers=["datetime", "demand"],
                numeric_cols=["demand"],
                timestep=self.timestep,
            )

        # Recalculate yearly_demand
        self.yearly_demand = np.round(float((self.data["demand"].sum()) / self.num_years), 1)
        
        # Reapply seasonal variation if transformation is enabled
        if demand_transform:
            timeseries_transformed = self.seasonal_variation(
                yearly_demand=self.yearly_demand,
                perc_constant=self.perc_constant,
                shift=self.shift,
                timestep = self.timestep,
                t_start=self.t_start,
                t_end=self.t_end
            )
            self.data.loc[:, "demand"] = timeseries_transformed[:len(self.data.loc[:, "demand"])]
    
    def seasonal_variation(
            self, 
            yearly_demand,  
            perc_constant,
            shift,
            timestep,
            t_start,
            t_end
        ):
        # Insert function with sinus to implement seasonal variation.
        yearly_demand_constant = perc_constant * yearly_demand
        start_day_of_year = t_start.day_of_year  # Start day of the year
        total_days = (t_end - t_start).days  # Total number of days in the time range
        
        # transform yearly demand into daily demand.
        daily_demand_constant = yearly_demand_constant / 365
        
        A = -(((2*np.pi)/365) * (365 * daily_demand_constant - yearly_demand)) / (- np.cos(shift + 365 * ((2*np.pi)/365)) + np.cos(shift) + 365 * ((2*np.pi)/365))

        demand_array = []
        for t in np.arange(start_day_of_year - 1, start_day_of_year + total_days):
            daily_tot = A * np.sin(((2*np.pi)/365)*t+shift) + daily_demand_constant + A
            
            # If timestep is not 86400 seconds (i.e., not a full day), spread the daily value over smaller timesteps
            if timestep != 86400:
                # Calculate how many timesteps fit in a day
                steps_per_day = 86400 // timestep  # Number of timesteps per day
                # Repeat the daily value for each timestep within the day
                for _ in range(steps_per_day):
                    demand_array.append((daily_tot / (86400 / timestep)))
            
            else:
                demand_array.append(daily_tot)

        return demand_array
    
    def write(self, fn_out):
        self.write_timeseries(df=self.demand, subdir="demand", fn_out=fn_out)


