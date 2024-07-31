import os
from typing import Optional
import numpy as np
import pandas as pd
import codecs
import toml

from RWHmodel.reservoir import Reservoir
from RWHmodel.timeseries import ConstantDemand, Demand, Forcing
from RWHmodel.hydro_model import HydroModel
from RWHmodel.utils import makedir
from RWHmodel.analysis import return_period

from RWHmodel.plot import plot_meteo, plot_run, plot_system_curve, plot_saving_curve


class Model(object):
    def __init__(
        self,
        root: str,
        name: str,
        setup_fn: str,
        forcing_fn: str,
        reservoir_initial_state: float = 0,
        timestep: Optional[int] = None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        demand_fn: Optional[str] = None,
        unit: str = "mm",
    ):
        # Setup folder structure
        if len(root)>0:
            self.root = root
        else:
            raise ValueError("Provide root of model folder")
        makedir(f"{self.root}/input")
        makedir(f"{self.root}/output")
        makedir(f"{self.root}/output/figures")
        makedir(f"{self.root}/output/runs")
        makedir(f"{self.root}/output/statistics")
        
        # Setup model run name
        if len(name)>0:
            self.name = name
        else:
            raise ValueError("Provide model run name")
        
        # Setup of area characteristics
        self.setup_from_toml(setup_fn=setup_fn)
        
        # Setup forcing
        self.forcing = Forcing(
            forcing_fn = forcing_fn,
            root = root,
            timestep = timestep,
            t_start = t_start,
            t_end = t_end
        )
        
        # Setup demand
        if type(demand_fn) in [float, int]:
            self.demand = ConstantDemand(
                forcing_fn = self.forcing.data,
                constant = demand_fn
            )
        else:
            self.demand = Demand(
                demand_fn = demand_fn,
                root = root,
                timestep = timestep,
                t_start = t_start,
                t_end = t_end,
                unit = unit, #TODO: test unit conversion
                setup_fn = self.config
            )
        if self.forcing.timestep != self.demand.timestep:
            raise ValueError("Forcing and demand timeseries have different timesteps. Change input files or resample by specifying timestep.")
        if len(self.forcing.data) != len(self.demand.data):
            raise ValueError(f"Forcing and demand timeseries have different starting and/or end dates. Forcing timeseries runs from {self.forcing.t_start} to {self.forcing.t_end}, while demand timeseries runs from {self.demand.t_start} to {self.demand.t_end}. Change input files or clip timeseries by specifying interval.")
        #TODO: add method to automatically clip forcing and demand timeseries to smalles interval?
        
        # Initiate hydro_model
        self.hydro_model = HydroModel(int_cap = self.config['int_cap'])
        
        # Initiate reservoir
        self.reservoir = Reservoir(self.config['reservoir_cap'], reservoir_initial_state)
        
        

    def setup_from_toml(self, setup_fn):
        folder = f"{self.root}/input"
        with codecs.open(os.path.join(folder, setup_fn), "r", encoding="utf-8") as f:
            area_chars = toml.load(f)
        
        self.config = area_chars
        #return area_chars



    def run(
            self,
            demand: Optional = None,
            reservoir_cap: Optional = None,
            save=True
        ):
        # Overwrite self with arguments if given.
        demand = demand if demand is not None else np.array(self.demand.data["demand"])
        reservoir_cap = reservoir_cap if reservoir_cap is not None else self.config["reservoir_cap"]
        
        ## Initialize numpy arrays
        net_precip = np.array(self.forcing.data["precip"] - self.forcing.data["pet"])

        reservoir_stor = np.zeros(len(net_precip))
        reservoir_overflow = np.zeros(len(net_precip))
        deficit = np.zeros(len(net_precip))
        dry_days = np.zeros(len(net_precip))
        consec_dry_days = np.zeros(len(net_precip))
        
        # Run hydro_model per timestep         
        int_stor, runoff = self.hydro_model.calc_runoff(net_precip=net_precip)   
        
        # Fill tank arrays
        for i in range(1, len(net_precip)):
            #reservoir_stor[i] = min(max(0, reservoir_stor[i-1] + runoff[i] - demand[i]), reservoir_cap)
            #reservoir_overflow[i] =  max(0, reservoir_stor[i-1] + runoff[i] - demand[i] - reservoir_cap)
            #deficit[i] = max(0, demand[i] - reservoir_stor[i])
            self.reservoir.update_state(runoff = runoff[i], demand = demand[i])
            reservoir_stor[i] = self.reservoir.reservoir_stor
            reservoir_overflow[i] = self.reservoir.reservoir_overflow
            deficit[i] = self.reservoir.deficit
            
            # Calculating days that reservoir does not suffice
            if reservoir_stor[i] >= demand[i]:
                dry_days[i] = 0
            elif reservoir_stor[i] < demand[i]:
                dry_days[i] = dry_days[i-1] + 1
            if dry_days[i-1] != 0 and dry_days[i] == 0:
                consec_dry_days[i] = dry_days[i-1]
            else:
                consec_dry_days[i] = 0
        
        # Convert to dataframe
        df_data = {
            'reservoir_stor': reservoir_stor,
            'reservoir_overflow': reservoir_overflow,
            'deficit': deficit,
            'consec_dry_days': consec_dry_days}
        df = pd.DataFrame(df_data, index = self.forcing.data['precip'].index)
        if save==True:
            self.results = df
            df.to_csv(f"{self.root}/output/runs/single_run_{reservoir_cap}.csv") #TODO: change output name
        return df

 
    def batch_run(self, method):
        # Batch run function to obtain solution space and statistics on output.
        methods = ["total_days", "consecutive_days"]
        # Check if input is correct
        if method not in methods:
            raise ValueError(
                f"Provide valid method from {methods}."
            )
        # Define parameters
        dem_min = self.config["dem_min"]
        dem_max = self.config["dem_max"]
        dem_step = self.config["dem_step"]
        demand_lst = list(range(dem_min, dem_max + 1, dem_step))
        
        cap_min = self.config["cap_min"]
        cap_max = self.config["cap_max"]
        cap_step = self.config["cap_step"]
        capacity_lst = list(range(cap_min, cap_max + 1, cap_step))
        
        T_range = self.config["T_return_list"]
        max_num_days = self.config["max_num_days"]

        df_system = pd.DataFrame(columns = T_range) 
        for reservoir_cap in capacity_lst:
            df_total = pd.DataFrame()
            dry_events = pd.DataFrame()
            req_storage = pd.DataFrame()
            for demand in demand_lst:
                run_df = self.run(demand = np.full(len(self.forcing.data["precip"]), demand), reservoir_cap = reservoir_cap, save = False)
                if method == "consecutive_days": 
                    dry_events = run_df["consec_dry_days"].sort_values(ascending = False).to_frame()
                    dry_events = dry_events.reset_index(drop = True)
                    dry_events = dry_events.rename(columns={'consec_dry_days': f'{demand}'})
                if method == "total_days": 
                    run_df_yearly = run_df.resample('YE').sum()
                    dry_events = run_df_yearly["consec_dry_days"].sort_values(ascending = False).to_frame()
                    dry_events = dry_events.reset_index(drop = True)
                    dry_events = dry_events.rename(columns={'consec_dry_days': f'{demand}'})
                df_total = pd.concat([df_total, dry_events[[f'{demand}']]], axis=1)
            df_total['T_return'] = self.forcing.num_years / (df_total.index + 1)
            req_storage = return_period(df_total)
            # Find optimal demand for specific tank size
            opt_demand_lst = []
            for column in req_storage.columns:
                try:
                    # Filter rows where the value in the column is less than or equal to max_num_days
                    boundary_condition = req_storage[req_storage[column] <= max_num_days].index
                    # Check if there are any indices that meet the condition
                    if not boundary_condition.empty:
                        # Get the last index from the filtered results
                        opt_demand = int(boundary_condition[-1])
                        opt_demand_lst.append(opt_demand)
                    else:
                        # Append 0 if no valid index was found
                        opt_demand_lst.append(0)
                except Exception as e:
                    opt_demand_lst.append(0)

            opt_demand_df = pd.DataFrame([opt_demand_lst], columns = T_range)
            df_system = pd.concat([df_system, opt_demand_df])
        df_system["tank_size"] = capacity_lst
        df_system = df_system.set_index("tank_size")
        self.results = df_system
        df_system.to_csv(f"{self.root}/output/runs/batch_run.csv") #TODO: change save name
    
    def plot(
        self,
        plot_type=None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        fn: Optional[str] = None,
        **kwargs
    ):
        plot_types = ["meteo", "run", "system_curve", "saving_curve"]
        if plot_type not in plot_types:
            raise ValueError(
                f"Provide valid plot type from {plot_types}."
            )
        # Overwrite self with arguments if given.
        t_start = t_start if t_start is not None else self.forcing.t_start
        t_end = t_end if t_end is not None else self.forcing.t_end
        #if t_start:
        #    t_start = pd.to_datetime(t_start)
        #else:
        #    t_start = self.t_start
        #if t_end:
        #    t_end = pd.to_datetime(t_end)
        #else:
        #    t_end = self.t_end
        
        
        
        if plot_type == "meteo":
            plot_meteo(
                self.root,
                self.name,
                self.forcing.data,
                t_start,
                t_end,
                aggregate = False
            )
        
        if plot_type == "run":
            if fn:
                run_fn = pd.read_csv(fn, sep=',')
            else:
                run_fn = self.results
            plot_run(
                self.root,
                self.name,
                run_fn,
                t_start,
                t_end,
                self.reservoir.reservoir_cap,
                self.demand.yearly_demand
            )
        
        if plot_type == "plot_system_curve":
            if fn:
                system_fn = pd.read_csv(fn, sep=',')
            else:
                system_fn = self.results
            plot_system_curve(
                self.root,
                self.name,
                system_fn,
                t_start,
                t_end,
                self.reservoir.reservoir_cap,
                self.demand.yearly_demand
            )
        
        if plot_type == "plot_saving_curve":
            pass

        
        

