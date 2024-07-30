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

from RWHmodel.plot import plot_meteo, plot_run, plot_system_curve, plot_saving_curve


class Model(object):
    def __init__( # was first called "setup"
        self,
        root: str,
        name: str,
        setup_fn: str,
        forcing_fn: str,
        reservoir_cap: float,
        reservoir_initial_state: float = 0,
        timestep: int = 86400,
        demand_fn: Optional[str] = None,
        independent_demand: bool = True, # toggle demand independent of available water in reservoir
        unit: str = "mm",
    ):
        # Setup folder structure
        if len(root)>0:
            self.root = root
        else:
            raise ValueError(f"Provide root of model folder")
        makedir("root/input")
        makedir("root/output")
        makedir("root/output/figures")
        makedir("root/output/runs")
        makedir("root/output/statistics")
        
        # Setup model run name
        if len(name)>0:
            self.name = name
        else:
            raise ValueError(f"Provide model run name")
        self.timestep = timestep
        
        # Setup of area characteristics
        self.int_cap = 2 #mm - placeholder interception storage capacity (later change to setup from config)
        self.setup_from_toml(setup_fn=setup_fn)
        
        # Setup forcing
        self.forcing = Forcing(forcing_fn, timestep, root)
        self.t_start = self.forcing.data.index.min() #TODO: add manual override of self.t_start when setting up (if interval needs to be excluded)
        self.t_end = self.forcing.data.index.max()
        
        # Setup demand
        if type(demand_fn) in [float, int]:
            self.demand = ConstantDemand(self.forcing, demand_fn)
        else:
            self.demand = Demand(
                demand_fn, root, timestep, unit=unit, setup_fn=setup_fn
            )
        self.demand.independent_demand = independent_demand
        
        # Initiate hydro_model
        self.hydro_model = HydroModel(self.int_cap) #TODO: later change to fetch from area_chars
        
        # Initiate reservoir
        self.reservoir = Reservoir(reservoir_cap, reservoir_initial_state)
        
        

    def setup_from_toml(self, setup_fn):
        # TODO: parse config with code from config.py and call self.setup with arguments from config
        # TODO: import area characteristics here.
        folder = f"{self.root}/input"
        with codecs.open(os.path.join(folder, setup_fn), "r", encoding="utf-8") as f:
            area_chars = toml.load(f)
        
        self.config = area_chars
        #return area_chars



    def run(self, demand, reservoir_cap, save=True):
 
        ## Initialize numpy arrays
        net_precip = np.array(self.forcing.data["precip"] - self.forcing.data["pet"])
        demand = np.array(self.demand.data["demand"])
        reservoir_cap = 100
       # reservoir_cap = self.area_chars.reservoir_cap

        reservoir_stor = np.zeros(len(net_precip))
        reservoir_overflow = np.zeros(len(net_precip))
        deficit = np.zeros(len(net_precip))
        dry_days = np.zeros(len(net_precip))
        consec_dry_days = np.zeros(len(net_precip))
 
        # Run hydro_model per timestep         
        int_stor, runoff = HydroModel.update_state(self, net_precip=net_precip)   
        # Fill tank arrays
        for i in range(1, len(net_precip)):
            reservoir_stor[i] = min(max(0, reservoir_stor[i-1] + runoff[i] - demand[i]), reservoir_cap)
            reservoir_overflow[i] =  max(0, reservoir_stor[i-1] + runoff[i] - demand[i] - reservoir_cap)
            deficit[i] = max(0, demand[i] - reservoir_stor[i])
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
            df.to_csv(f"root/output/runs/single_run_{reservoir_cap}.csv")
        return df

        # TODO: add to plot script: Reservoir storage, overflow and deficit against volume (mm/day)
 

    def batch_run(self, method):
        # Batch run function to obtain solution space and statistics on output.
        methods = ["total_days", "consecutive_days"]
        # Check if input is correct
        if method not in methods:
            raise ValueError(
                f"Provide valid method from {methods}."
            )
 
        # Define parameters
        # dem_min = self.area_chars.dem_min
        # dem_max = self.area_chars.dem_max
        # dem_step = self.area_chars.dem_step
        dem_min = 2
        dem_max = 10
        dem_step = 2
        demand_lst = list(range(dem_min, dem_max + 1, dem_step))
        # cap_min = self.area_chars.cap_min
        # cap_max = self.area_chars.cap_max
        # cap_step = self.area_chars.cap_step
        cap_min = 2
        cap_max = 10
        cap_step = 2
        capacity_lst = list(range(cap_min, cap_max + 1, cap_step))
        # T_range = self.area_chars.T_return_list
        # max_num_days = self.area_chars.max_num_days
        T_range = [1,2,5,10,20,50,100]
        max_mun_days = 7        
        #Make arrays
        df = pd.DataFrame()
        if method == "consecutive_days":
            for reservoir_cap in capacity_lst:
                for demand in demand_lst:
                    df[demand] = self.run(demand = demand, reservoir_cap = reservoir_cap, save = False)



            # df_system = pd.DataFrame(columns = T_range)
            # df_system["reservoir_capacity"] = reservoir_range
            # df_system = df_system.set_index("reservoir_capacity")

            #     # creating dataframe with events
            #     events = output_df["max_num_dry"].sort_values(ascending = False).to_frame()
            #     events = events.rename(columns = {"max_num_dry": yearly_demand})
            #     events = events.reset_index(drop = True)
            #     df = pd.concat([df, events], axis = 1)
    
    
    def plot(
        self,
        plot_type=None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
    ):
        plot_types = ["meteo", "run", "system_curve", "saving_curve"]
        if plot_type not in plot_types:
            raise ValueError(
                f"Provide valid plot type from {plot_types}."
            )
        if t_start:
            t_start = pd.to_datetime(t_start)
        else:
            t_start = self.t_start
        if t_end:
            t_end = pd.to_datetime(t_end)
        else:
            t_end = self.t_end
        
        if plot_type == "meteo":
            plot_meteo(
                self.root,
                self.name,
                self.forcing.data,
                t_start,
                t_end,
                aggregate = False
            )
        
        #plot_run()
        
        

