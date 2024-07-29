from typing import Optional
import numpy as np
import pandas as pd

from RWHmodel.reservoir import Reservoir
from RWHmodel.timeseries import ConstantDemand, Demand, Forcing
from RWHmodel.hydro_model import HydroModel


class Model(object):
    # def __init__(self):
    #     pass
        
    
    def __init__( # was first called "setup"
        self,
        root: str,
        name: str,
        forcing_fn: str,
        reservoir_cap: float,
        reservoir_initial_state: float = 0,
        timestep: int = 86400,
        demand_fn: Optional[str] = None,
        independent_demand: bool = True, # toggle demand independent of available water in reservoir
        unit: str = "mm",
        area_chars: Optional[dict] = None,
    ):
        if len(root)>0:
            self.root = root
        else:
            raise ValueError(f"Provide root of model folder")
        if len(name)>0:
            self.name = name
        else:
            raise ValueError(f"Provide model run name")
        self.timestep = timestep
        
        # Setup forcing
        self.forcing = Forcing(forcing_fn, timestep, root)
        
        # Setup demand
        if type(demand_fn) in [float, int]:
            self.demand = ConstantDemand(self.forcing, demand_fn)
        else:
            self.demand = Demand(
                demand_fn, root, timestep, unit=unit, area_chars=area_chars
            )
        self.demand.independent_demand = independent_demand
        
        # Setup of area characteristics
        self.int_cap = 2 #mm - placeholder interception storage capacity (later change to setup from config)
        
        # Initiate hydro_model
        self.hydro_model = HydroModel(self.int_cap) #TODO: later change to fetch from area_chars
        
        # Initiate reservoir
        self.reservoir = Reservoir(reservoir_cap, reservoir_initial_state)
        

    def setup_from_config(self, config_fn):
        # TODO: parse config with code from config.py and call self.setup with arguments from config
        # TODO: import area characteristics here.
        pass



    def run(self):

        ## Initialize numpy arrays
        net_precip = np.array(self.forcing.data["precip"] - self.forcing.data["pet"])
        demand = np.array(self.demand.data["demand"])
        
        reservoir_stor = np.zeros(len(net_precip))
        reservoir_overflow = np.zeros(len(net_precip))
        deficit = np.zeros(len(net_precip))
        dry_days = np.zeros(len(net_precip))
        consec_dry_days = np.zeros(len(net_precip))

        # Run hydro_model per timestep         
        int_stor, runoff = HydroModel.update_state(self, net_precip=net_precip)   
        
        # Fill tank arrays
        for i in range(1, len(net_precip)):
            # reservoir_stor[i] = min(max(0, reservoir_stor[i-1] + runoff[i] - demand[i]), self.reservoir_cap)
            # reservoir_overflow[i] =  max(0, reservoir_stor[i-1] + runoff[i] - demand[i] - self.reservoir_cap)
            # deficit[i] = max(0, demand[i] - reservoir_stor[i])
            if reservoir_stor[i] >= demand[i]:
                dry_days[i] = 0
            elif reservoir_stor[i] < demand[i]:
                dry_days[i] = dry_days[i-1] + 1
            if dry_days[i-1] != 0 and dry_days[i] == 0:
                consec_dry_days[i] = dry_days[i-1]
            else:
                consec_dry_days[i] = 0
        

        #TODO: probably quicker to calculate in np.array, and after calculations transform to df and insert datetime again.
        df = pd.DataFrame("reservoir_stor" : reservoir_stor,
                          "reservoir_overflow" : reservoir_overflow,
                          "deficit" : deficit,
                          "consec_dry_days" : consec_dry_days)
        
        index==self.forcing.data.index
        
        # TODO: add to plot script: Reservoir storage, overflow and deficit against volume (mm/day)
 
    def batch_run(self, method, reservoir_range, demand_range, T_range=[1,2,5,10,20,50,100]):
        # Batch run function to obtain solution space and statistics on output.
        methods = ["total_days", "consecutive_days"]
        if method not in methods:
            raise ValueError(
                f"Provide valid method from {methods}."
            )
        if not isinstance(reservoir_range, list) or not isinstance(demand_range, list) or not isinstance(T_range, list):
            raise ValueError(
                "Provide reservoir_range and demand_range as lists."
            )
        
        if method == "consecutive_days":
            df_system = pd.DataFrame(columns = T_range)
            df_system["reservoir_capacity"] = reservoir_range
            df_system = df_system.set_index("reservoir_capacity")
        
        for reservoir_cap in reservoir_range:
            for demand in demand_range:
                df = self.run() #TODO: how to insert demand here?
                
                
                
                # creating dataframe with events
                events = output_df["max_num_dry"].sort_values(ascending = False).to_frame()
                events = events.rename(columns = {"max_num_dry": yearly_demand})
                events = events.reset_index(drop = True)
                df = pd.concat([df, events], axis = 1)
        
        

