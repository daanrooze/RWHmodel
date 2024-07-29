from typing import Optional
import numpy as np
import pandas as pd

from RWHmodel.reservoir import Reservoir
from RWHmodel.timeseries import ConstantDemand, Demand, Forcing
from RWHmodel.hydro_model import HydroModel


class Model(object):
    #def __init__(self):
    #    pass
        
    
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
        # calc net precipitation flux
        self.forcing.data["net_precip"] = self.forcing.data["precip"] - self.forcing.data["pet"]
        # Initializing dataframe for calculations
        df = self.forcing.data
        # Adding desired demand to df
        df["demand"] = self.demand.data
        # Adding empty columns for to be calulcated factors:
        df['int_stor'] = np.nan            # interception storage
        df['runoff'] = np.nan              # runoff
        df['reservoir_stor'] = np.nan      # storage in reservoir
        df['reservoir_overflow'] = np.nan  # reservoir overflow
        df['deficit'] = np.nan             # deficit: original demand that cannot be met


        # Other way, using numpy arrays
        net_precip = np.array(self.forcing.data["precip"] - self.forcing.data["pet"])
        #flux = net_precip - self.demand.data 
        ### initialize numpy arrays
        int_stor = np.zeros(self.forcing.data["net_precip"].shape[0])
        runoff = np.zeros(self.forcing.data["net_precip"].shape[0])
        reservoir_stor = np.zeros(self.forcing.data["net_precip"].shape[0])
        reservoir_overflow = np.zeros(self.forcing.data["net_precip"].shape[0])
        deficit = np.zeros(self.forcing.data["net_precip"].shape[0])

        
        lst = [
            {
                "int_stor": 0,
                "runoff": np.nan,
                "reservoir_stor": 0,
                "reservoir_overflow": np.nan,
                "deficit": np.nan
                }
            ]
        

        # TODO: implement iteration over fluxes and update reservoir
        # iterate over timesteps using hydro_model
        iters = np.shape(net_precip)[0]
        for t in iters:
            pass
            
        #test_df = HydroModel()
        
        # Using apply function:
        # Using apply to calculate sum, product, and difference and update the DataFrame
        #df['int_cap'] = 2 # create column with interception capacity for apply() function
        #df[['int_stor', 'runoff']] = df.apply(HydroModel.update_state, axis=1)
        
        #TODO: probably quicker to calculate in np.array, and after calculations transform to df and insert datetime again.

 


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
        
        

