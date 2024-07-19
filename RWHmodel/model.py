from typing import Optional
import numpy as np

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
        # Adding empty arrays for to be calulcated factors:
        df['int_stor'] = np.nan            # interception storage
        df['runoff'] = np.nan              # runoff
        df['reservoir_stor'] = np.nan      # storage in reservoir
        df['reservoir_overflow'] = np.nan  # reservoir overflow
        df['deficit'] = np.nan             # deficit: original demand that cannot be met


        # Other way, using numpy arrays
        #old: flux = net_precip - self.demand.data 
        ### initialize numpy arrays for deficit, runoff, and storage
        # int_stor = np.zeros(self.forcing.data["net_precip"].shape[0])
        # runoff = np.zeros(self.forcing.data["net_precip"].shape[0])
        # reservoir_stor = np.zeros(self.forcing.data["net_precip"].shape[0])
        # reservoir_overflow = np.zeros(self.forcing.data["net_precip"].shape[0])
        # deficit = np.zeros(self.forcing.data["net_precip"].shape[0])
        
        # TODO: implement iteration over fluxes and update reservoir
        # iterate over timesteps using hydro_model
        
        #test_df = HydroModel()
        
        # Using apply function:
        # Using apply to calculate sum, product, and difference and update the DataFrame
        df['int_cap'] = 2 # create column with interception capacity for apply() function
        df[['int_stor', 'runoff']] = df.apply(HydroModel.update_state, axis=1)
        #df = df.apply(HydroModel.update_state(self), axis=1, raw=False, engine="python")
        #df = df.apply(Reservoir, axis=1, raw=False, engine="python")
        
        #TODO: probably quicker to calculate in np.array, and after calculations transform to df and insert datetime again.
        
        
        
        

