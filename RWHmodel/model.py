from typing import Optional
import numpy as np

from RWHmodel.reservoir import Reservoir
from RWHmodel.timeseries import ConstantDemand, Demand, Forcing


class Model:
    def setup(
        self,
        root: str,
        name: str,
        forcing_fn: str,
        reservoir_volume: float,
        reservoir_state: float, # Deze niet nodig hier?
        timestep: int = 86400,
        constant_demand: Optional[float] = None,
        demand_fn: Optional[str] = None,
        demand_reduction: bool = False, # TODO: other name for variable. Does demand stop when reservoir is empty?
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
        
        # Setup demand #TODO: rewrite to take 1 arg; and class function takes either number and fill table, or sets up df
        if constant_demand and demand_fn:
            raise ValueError("Not allowed to provide both a constant demand and a demand timeseries file")
        elif constant_demand:
            self.demand = ConstantDemand(self.forcing, constant_demand)
        else:
            self.demand = Demand(
                demand_fn, root, timestep, unit=unit, area_chars=area_chars
            )

        self.reservoir = Reservoir(reservoir_volume, reservoir_state)

    def setup_from_config(self, config_fn):
        # TODO: parse config with code from config.py and call self.setup with arguments from config
        pass

    def run(self):
        # calc flux #TODO: move to 'hydro_model.py'
        forcing_flux = self.forcing.data["precip"] - self.forcing.data["pet"]
        flux = forcing_flux - self.demand.data #TODO: implement demand_reduction: when True, demand stops when res is empty
        # initialize numpy arrays for deficit, runoff, and storage
        deficit = np.zeros(flux.shape[0])
        runoff = np.zeros(flux.shape[0])
        storage = np.zeros(flux.shape[0])
        # TODO: implement iteration over fluxes and update reservoir
