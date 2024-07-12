from typing import Optional
import numpy as np

from RWHmodel.reservoir import Reservoir
from RWHmodel.timeseries import ConstantDemand, Demand, Forcing


class Model:
    def setup(
        self,
        forcing_fn: str,
        timestep: int,
        root: str,
        reservoir_volume: float,
        reservoir_state: float,
        constant_demand: Optional[float] = None,
        demand_fn: Optional[str] = None,
        unit: str = "mm",
        area_chars: Optional[dict] = None,
    ):
        self.forcing = Forcing(forcing_fn, timestep, root)
        if constant_demand and demand_fn:
            raise ValueError("Not allowed to give a constant demand and a demand file")

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
        # calc flux
        forcing_flux = self.forcing.data["precip"] - self.forcing.data["pet"]
        flux = forcing_flux - self.demand.data
        # initialize numpy arrays for deficit, runoff, and storage
        deficit = np.zeros(flux.shape[0])
        runoff = np.zeros(flux.shape[0])
        storage = np.zeros(flux.shape[0])
        # TODO: implement iteration over fluxes and update reservoir
