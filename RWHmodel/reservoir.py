
class Reservoir():    
    def __init__(self,
        reservoir_cap: float,
        reservoir_stor: float = 0.0,
        connected_srf_area: float = None,
        unit: str = "mm"
    ):
        # Convert reservoir capacity to mm if unit set to "m3".
        if unit == "m3":
            reservoir_cap = (reservoir_cap / connected_srf_area) * 1000
            reservoir_stor = (reservoir_stor / connected_srf_area) * 1000
        
        self.reservoir_cap =  reservoir_cap
        self.reservoir_stor = reservoir_stor
        self.deficit = 0
        self.reservoir_overflow = 0
 
    def update_state(self, runoff, demand):
        # Calculate tentative new storage after runoff and demand
        reservoir_stor = self.reservoir_stor + runoff - demand

        if reservoir_stor > self.reservoir_cap:
            # Overflow
            self.reservoir_overflow = reservoir_stor - self.reservoir_cap
            reservoir_stor = self.reservoir_cap
            self.deficit = 0.0
        elif reservoir_stor >= 0:
            # Demand fully met, no overflow
            self.deficit = 0.0
            self.reservoir_overflow = 0.0
        else:
            # Demand not fully met, deficit is shortfall
            self.deficit = -reservoir_stor  # positive deficit amount
            reservoir_stor = 0.0
            self.reservoir_overflow = 0.0

        self.reservoir_stor = reservoir_stor


class ReservoirOpen(Reservoir):
    def __init__(
        self,
        reservoir_cap: float,
        reservoir_stor: float = 0.0,
        connected_srf_area: float = None,
        reservoir_srf_area: float = None,
        unit: str = "mm"
    ):
        if connected_srf_area is None or reservoir_srf_area is None:
            raise ValueError("Both connected_srf_area and reservoir_srf_area must be provided.")

        self.connected_srf_area = connected_srf_area
        self.reservoir_srf_area = reservoir_srf_area

        # Convert reservoir volume to mm-equivalent over connected area
        if unit == "m3":
            reservoir_cap = (reservoir_cap / connected_srf_area) * 1000
            reservoir_stor = (reservoir_stor / connected_srf_area) * 1000

        self.reservoir_cap = reservoir_cap
        self.reservoir_stor = reservoir_stor
        self.deficit = 0.0
        self.reservoir_overflow = 0.0

    def update_state(self, runoff, demand, net_precip):
        """
        Parameters
        ----------
        runoff : float
            Runoff from connected surface [mm over connected area]
        demand : float
            Demand [mm over connected area]
        net_precip : float
            Net precipitation on reservoir surface [mm over reservoir surface]
        """

        # Convert reservoir precipitation to mm-equivalent over connected area
        net_precip_equiv = net_precip * (
            self.reservoir_srf_area / self.connected_srf_area
        )

        # Update tentative storage
        reservoir_stor = (
            self.reservoir_stor
            + runoff
            + net_precip_equiv
            - demand
        )

        if reservoir_stor > self.reservoir_cap:
            self.reservoir_overflow = reservoir_stor - self.reservoir_cap
            reservoir_stor = self.reservoir_cap
            self.deficit = 0.0
        elif reservoir_stor >= 0:
            self.reservoir_overflow = 0.0
            self.deficit = 0.0
        else:
            self.deficit = -reservoir_stor
            self.reservoir_overflow = 0.0
            reservoir_stor = 0.0

        self.reservoir_stor = reservoir_stor



class ReservoirGroundwater(Reservoir):
    """Reservoir with a groundwater connection, losing a fraction of storage per timestep."""
    def __init__(self, reservoir_cap, reservoir_stor=0.0, srf_area=None, unit="mm", gw_loss_factor=0.0):
        super().__init__(reservoir_cap, reservoir_stor, srf_area, unit)
        self.gw_loss_factor = gw_loss_factor  # fraction of storage lost per timestep

    def update_state(self, runoff, demand):
        # Add runoff
        self.reservoir_stor += runoff
        # Apply groundwater loss
        self.reservoir_stor *= (1 - self.gw_loss_factor)
        # Handle overflow and deficit
        if self.reservoir_stor > self.reservoir_cap:
            self.reservoir_overflow = self.reservoir_stor - self.reservoir_cap
            self.reservoir_stor = self.reservoir_cap
        else:
            self.reservoir_overflow = 0.0

        self.deficit = max(0, demand - self.reservoir_stor)
        self.reservoir_stor = max(0, self.reservoir_stor - demand)