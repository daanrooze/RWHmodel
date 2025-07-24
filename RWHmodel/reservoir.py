
class Reservoir():    
    def __init__(self,
        reservoir_cap: float,
        reservoir_stor: float = 0.0,
        srf_area: float = None,
        unit: str = "mm"
    ):
        # Convert reservoir capacity to mm if unit set to "m3".
        if unit == "m3":
            reservoir_cap = (reservoir_cap / srf_area) * 1000
            reservoir_stor = (reservoir_stor / srf_area) * 1000
        
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
