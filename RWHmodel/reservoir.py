
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
        """
        reservoir_stor = self.reservoir_stor + runoff - demand
        if reservoir_stor > self.reservoir_cap:
            self.reservoir_overflow = reservoir_stor - self.reservoir_cap
            reservoir_stor = self.reservoir_cap
        elif reservoir_stor > demand:
            self.deficit = 0.0
            self.reservoir_overflow = 0
        elif reservoir_stor > 0 and reservoir_stor <= demand:
            self.deficit = demand - reservoir_stor
            self.reservoir_overflow = 0
        elif reservoir_stor <= 0:
            self.deficit = demand
            reservoir_stor = 0.0
            self.reservoir_overflow = 0
        self.reservoir_stor = reservoir_stor
        """
        available = self.reservoir_stor + runoff  # total available water
        reservoir_stor = available - demand       # after demand

        if reservoir_stor > self.reservoir_cap:
            # overflow happens
            self.reservoir_overflow = reservoir_stor - self.reservoir_cap
            reservoir_stor = self.reservoir_cap
            self.deficit = 0.0
        elif reservoir_stor >= 0:
            # demand fully met, no overflow
            self.reservoir_overflow = 0.0
            self.deficit = 0.0
        else:
            # demand not met, deficit > 0
            self.reservoir_overflow = 0.0
            self.deficit = -reservoir_stor  # unmet demand = positive number
            reservoir_stor = 0.0

        self.reservoir_stor = reservoir_stor
        
