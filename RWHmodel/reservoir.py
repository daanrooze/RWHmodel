
class Reservoir():    
    def __init__(self,
        reservoir_cap: float,
        reservoir_stor: float = 0.0,
    ):
        self.reservoir_cap =  reservoir_cap
        self.reservoir_stor = reservoir_stor
        self.deficit = 0
        self.reservoir_overflow = 0
 
    def update_state(self, runoff, demand):
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
        
