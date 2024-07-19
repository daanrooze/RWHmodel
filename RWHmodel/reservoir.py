
class Reservoir():    
    def __init__(self,
        reservoir_cap: float,
        reservoir_stor: float = 0.0,
        independent_demand: bool = True
    ):
        self.reservoir_cap =  reservoir_cap
        self.reservoir_stor = reservoir_stor
        self.runoff = 0
        self.deficit = 0
        self.independent_demand

    def update_state(self, reservoir_stor, runoff, demand):
        if self.independent_demand == True: # In this case, demand will continue with its original size.
            reservoir_stor = self.reservoir_stor + runoff
            if reservoir_stor > self.reservoir_cap:
                reservoir_overflow = reservoir_stor - self.reservoir_cap
                reservoir_stor = self.reservoir_cap
            elif reservoir_stor < 0:
                deficit = abs(reservoir_stor)
                reservoir_stor = 0.0

        else: #In this case, demand will stop once reservoir is empty. Or is this not needed? Just track the deficit?
            reservoir_stor = self.reservoir_stor + runoff
            if reservoir_stor > self.reservoir_cap:
                reservoir_overflow = reservoir_stor - self.reservoir_cap
                reservoir_stor = self.reservoir_cap

            deficit = 0
        
        return {
            'reservoir_stor': reservoir_stor,
            'reservoir_overflow': reservoir_overflow,
            'deficit': deficit
        }
        

        
        # van Tjalling:
        state = self.state + flux
        if state > self.reservoir_cap:
            self.runoff = state - self.reservoir_cap
            state = self.reservoir_cap
        elif state < 0:
            self.deficit = abs(state) # make negative number of deficit an absolute number
            state = 0.0
        self.state = state
        
        
        # old: reservoir_stor = min( max(0, reservoir_stor + runoff - demand), self.reservoir_cap)