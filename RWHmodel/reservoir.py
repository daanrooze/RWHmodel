
class Reservoir():    
    def __init__(self, reservoir_volume: float, state: float = 0.0):
        self.reservoir_volume =  reservoir_volume
        self.state = state
        self.runoff = 0
        self.deficit = 0

    def update_state(self, flux):
        state = self.state + flux
        if state > self.reservoir_volume:
            self.runoff = state - self.reservoir_volume
            state = self.reservoir_volume
        elif state < 0:
            self.deficit = abs(state) # make negative number of deficit an absolute number
            state = 0.0
        self.state = state
        

        
