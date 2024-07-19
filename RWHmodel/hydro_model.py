

### TRADITION LOOP IN THIS CELL

class HydroModel():    
    def __init__(self, int_cap):
        self.int_cap = int_cap

    def update_state(self, int_stor, net_precip):
        # Calculate the interception storage at this timestep
        int_stor = max(min(int_stor + net_precip, self.int_cap), 0)
        # Calculate the runoff at this timestep
        runoff = max(0, net_precip - int_stor)

        return {
            'int_stor': int_stor,
            'runoff': runoff
        }


#%%

### TRYING WITH APPLY() FUNCTION IN THIS CELL

import pandas as pd

class HydroModel:
    def __init__(self, int_cap):
        self.int_cap = int_cap
    
    def update_state(row):
        # Calculate the interception storage at this timestep
        int_stor = min(max(0, row['net_precip']), row['int_cap']) #TODO: how to refer to row above? Difficult with apply function
        # Calculate the runoff at this timestep
        runoff = max(0, row['net_precip'] - int_stor)
        return pd.Series({
            'int_stor': int_stor,
            'runoff': runoff
        })
    
    """
    def update_state(self, row):
        int_stor = min(max(0, row['net_precip']), self.int_cap)
        runoff = max(0, row['net_precip'] - int_stor)
        return pd.Series({
            'intstor': int_stor,
            'runoff': runoff
        })
    """
