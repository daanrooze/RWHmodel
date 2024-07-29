import numpy as np

### TRADITION LOOP IN THIS CELL

class HydroModel():    
    def __init__(self, int_cap):
        self.int_cap = int_cap 

    def update_state(self, net_precip):
        # Calculate the interception storage at this timestep
 
        ### initialize numpy arrays
        int_stor = np.zeros(len(net_precip))
        runoff = np.zeros(len(net_precip))
        
        ### fill numpy arrays
        for i in range(1, len(net_precip)):
            int_stor[i] = max(min(int_stor[i-1] + net_precip[i], self.int_cap), 0)
            runoff[i] = max(0, net_precip[i] - int_stor[i])


        return int_stor, runoff

#%%

### TRYING WITH APPLY() FUNCTION IN THIS CELL

# import pandas as pd

# class HydroModel:
#     def __init__(self, int_cap):
#         self.int_cap = int_cap
    
#     def update_state(row):
#         # Calculate the interception storage at this timestep
#         int_stor = min(max(0, row['net_precip']), row['int_cap']) #TODO: how to refer to row above? Difficult with apply function
#         # Calculate the runoff at this timestep
#         runoff = max(0, row['net_precip'] - int_stor)
#         return pd.Series({
#             'int_stor': int_stor,
#             'runoff': runoff
#         })
    
#     """
#     def update_state(self, row):
#         int_stor = min(max(0, row['net_precip']), self.int_cap)
#         runoff = max(0, row['net_precip'] - int_stor)
#         return pd.Series({
#             'intstor': int_stor,
#             'runoff': runoff
#         })
#     """
