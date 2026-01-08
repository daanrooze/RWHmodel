import numpy as np

class HydroModel:
    def __init__(self, int_cap):
        self.int_cap = int_cap

    def calc_runoff(self, net_precip):
        # Function that calculates the interception storage and runoff

        ### initialize numpy arrays
        int_stor = np.zeros(len(net_precip))
        runoff = np.zeros(len(net_precip))

        ### fill numpy arrays
        for i in range(1, len(net_precip)):
            if net_precip[i] >= 0:
                space_left = self.int_cap - int_stor[i - 1]         # Compute remaining interception storage capacity
                used_for_storage = min(space_left, net_precip[i])   # Determine the portion of net precipitation that can fill interception storage
                int_stor[i] = int_stor[i - 1] + used_for_storage    # Update interception storage with captured precipitation
                runoff[i] = net_precip[i] - used_for_storage        # Assign the remaining water as runoff
            else:
                new_storage = int_stor[i - 1] + net_precip[i]       # Reduce interception storage due to negative net precipitation
                int_stor[i] = max(0, new_storage)                   # Ensure storage does not fall below zero
                runoff[i] = 0.0                                     # No runoff occurs when net precipitation is negative

        return int_stor, runoff
