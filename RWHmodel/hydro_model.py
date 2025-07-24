import numpy as np

class HydroModel:
    def __init__(self, int_cap):
        self.int_cap = int_cap

    def calc_runoff(self, net_precip):
        # Calculate the interception storage at this timestep

        ### initialize numpy arrays
        int_stor = np.zeros(len(net_precip))
        runoff = np.zeros(len(net_precip))

        ### fill numpy arrays
        for i in range(1, len(net_precip)):
            if net_precip[i] >= 0:
                space_left = self.int_cap - int_stor[i - 1]
                used_for_storage = min(space_left, net_precip[i])
                int_stor[i] = int_stor[i - 1] + used_for_storage
                runoff[i] = net_precip[i] - used_for_storage
            else:
                # evaporation reduces storage, never below zero
                new_storage = int_stor[i - 1] + net_precip[i]
                int_stor[i] = max(0, new_storage)
                runoff[i] = 0.0

        return int_stor, runoff
