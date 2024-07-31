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
            int_stor[i] = max(min(int_stor[i - 1] + net_precip[i], self.int_cap), 0)
            runoff[i] = max(0, net_precip[i] - int_stor[i])

        return int_stor, runoff

