
class HydroModel():    
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
        
#TODO: implement independent_demand: when True, demand stops when res is empty
        
#%%

def hydro_model(
        self,
        precip,
        pet,
        forcing_fn: str = None,
        surf_char: dict = None
    ) -> None:
        """Calculate time-series of runoff based on input forcing hydrology (precipitation, PET) and
        receiving surface area characteristics.

        Returns Pandas DataFrame

        Parameters
        ----------
        p_atm : float 
            rainfall during current time step [mm]
        e_pot_ow : float
            potential open water evaporation during current time step [mm]
            
        Returns
        ----------
        (dictionary): A dictionary of computed states and fluxes of paved roof during current time step:

        * **int_pr** -- Interception storage on paved roof after rainfall at the beginning of current time step [mm]
        * **e_atm_pr** -- Evaporation from interception storage on paved roof during current time step [mm]
        * **intstor_pr** -- Remaining interception storage on paved roof at the end of current time step [mm]
        * **r_pr_meas** -- Runoff from paved roof to measure during current time step (not necessarily on paved roof itself) [mm]
        * **r_pr_swds** -- Runoff from paved roof to storm water drainage system (SWDS) during current time step [mm]
        * **r_pr_mss** -- Runoff from paved roof to combined sewer system (MSS) during current time step [mm]
        * **r_pr_up** -- Runoff from paved roof to unpaved during current time step [mm]
        """

        
        if surf_char == None:
            #TODO: add link to default surface characteristics data.
            pass 
        
        df = forcing_fn
        interception_storage = max(min(0, df['PRECIPITATION'] - df['PET']), surf_char['int_cap'])
        runoff = max(0, df['PRECIPITATION'] - df['PET'] - interception_storage)
        
        return {
            "int_pr": int_pr,
            "e_atm_pr": e_atm_pr,
            "intstor_pr": intstor_pr,
            "r_pr_meas": r_pr_meas,
            "r_pr_swds": r_pr_swds,
            "r_pr_mss": r_pr_mss,
            "r_pr_up": r_pr_up,
        }