import os
from typing import Optional
import numpy as np
import pandas as pd
import codecs
import toml

from RWHmodel.reservoir import Reservoir
from RWHmodel.timeseries import ConstantDemand, Demand, Forcing
from RWHmodel.hydro_model import HydroModel
from RWHmodel.utils import makedir
from RWHmodel.analysis import return_period

from RWHmodel.plot import plot_meteo, plot_run, plot_system_curve, plot_saving_curve


class Model(object):
    def __init__(
        self,
        root: str,
        name: str,
        setup_fn: str,
        forcing_fn: str,
        demand_fn: str,
        reservoir_initial_state: float = 0,
        timestep: Optional[int] = None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        unit: str = "mm",
    ):
        # Setup folder structure
        if len(root)>0:
            self.root = root
        else:
            raise ValueError("Provide root of model folder")
        makedir(f"{self.root}/input")
        makedir(f"{self.root}/output")
        makedir(f"{self.root}/output/figures")
        makedir(f"{self.root}/output/runs")
        makedir(f"{self.root}/output/statistics")
        
        # Setup model run name
        if len(name)>0:
            self.name = name
        else:
            raise ValueError("Provide model run name")
        
        if timestep is not None and timestep not in [3600, 86400]:
            raise ValueError("Provide model timestep in 3600 or 86400 seconds")
        
        # Setup of area characteristics
        self.setup_from_toml(setup_fn=setup_fn)
        
        # Setup forcing
        self.forcing = Forcing(
            forcing_fn = forcing_fn,
            root = root,
            timestep = timestep,
            t_start = t_start,
            t_end = t_end
        )
        
        # Setup demand
        if type(demand_fn) in [float, int]:
            self.demand = ConstantDemand(
                forcing_fn = self.forcing.data,
                constant = demand_fn
            )
        else:
            self.demand = Demand(
                demand_fn = demand_fn,
                root = root,
                timestep = timestep,
                t_start = t_start,
                t_end = t_end,
                unit = unit,
                setup_fn = self.config
            )
        if self.forcing.timestep != self.demand.timestep:
            raise ValueError("Forcing and demand timeseries have different timesteps. Change input files or resample by specifying timestep.")
        if len(self.forcing.data) != len(self.demand.data):
            raise ValueError(f"Forcing and demand timeseries have different starting and/or end dates. Forcing timeseries runs from {self.forcing.t_start} to {self.forcing.t_end}, while demand timeseries runs from {self.demand.t_start} to {self.demand.t_end}. Change input files or clip timeseries by specifying interval.")
        #TODO: add method to automatically clip forcing and demand timeseries to smalles interval?
        
        # Initiate hydro_model
        self.hydro_model = HydroModel(int_cap = self.config['int_cap'])
        
        # Initiate reservoir
        if not 'reservoir_cap' in self.config:
            self.config['reservoir_cap'] = self.config['cap_min']
        self.reservoir = Reservoir(self.config['reservoir_cap'], reservoir_initial_state)
        
    def setup_from_toml(self, setup_fn):
        folder = f"{self.root}/input"
        with codecs.open(os.path.join(folder, setup_fn), "r", encoding="utf-8") as f:
            area_chars = toml.load(f)
        
        self.config = area_chars

    def run(
            self,
            demand: Optional = None,
            seasonal_variation = False,
            reservoir_cap: Optional = None,
            save=True,
        ):
        # Overwrite self with arguments if given.
        demand_array = np.full(len(self.forcing.data["precip"]), demand) if demand is not None else np.array(self.demand.data["demand"])
        if seasonal_variation:
            if self.demand.timestep == 86400:
                yearly_demand = demand_array[0] * 365
            else:
                yearly_demand = demand_array[0] * 24 * 365
            A, daily_demand_constant = self.demand.seasonal_variation(yearly_demand = yearly_demand, perc_constant = self.config["perc_constant"],shift= self.config["shift"])
            daily_demand_array = []
            for t in range(len(self.forcing.data["precip"])):
                daily_tot = A * np.sin(((2*np.pi)/365)*t+self.config["shift"]) + daily_demand_constant + A
                daily_demand_array.append(daily_tot)
            demand_array = daily_demand_array
        else:
            demand_array = demand_array
        reservoir_cap = reservoir_cap if reservoir_cap is not None else self.config["reservoir_cap"]
        
        ## Initialize numpy arrays
        net_precip = np.array(self.forcing.data["precip"] - self.forcing.data["pet"])

        reservoir_stor = np.zeros(len(net_precip))
        reservoir_overflow = np.zeros(len(net_precip))
        deficit = np.zeros(len(net_precip))
        dry_days = np.zeros(len(net_precip))
        consec_dry_days = np.zeros(len(net_precip))
        
        # Run hydro_model per timestep         
        int_stor, runoff = self.hydro_model.calc_runoff(net_precip=net_precip)   
        
        # Fill tank arrays
        for i in range(1, len(net_precip)):
            #reservoir_stor[i] = min(max(0, reservoir_stor[i-1] + runoff[i] - demand[i]), reservoir_cap) #TODO: remove
            #reservoir_overflow[i] =  max(0, reservoir_stor[i-1] + runoff[i] - demand[i] - reservoir_cap) #TODO: remove
            #deficit[i] = max(0, demand[i] - reservoir_stor[i]) #TODO: remove
            self.reservoir.update_state(runoff = runoff[i], demand = demand_array[i])
            reservoir_stor[i] = self.reservoir.reservoir_stor
            reservoir_overflow[i] = self.reservoir.reservoir_overflow
            deficit[i] = self.reservoir.deficit
            # Calculating days that reservoir does not suffice
            if reservoir_stor[i] >= demand_array[i]:
                dry_days[i] = 0
            elif reservoir_stor[i] < demand_array[i]:
                dry_days[i] = dry_days[i-1] + 1
            if dry_days[i-1] != 0 and dry_days[i] == 0:
                consec_dry_days[i] = dry_days[i-1]
            else:
                consec_dry_days[i] = 0
        
        # Convert to dataframe
        df_data = {
            'reservoir_stor': reservoir_stor,
            'reservoir_overflow': reservoir_overflow,
            'deficit': deficit,
            'consec_dry_days': consec_dry_days}
        df = pd.DataFrame(df_data, index = self.forcing.data['precip'].index)
        if save==True:
            df.to_csv(f"{self.root}/output/runs/{self.name}_single_run_reservoir={reservoir_cap}_demand={demand}.csv")
        self.results = df
        return df

 
    def batch_run(
            self,
            method,
            seasonal_variation=False,
            log=False,
            save=False #TODO: veranderd
        ):
        # Batch run function to obtain solution space and statistics on output.
        methods = ["total_days", "consecutive_days"]
        # Check if input is correct
        if method not in methods:
            raise ValueError(
                f"Provide valid method from {methods}."
            )
        if self.demand.unit != "mm" and len(self.config["typologies_name"]) > 1:  #TODO: veranderd
            raise ValueError(  #TODO: veranderd
                "Ambiguous surface area. Unit conversion can only be used for a maxmimum of one surface area."  #TODO: veranderd
            )  #TODO: veranderd
        #TODO: schrijf check of alle .config waardes aanwezig zijn
        
        # Define parameters
        dem_min = self.config["dem_min"]
        dem_max = self.config["dem_max"]
        dem_step = self.config["dem_step"]
        demand_lst = list(np.arange(dem_min, dem_max, dem_step)) #TODO: veranderd
        # Create reservoir range
        cap_min = self.config["cap_min"]
        cap_max = self.config["cap_max"]
        cap_step = self.config["cap_step"]
        capacity_lst = list(np.arange(cap_min, cap_max, cap_step)) #TODO: veranderd
        
        #T_range = self.config["T_return_list"]
        max_num_days = self.config["max_num_days"]

        df_system = pd.DataFrame(columns = self.config["T_return_list"]) 
        for reservoir_cap in capacity_lst:
            
            df_total = pd.DataFrame()
            dry_events = pd.DataFrame()
            req_storage = pd.DataFrame()
            for demand in demand_lst:
                if log == True:
                    print(f"Running with reservoir capacity {reservoir_cap} mm and demand {demand}.")
                
                run_df = self.run(demand = demand, reservoir_cap = reservoir_cap, save = save, seasonal_variation = seasonal_variation) #TODO: veranderd
                
                if method == "consecutive_days": 
                    dry_events = run_df["consec_dry_days"].sort_values(ascending = False).to_frame()
                    dry_events = dry_events.reset_index(drop = True)
                    dry_events = dry_events.rename(columns={'consec_dry_days': f'{demand}'})
                if method == "total_days": 
                    run_df_yearly = run_df.resample('YE').sum()
                    dry_events = run_df_yearly["consec_dry_days"].sort_values(ascending = False).to_frame()
                    dry_events = dry_events.reset_index(drop = True)
                    dry_events = dry_events.rename(columns={'consec_dry_days': f'{demand}'})
                df_total = pd.concat([df_total, dry_events[[f'{demand}']]], axis=1)
            df_total['T_return'] = self.forcing.num_years / (df_total.index + 1)
            req_storage = return_period(df_total, self.config["T_return_list"])
            # Find optimal demand for specific tank size
            opt_demand_lst = []
            for column in req_storage.columns:
                try:
                    # Filter rows where the value in the column is less than or equal to max_num_days
                    boundary_condition = req_storage[req_storage[column] <= max_num_days].index
                    # Check if there are any indices that meet the condition
                    if not boundary_condition.empty:
                        # Get the last index from the filtered results
                        opt_demand = int(boundary_condition[-1])
                        opt_demand_lst.append(opt_demand)
                    else:
                        # Append 0 if no valid index was found
                        opt_demand_lst.append(0)
                except Exception as e:
                    opt_demand_lst.append(0)

            opt_demand_df = pd.DataFrame([opt_demand_lst], columns = self.config["T_return_list"])
            df_system = pd.concat([df_system, opt_demand_df])
        df_system["tank_size"] = capacity_lst
        df_system = df_system.set_index("tank_size")
        self.statistics = df_system #TODO: veranderd
        df_system.to_csv(f"{self.root}/output/statistics/{self.name}_batch_run_{method}.csv") #TODO: veranderd
    
    def plot(
        self,
        plot_type=None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        fn: Optional[str] = None,
        T_return_list: Optional[[int]] = None,
        **kwargs
    ):
        plot_types = ["meteo", "run", "system_curve", "saving_curve"]
        if plot_type not in plot_types:
            raise ValueError(
                f"Provide valid plot type from {plot_types}."
            )
        # Overwrite self with arguments if given.
        t_start = pd.to_datetime(t_start) if t_start is not None else self.forcing.t_start
        t_end = pd.to_datetime(t_end) if t_end is not None else self.forcing.t_end
        
        if plot_type == "meteo":
            plot_meteo(
                root = self.root,
                name = self.name,
                forcing_fn = self.forcing.data,
                t_start = t_start,
                t_end = t_end,
                aggregate = False
            )
        
        if plot_type == "run":
            if fn:
                fn = pd.read_csv(fn, sep=',')
            else:
                fn = self.results
            plot_run(
                root = self.root,
                name = self.name,
                run_fn = fn,
                demand_fn = self.demand.data,
                t_start = t_start,
                t_end = t_end,
                reservoir_cap = self.reservoir.reservoir_cap,
                yearly_demand = self.demand.yearly_demand
            )
        
        if plot_type in ["system_curve", "saving_curve"]:
            if fn:
                fn = pd.read_csv(fn, sep=',')
            else:
                fn = self.statistics
            
            if T_return_list is None:
                   T_return_list = self.config['T_return_list']  # Use default from config if not provided
        
        if plot_type == "system_curve":
            plot_system_curve(
                root = self.root,
                name = self.name,
                system_fn = fn,
                max_num_days = self.config['max_num_days'],
                max_demand = self.config['dem_max'],
                T_return_list = T_return_list,
                validation = False
            )
        
        if plot_type == "saving_curve":
            plot_saving_curve(
                root = self.root,
                name = self.name,
                system_fn = fn,
                max_num_days = self.config['max_num_days'],
                typologies_name = self.config['typologies_name'],
                typologies_demand = self.config['typologies_demand'],
                typologies_area = self.config['typologies_area'],
                T_return_list = T_return_list,
                ambitions = None
            )


        
        

