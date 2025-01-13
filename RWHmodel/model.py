import os
from typing import Optional
import numpy as np
import pandas as pd
import codecs
import toml

from RWHmodel.reservoir import Reservoir
from RWHmodel.timeseries import Demand, Forcing
from RWHmodel.hydro_model import HydroModel
from RWHmodel.utils import makedir, check_variables, convert_mm_to_m3, colloquial_date_text
from RWHmodel.analysis import return_period

from RWHmodel.plot import plot_meteo, plot_run, plot_run_coverage, plot_system_curve, plot_saving_curve


class Model(object):
    def __init__(
        self,
        root: str,
        name: str,
        setup_fn: str,
        forcing_fn: str,
        demand_fn: Optional[str] = None,
        demand_transform = False,
        reservoir_range: Optional[list] = None,
        reservoir_initial_state: float = 0, # as fraction of reservoir capacity
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
        for folder in ['input', 'output', 'output/figures', 'output/runs', 'output/runs/summary', 'output/statistics']:
            makedir(os.path.join(self.root, folder))
        
        # Check if seasonal transformation can be applied
        if type(demand_fn)==str and demand_transform==True:
                    raise ValueError(
                        "Cannot transpose timeseries with seasonal variation."
        )

        # Set model mode default to 'single'. Overrides to 'batch' if demand_fn is of type list.
        if type(demand_fn)==list:
            self.mode = 'batch'
        else: 
            self.mode = 'single'
        
        self.unit = unit
        
        # Setup model run name
        if len(name)>0:
            self.name = name
        else:
            raise ValueError("Provide model run name")
        
        if timestep is not None and timestep not in [3600, 86400]:
            raise ValueError("Provide model timestep in 3600 or 86400 seconds")
        
        # Setup of area characteristics
        self.setup_from_toml(setup_fn=setup_fn)

        # Check whether all required variables are provided
        check_variables(self.mode, self.config, demand_transform) #TODO: improve functionality so mode is automatically determined
        
        # Setup forcing
        self.forcing = Forcing(
            forcing_fn = forcing_fn,
            root = root,
            timestep = timestep,
            t_start = t_start,
            t_end = t_end
        )
        
        # Setup demand if given
        if type(demand_fn)==list:
            self.config["dem_min"] = demand_fn[0]
            self.config["dem_max"] = demand_fn[1]
            if len(demand_fn) == 3:
                self.config["dem_step"] = demand_fn[2]
            else:
                self.config["dem_step"] = 100  # Default number of steps = 100

        self.demand = Demand(
            root = root,
            demand_fn = demand_fn,
            demand_transform = demand_transform,
            forcing_fn = self.forcing.data,
            timestep = timestep,
            t_start = t_start,
            t_end = t_end,
            unit = unit,
            setup_fn = self.config,
            perc_constant = self.config["perc_constant"],
            shift= self.config["shift"],
        )

        if self.forcing.timestep != self.demand.timestep:
            raise ValueError("Forcing and demand timeseries have different timesteps. Change input files or resample by specifying timestep.")
        if len(self.forcing.data) != len(self.demand.data):
            raise ValueError(
                f"Forcing and demand timeseries have different starting and/or end dates. "
                f"Forcing timeseries runs from {self.forcing.t_start} to {self.forcing.t_end}, "
                f"while demand timeseries runs from {self.demand.t_start} to {self.demand.t_end}. "
                "Change input files or clip timeseries by specifying interval."
            )
        
        # Initiate hydro_model
        self.hydro_model = HydroModel(int_cap = self.config['int_cap'])
        
        # Generate reservoir range if given
        if reservoir_range:
            self.config["cap_min"] = reservoir_range[0]
            self.config["cap_max"] = reservoir_range[1]
            if len(reservoir_range) == 3:
                self.config["cap_step"] = reservoir_range[2]
            else:
                self.config["cap_step"] = 100  # Default number of steps = 100
        
        # Set reservoir capacity
        if not 'reservoir_cap' in self.config:
            self.config['reservoir_cap'] = self.config['cap_min']
        # Convert reservoir capacity to mm if unit set to "m3".
        if self.unit == "m3":
            self.config['reservoir_cap'] = (self.config['reservoir_cap'] / self.config["srf_area"]) * 1000
        # Check if percentage of reservoir_initial_state is not greater than 1
        if reservoir_initial_state > 1:
            raise ValueError("Provide initial reservoir state as fraction of reservoir capacity (between 0 and 1).")
        # Check if the maximum demand exceeds the reservoir capacity.
        if (self.demand.data.max() > self.config['reservoir_cap']).any():
            print("Warning: maximum demand is greater than the reservoir capacity.")

        # Initiate reservoir
        self.reservoir = Reservoir(self.config['reservoir_cap'], reservoir_initial_state * self.config['reservoir_cap'])

        
    def setup_from_toml(self, setup_fn):
        folder = f"{self.root}/input"
        with codecs.open(os.path.join(folder, setup_fn), "r", encoding="utf-8") as f:
            area_chars = toml.load(f)
        self.config = area_chars


    def run(
            self,
            reservoir_cap: Optional[float] = None,
            save=True,
        ):
        """
        #TODO: move from here ######################################################################################################################
        # Overwrite data in self with arguments if given.
        demand_array = np.full(len(self.forcing.data["precip"]), demand) if demand is not None else np.array(self.demand.data["demand"])
        # Implement seasonal variation transformation if given.
        if type(self.demand.fn)==str and seasonal_variation==True: #TODO: move outside of run function to batch run
            raise ValueError(
                "Cannot transpose timeseries with seasonal variation."
            )
        
        if self.demand.timestep == 86400: #TODO: move outside of run function to batch run
            yearly_demand = demand_array[0] * 365
        else:
            yearly_demand = demand_array[0] * 24 * 365
        
        if seasonal_variation: #TODO: move outside of run function to batch run
            demand_array = self.demand.seasonal_variation(
                yearly_demand = yearly_demand,
                perc_constant = self.config["perc_constant"],
                shift= self.config["shift"],
                t_start = self.forcing.t_start,
                t_end = self.forcing.t_end
            )
        self.demand.data.loc[:, "demand"] = demand_array #TODO: move outside of run function to batch run
        
        #TODO: move till here ######################################################################################################################
        """
        demand_array = self.demand.data.loc[:, "demand"]
        
        ## Initialize numpy arrays
        net_precip = np.array(self.forcing.data["precip"] - self.forcing.data["pet"])
        reservoir_stor = np.zeros(len(net_precip))
        reservoir_overflow = np.zeros(len(net_precip))
        deficit = np.zeros(len(net_precip))
        dry_days = np.zeros(len(net_precip))
        deficit_timesteps = np.zeros(len(net_precip))
        
        # Run hydro_model per timestep         
        int_stor, runoff = self.hydro_model.calc_runoff(net_precip=net_precip)   
        
        #TODO: 2 lines below just for intermediate checking
        print(demand_array)
        print(reservoir_cap)

        # Run reservoir model per timestep
        for i in range(1, len(net_precip)):
            self.reservoir.update_state(runoff = runoff[i], demand = demand_array.iloc[i])
            reservoir_stor[i] = self.reservoir.reservoir_stor
            reservoir_overflow[i] = self.reservoir.reservoir_overflow
            deficit[i] = self.reservoir.deficit
            # Tracking the timesteps for which the reservoir does not suffice.
            if reservoir_stor[i] >= demand_array.iloc[i]:
                dry_days[i] = 0
            elif reservoir_stor[i] < demand_array.iloc[i]:
                dry_days[i] = dry_days[i-1] + 1
            if dry_days[i-1] != 0 and dry_days[i] == 0:
                deficit_timesteps[i] = dry_days[i-1]
            else:
                deficit_timesteps[i] = 0
        
        # Convert to dataframe
        df_data = {
            'reservoir_stor': reservoir_stor,
            'reservoir_overflow': reservoir_overflow,
            'deficit': deficit,
            'deficit_timesteps': deficit_timesteps}
        df = pd.DataFrame(df_data, index = self.forcing.data['precip'].index)
        
        # Convert back to desired units
        if self.unit == "m3":
            df = convert_mm_to_m3(df = df, col = "reservoir_stor", surface_area = self.config["srf_area"])
            df = convert_mm_to_m3(df = df, col = "reservoir_overflow", surface_area = self.config["srf_area"])
            df = convert_mm_to_m3(df = df, col = "deficit", surface_area = self.config["srf_area"])
            self.demand.data = convert_mm_to_m3(df = self.demand.data, col = "demand", surface_area = self.config["srf_area"])
        
        # Calculate summarized results
        demand_deficit = df['deficit'].sum()
        demand_from_reservoir = self.demand.data['demand'].sum() - df['deficit'].sum()
        
        if self.mode=='single':
            print(f"Total demand:                   {np.round( self.demand.data['demand'].sum(), 1 )} {self.unit}")
            print(f"Total demand from reservoir:    {np.round( demand_from_reservoir, 1 )} {self.unit}")
            print(f"Total deficit:                  {np.round( demand_deficit, 1 )} {self.unit}")
        
        self.results = df
        self.results_summary = {}
        self.results_summary['demand_from_reservoir'] = demand_from_reservoir
        self.results_summary['demand_deficit'] = demand_deficit
        
        # Save results
        if save==True:
            df.to_csv(f"{self.root}/output/runs/{self.name}_single_run_reservoir={np.round(self.reservoir.reservoir_cap,1)}_yr_demand={np.round(self.demand.yearly_demand, 1)}.csv")
        
        return df



    def batch_run(
        self,
        method,
        log=False,
        save=False
    ):
        # Batch run function to obtain solution space and statistics on output.
        # Check if input is correct
        methods = ["total_days", "consecutive_days"]
        if method not in methods:
            raise ValueError(f"Provide valid method from {methods}.")
        if self.unit != "mm" and len(self.config["typologies_name"]) > 1:
            raise ValueError(
                "Ambiguous surface area. Unit conversion can only be used for a maximum of one surface area."
            )
    
        # Define demand range
        demand_lst = list(np.linspace(self.config["dem_min"], self.config["dem_max"], self.config["dem_step"]))
        # Create reservoir capacity range
        capacity_lst = list(np.linspace(self.config["cap_min"], self.config["cap_max"], self.config["cap_step"]))
        
        max_num_days = self.config["max_num_days"]
    
        # Initialize df_system to store demand and reservoir figures for defined return periods (satisfying the max_num_days requirement)
        df_system = pd.DataFrame(columns=self.config["T_return_list"] + ['reservoir_cap'])
        
        # Initialize df_coverage to store summaries of coverage and deficit for each run.
        df_coverage = pd.DataFrame(columns=demand_lst)
        df_coverage['reservoir_cap'] = capacity_lst
        df_coverage.set_index('reservoir_cap', inplace=True)
    
        for reservoir_cap in capacity_lst:
            
            # Update reservoir capacity to self
            self.reservoir.reservoir_cap = reservoir_cap
            
            df_deficit_events_total = pd.DataFrame()
            deficit_events_T_return = pd.DataFrame()
            
            for demand in demand_lst:
                
                # Update timeseries, including seasonal transformation and yearly demand
                Demand.update_demand(self.demand, update_data = demand)
                """
                # Implement seasonal variation transformation if given.
                # Update yearly demand
                self.demand.yearly_demand = demand * (86400 / self.demand.timestep) * 365

                if self.demand.transform:
                    self.demand.data.loc[:, "demand"] = self.demand.seasonal_variation(
                        yearly_demand = self.demand.yearly_demand,
                        perc_constant = self.config["perc_constant"],
                        shift= self.config["shift"],
                        t_start = self.forcing.t_start,
                        t_end = self.forcing.t_end
                    )
                #else:
                #    self.demand.data.loc[:, "demand"] = 
                
                """
                if log:
                    timestep_txt = colloquial_date_text(self.forcing.timestep)
                    print(f"Running with reservoir capacity {np.round(reservoir_cap, 2)} mm and demand {np.round(demand, 2)} mm/{timestep_txt}.")
                
                df_run = self.run(reservoir_cap=reservoir_cap, save=save)
                
                df_deficit_events = pd.DataFrame()
                if method == "consecutive_days": 
                    df_deficit_events = df_run["deficit_timesteps"].sort_values(ascending=False).to_frame()
                    df_deficit_events = df_deficit_events.reset_index(drop=True)
                    df_deficit_events = df_deficit_events.rename(columns={'deficit_timesteps': f'{demand}'})
                if method == "total_days": 
                    df_run = df_run.resample('YE').sum()
                    df_deficit_events = df_run["deficit_timesteps"].sort_values(ascending=False).to_frame()
                    df_deficit_events = df_deficit_events.reset_index(drop=True)
                    df_deficit_events = df_deficit_events.rename(columns={'deficit_timesteps': f'{demand}'})
                    
                df_deficit_events_total = pd.concat([df_deficit_events_total, df_deficit_events[[f'{demand}']]], axis=1)
                
                # Calculate coverage
                total_demand_sum = self.demand.data['demand'].sum()
                if np.isnan(total_demand_sum) or total_demand_sum == 0:
                    df_coverage.loc[reservoir_cap, demand] = 1
                else:
                    df_coverage.loc[reservoir_cap, demand] = (self.results_summary['demand_from_reservoir'] / total_demand_sum)
            
            df_deficit_events_total['T_return'] = self.forcing.num_years / (df_deficit_events_total.index + 1)
            deficit_events_T_return = return_period(df_deficit_events_total, self.config["T_return_list"])
            
            # Find maximum demand for specific reservoir size that satisfies the max_num_days requirement
            opt_demand_lst = []
            for column in deficit_events_T_return.columns:
                boundary_condition = deficit_events_T_return[deficit_events_T_return[column] <= max_num_days].index
                opt_demand = boundary_condition[-1] if not boundary_condition.empty else 0
                opt_demand_lst.append(opt_demand)
                
            # Create a DataFrame for the current reservoir size's results
            opt_demand_df = pd.DataFrame([opt_demand_lst], columns=self.config["T_return_list"])
            # Add the reservoir size as a new column to this DataFrame
            opt_demand_df["reservoir_cap"] = reservoir_cap
    
            # Ensure the dtypes match before concatenation
            for col in df_system.columns:
                if col in opt_demand_df.columns and df_system[col].isna().all():
                    df_system[col] = df_system[col].astype(opt_demand_df[col].dtype)
            
            # Append this DataFrame to df_system
            df_system = pd.concat([df_system, opt_demand_df], ignore_index=True)
    
        # Move 'reservoir_cap' column to the front
        df_system = df_system[ ['reservoir_cap'] + [ col for col in df_system.columns if col != 'reservoir_cap' ] ]
        df_system.columns = df_system.columns.astype(str)
        
        df_coverage = df_coverage.reset_index(drop=False)
        
        self.statistics = df_system
        self.results_summary = df_coverage
        
        # Save output to csv
        df_system.to_csv(f"{self.root}/output/statistics/{self.name}_batch_run_{method}.csv", index=False)
        df_coverage.to_csv(f"{self.root}/output/runs/summary/{self.name}_batch_run_coverage_summary.csv", index=False)


    def plot(
        self,
        plot_type=None,
        t_start: Optional[str] = None,
        t_end: Optional[str] = None,
        timestep: Optional[int] = None,
        fn: Optional[str] = None,
        T_return_list: Optional[int] = None,
        validation: Optional[str] = None,
        reservoir_max: Optional[int] = None,
        **kwargs
    ):
        plot_types = ["meteo", "run", "run_coverage", "system_curve", "saving_curve"]
        if plot_type not in plot_types:
            raise ValueError(
                f"Provide valid plot type from {plot_types}."
            )
        # Overwrite self with arguments if given.
        t_start = pd.to_datetime(t_start) if t_start is not None else self.forcing.t_start
        t_end = pd.to_datetime(t_end) if t_end is not None else self.forcing.t_end
        if not timestep:
            timestep = self.demand.timestep
        
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
            if not hasattr(self, 'results') and not fn:
                raise ValueError(
                    f"Run model or load in previous batch run results to plot {plot_type}."
                )
            if fn:
                fn = pd.read_csv(fn, sep=',')
            else:
                fn = self.results
            
            plot_run(
                root = self.root,
                name = self.name,
                run_fn = fn,
                demand_fn = self.demand.data,
                unit = self.unit,
                t_start = t_start,
                t_end = t_end,
                reservoir_cap = self.reservoir.reservoir_cap,
                yearly_demand = self.demand.yearly_demand
            )
    
        if plot_type == "run_coverage":
            if not hasattr(self, 'results_summary') and not fn:
                raise ValueError(
                    f"Run model or load in previous batch run results to plot {plot_type}."
                )
            if fn:
                fn = pd.read_csv(fn, sep=',')
            else:
                fn = self.results_summary
                
            plot_run_coverage(
                root = self.root,
                name = self.name,
                run_fn = fn,
                unit = self.unit,
                timestep = timestep,
            )
        
        if plot_type in ["system_curve", "saving_curve"]:
            if not hasattr(self, 'statistics') and not fn:
                raise ValueError(
                    f"Perform batch run or load in previous batch run results to plot {plot_type}."
                )
            if fn:
                fn = pd.read_csv(fn, sep=',')
            else:
                fn = self.statistics
            
            if T_return_list is None:
                   T_return_list = self.config['T_return_list']
            
            if plot_type == "system_curve":
                plot_system_curve(
                    root = self.root,
                    name = self.name,
                    system_fn = fn,
                    max_num_days = self.config['max_num_days'],
                    timestep = timestep,
                    T_return_list = T_return_list,
                    validation = validation
                )
            
            if plot_type == "saving_curve":
                plot_saving_curve(
                    root = self.root,
                    name = self.name,
                    unit = self.unit,
                    system_fn = fn,
                    max_num_days = self.config['max_num_days'],
                    typologies_name = self.config['typologies_name'],
                    typologies_demand = self.config['typologies_demand'],
                    typologies_area = self.config['typologies_area'],
                    T_return_list = T_return_list,
                    reservoir_max = reservoir_max,
                    ambitions = None
                )
