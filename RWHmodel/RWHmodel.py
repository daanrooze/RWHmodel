# Importing packages and modules

import os
from os.path import basename, dirname, isdir, isfile, join
from typing import Optional, Dict, Any, Union, List

import numpy as np
import pandas as pd
import geopandas as gpd

import codecs
import toml


#%%

class RWHmodel:
    def __init__(
        self,
        root: str = None,
        name: str = None,
        ts: int = 86400
    ):
        """
        Initialization of RWH model class.

        Parameters
        ----------
        root : str, optional
            Root location of the model. The default is None.
        name : str, optional
            Run name of the model. The default is None.
        ts : int
            Timestep of the calculations in seconds. The default is 86400 seconds (1 day).

        Returns
        -------
        None.

        """
        if len(root)>0:
            self.root = root
        else:
            raise IOError(f"Provide root of model folder")
        if len(name)>0:
            self.name = name
        else:
            raise IOError(f"Provide model run name")
        self.ts = ts

#%%

class area_characteristics:
    def __init__(
        self
    ):
        return
    
    def read_area_characteristics(
        self,
        area_chars_fn,
    ):    
        with codecs.open(area_chars_fn, "r", encoding="utf-8") as f:
            area_chars = toml.load(f)
        area_chars = pd.read_csv(area_chars_fn)
        return area_chars

    def setup_area_characteristics(
        self,
        area_chars_fn,
    ):
        area_chars = self.read_area_characteristics(area_chars_fn)
        self.area_chars = area_chars
        


#%%

class forcing:
    def __init__(
        self
    ):
        return
        
    def read_forcing(
        self,
        forcing_fn: str = None,
        #TODO: add column names to parse input?
    ) -> None:
        """Reads local forcing data from csv.

        Returns Pandas DataFrame with time series of input forcing

        Parameters
        ----------
        forcing_fn : str, default None
            Precipitation and Potential Evapotranspiration data source.
        """
        #TODO: add function to read forcing data from data catalog instead of manual input
        #TODO: Possibility of adding default forcing data to DATADIR folder.
        #if forcing_fn == None:
            # Resort to default timeseries in DATADIR folder
            #TODO: add link to default data
            #pass 
        
        df = pd.read_csv(forcing_fn, sep=",")

        if not all(item in df.columns for item in ['datetime', 'precip', 'pet']):
                raise IOError("Provide forcing csv with headers 'datetime', 'precip', 'pet'")
        if not df['precip'].dtypes in ['float64', 'int', 'int64']:
            raise IOError("Provide precip values as float or int'")
        if not df['pet'].dtypes in ['float64', 'int', 'int64']:
            raise IOError("Provide pet values as float or int'")
            
        df["datetime"] =  pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M")
        df = df.set_index('datetime')
        df = df.resample(f'{self.ts}s', label='right').sum()
        
        return df


    #TODO: later, add function "read_forcing_NOAA" and use API (use script Roel)


    def statistics_forcing(
        self,
        statistics: bool = False,
    ) -> None:
        """Presents basic forcing statistics and plots.

        Returns Pandas DataFrame with time series of input forcing

        Parameters
        ----------
        statistics : bool, default False
            Option to provide basic statistics and forcing graphs in the data output folder.
        """        
        df = self.forcing
        if statistics == True:
            #TODO: add possibility to perform basic rainfall/evaporation statistics (not a priority).
            pass
        self.forcing = df

    def setup_forcing(
        self,
        forcing_fn,
    ):
        """
        Returns forcing timeseries to object.

        Parameters
        ----------
        forcing_fn : str, default None
            Precipitation and Potential Evapotranspiration data source.
        """
        if forcing_fn == None:
            raise IOError("No forcing file provided")
        
        df = self.read_forcing(forcing_fn=forcing_fn)
        self.forcing = df
        
    def write_forcing(
        self,
        forcing_fn: str = None,
        fn_out: str = None,
        file_format: str = "csv"
    ) -> None:
        """Writes forcing dataframe to desired file format (default is csv).
        
        Parameters
        ----------
        forcing_fn : str, default None
            Precipitation and Potential Evapotranspiration data source.
        fn_out: str, default None
            Path to export location. Defaults to root / output folder.
        file_format: str, default csv
            Specifies the file format for exporting forcing data.
        """        
        ff_list = ["csv"]
        if file_format not in ff_list:
            raise IOError(f"Provide supported file format from {ff_list}")
        
        if fn_out == None:
            path = join(self.root, "output", "forcing", f"Forcing_{self.name}.{file_format}")
        else:
            path = fn_out
        
        if file_format == "csv":
            df.to_csv(path, sep=',', date_format="%d-%m-%Y %H:%M")


#%%



class demand:
    def __init__(
        self
    ):
        return
    
    def read_demand(
        self,
        demand_fn: str = None,
        unit: str = "mm"
        #TODO: add column names to parse input?
    ) -> None:
        """Reads local demand data from csv.

        Returns Pandas DataFrame with time series of input demand forcing.

        Parameters
        ----------
        demand_fn : str, default None
            Demand data source.
        unit : str, default "mm"
            Unit of demand.
        """       
        df = pd.read_csv(demand_fn, sep=",")

        if not all(item in df.columns for item in ['datetime', 'demand']):
                raise IOError("Provide demand csv with headers 'datetime', 'demand'")
        if not df['demand'].dtypes in ['float64', 'int', 'int64']:
            raise IOError("Provide demand values as float or int'")
            
        df["datetime"] =  pd.to_datetime(df["datetime"], format="%d-%m-%Y %H:%M")
        df = df.set_index('datetime')
        df = df.resample(f'{self.ts}s', label='right').sum()
        
        demand_units_list = ["mm", "m3"]
        if unit not in demand_units_list:
            raise IOError(f"Provide units in format from {demand_units_list} per timestep")
        if unit == "m3":
            df['demand'] = df['demand'] / self.area_chars['srf_area ']
        
        return df
    
    def soil_moisture_model(
        self,
    ):
        #TODO: insert Laura's function (or copy from UWBM?)
        pass
    
    def setup_demand(
        self,
        demand_fn,
    ):
        """
        Returns forcing timeseries to object.

        Parameters
        ----------
        forcing_fn : str, default None
            Precipitation and Potential Evapotranspiration data source.
        """
        if demand_fn == None:
            raise IOError("No forcing file provided")
        
        df = self.demand_fn(demand_fn=demand_fn)
        self.demand = df
    
    def write_demand(
        self,
    ):
        #TODO: function 
        pass
    
    

#%%
        
class hydro_model:
    def __init__(
        self
    ):
        forcing_fn = self.forcing
        pass
    
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


#%%

class reservoir_model:
    def __init__(
        self,
        reservoir_dim: float = 0,
        storage: float = 0,
        inflow: float = 0,
        demand: float = 0,
        constant_demand: bool = True,
        reservoir_ini_stor_perc: float = 0 #TODO: this is in percentage. Should we go for absolute number instead?
        #TODO: option for dynamic demand (like the 70%-rule) ???
    ):
        """ Creates an instance of the reservoir model """
        self.reservoir_dim = reservoir_dim
        self.inflow = inflow
        self.demand = demand
        self.constant_demand = constant_demand
        
        # Calculate initial state
        self.storage = self.reservoir_dim * reservoir_ini_stor_perc

    def reservoir_model(
        self
    ):
        """Calculate reservoir storage and overflow.

        Returns ????

        Parameters
        ----------
        reservoir_dim : float
            Dimension of reservoir [mm].
        storage: float
            Amount of water stored in the current timestep [mm].
        inflow : float
            Dictionary with characteristics of the receiving surface data source [mm].
        demand: float
            Precipitation and Potential Evapotranspiration data source [mm].
        constant_demand : bool, default True
            Precipitation and Potential Evapotranspiration data source [mm].
        """
        # Reduce demand if condition is satisfied and storage is not sufficient. Otherwise track lack in water.
        if constant_demand == False:
            demand = min(demand, storage)
        else:
            gap = max(0, demand - storage)
        
        # Caculate the water storage in the reservoir for the current timestep
        storage = max(reservoir_dim, storage - demand)
        
        # Calculate overflow
        
        
        self.overflow = overflow
        self.gap = gap
        self.storage = storage
        
        return {
            
        }



#%%

class model:
    """ Class to iterate over timesteps with 'hydro_model' and 'reservoir_model' """
    def __init__(
        self
    ):
        return
    
    def running(
        self
    ):
        #datetime = self.forcing['datetime']
        #precip = self.forcing['precip']
        #pet = self.forcing['pet']
        #iters = np.shape(datetime)[0]
        # Duplicate and expand forcing df to store results.
        df = self.forcing
        df['net_precip'] = df['precip'] - df['pet']
        df['int_stor'] = np.nan
        df['runoff'] = np.nan
        df['reservoir_stor'] = np.nan
        df['reservoir_overflow'] = np.nan
        df['deficit'] = np.nan
        
        # Apply the hydro_model and reservoir_model to return the values of the df columns (see above)
        df.apply(
            self.hydro_model.hydro_model(),
            axis=1
        )


#%%

class analysis:
    """ Class to perform statistical analysis on RWH output """
    def __init(
        self
    ):
        """
        Initialization of analysis class.

        Returns
        -------
        None.

        """
    
    def func_log(a, b, x):
        return a * np.log(x) + b
    
    def func_system_curve(x, a, b, n):
        y = a * x ** n  / (x ** n + b)
        return y
    
    def func_system_curve_inv(y, a, b, n):
        x = ((a - y)  / (b * y))**(-1/n)
        return x
    
    def return_period(df): #TODO: variabelen namen nalopen. Deze komen nog uit UWBM
        colnames = df.columns[:-1]
        df_vars = pd.DataFrame(columns=["q", "a", "b"])
        df_vars["q"] = np.zeros(len(df.keys()[:-1]))

        for i, col in enumerate(colnames):
            
            mcl = df[col].count()
            
            x = (
                df["T_return"][0:mcl]
                .reindex(df["T_return"][0:mcl].index[::-1])
                .reset_index(drop=True)
            ).astype('float64')
            y = df[col][0:mcl].reindex(df[col][0:mcl].index[::-1]).reset_index(drop=True).astype('float64')
            a, b = np.polyfit(np.log(x)[0:mcl], y[0:mcl], 1)

            df_vars.loc[i] = [col, a, b]

        # Calculate required storage capacity for a set of return periods
        req_storage = pd.DataFrame()
        req_storage["Treturn"] = [1, 2, 5, 10, 20, 50, 100]
        for i, key in enumerate(df_vars["q"]):
            req_storage[key] = func_log(df_vars["a"][i], df_vars["b"][i], req_storage["Treturn"])
        req_storage = req_storage.set_index("Treturn")
        req_storage.index.name = None
        req_storage = req_storage.T
        
        return req_storage

#%%
''' TESTING '''

time_tuple = ("1990-01-01", "2019-12-31")
timesteps = pd.date_range(start = time_tuple[0], end = time_tuple[1], freq="d")
num_years = np.round(len(timesteps) / 365, 0)

surf_char = {
    "int_cap": 2 #Interception capacity in millimeter
    }

df = pd.DataFrame()



''' ------- '''







#%%

### OLD SCRIPTS ###




###Bestand: bodem_vocht_model.py

class BodemVochtModel:
    def __init__(self, bodemtype, oppervlakte, neerslag):
        self.bodemtype = bodemtype
        self.oppervlakte = oppervlakte
        self.neerslag = neerslag

    def bereken_bodemvocht(self):
        # Implementeer hier de berekening voor het bodemvocht
        pass
###Bestand: hydrologisch_afstroom_model.py

class HydrologischAfstroomModel:
    def __init__(self, oppervlakte, neerslag, afvoer_coefficient):
        self.oppervlakte = oppervlakte
        self.neerslag = neerslag
        self.afvoer_coefficient = afvoer_coefficient
        """
        forcing_fn : str, default None
            Precipitation and Potential Evapotranspiration data source.
        surf_char : str, default None
            Dictionary with characteristics of the receiving surface data source.
        """

    def bereken_afvoer(self):
        # Implementeer hier de berekening voor de afvoer op basis van neerslag en afvoercoëfficiënt
        pass
###Bestand: analyse_module.py

from bodem_vocht_model import BodemVochtModel
from hydrologisch_afstroom_model import HydrologischAfstroomModel

class AnalyseModule:
    def __init__(self, bodem_vocht_model, hydrologisch_afstroom_model):
        self.bodem_vocht_model = bodem_vocht_model
        self.hydrologisch_afstroom_model = hydrologisch_afstroom_model

    def voer_analyse_uit(self):
        bodemvocht = self.bodem_vocht_model.bereken_bodemvocht()
        afvoer = self.hydrologisch_afstroom_model.bereken_afvoer()

        # Voer verdere analyses uit op basis van bodemvocht en afvoer
        # Hieronder staat een voorbeeld van een eenvoudige analyse
        if bodemvocht > 50 and afvoer < 20:
            print("Optimale omstandigheden voor regenwaterhergebruik!")
        else:
            print("Niet ideale omstandigheden voor regenwaterhergebruik.")

#%%
#Bestand: main.py
from bodem_vocht_model import BodemVochtModel
from hydrologisch_afstroom_model import HydrologischAfstroomModel
from analyse_module import AnalyseModule

# Creëer instanties van de klassen
bodem_vocht_model = BodemVochtModel(bodemtype="zand", oppervlakte=1000, neerslag=500)
hydrologisch_afstroom_model = HydrologischAfstroomModel(oppervlakte=1000, neerslag=500, afvoer_coefficient=0.2)
analyse_module = AnalyseModule(bodem_vocht_model, hydrologisch_afstroom_model)

# Voer analyse uit
analyse_module.voer_analyse_uit()
#Deze code organiseert de functionaliteit van het regenwaterhergebruikmodel in afzonderlijke modules. Het hoofdbestand (main.py) importeert deze modules en maakt vervolgens instanties van de klassen aan om analyses uit te voeren.

