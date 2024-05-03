# Importing packages and modules

import os
from os.path import basename, dirname, isdir, isfile, join
from typing import Optional, Dict, Any, Union, List

import numpy as np
import pandas as pd
import geopandas as gpd


#%%


class hydro_model:
    def __init__(self, oppervlakte, neerslag, afvoer_coefficient):
        self.oppervlakte = oppervlakte
        self.neerslag = neerslag
        self.afvoer_coefficient = afvoer_coefficient



    def check_forcing(
        self,
        forcing_fn: str = None,
        statistics: bool = False,
        **kwargs    
    ) -> None:
        """Checks provided forcing data for consistency.
        Presents basic forcing statistics and plots if requested.

        Returns Pandas DataFrame with time series of input forcing

        Parameters
        ----------
        forcing_fn : str, default None
            Precipitation and Potential Evapotranspiration data source.
        statistics : bool, default False
            Option to provide basic statistics and forcing graphs in the data output folder.
        """
        if forcing_fn == None:
            print("No forcing file provided") #TODO how to give back error message?
        
        
        
        

    def hydro_model(
        self,
        forcing_fn: str = None,
        **kwargs
    ) -> None:
        """Calculate time-series of runoff based on input forcing hydrology (precipitation, PET) and
        receiving surface area characteristics.

        Returns Pandas DataFrame with time series of input forcing

        Parameters
        ----------
        forcing_fn : str, default None
            Precipitation and Potential Evapotranspiration data source.
        surf_char : str, default None
            Library with characteristics of the receiving surface data source.
        """
        if forcing_fn == None:
            # Resort to default timeseries in data folder
            #TODO: add link to default data
            
            pass 
        
        # Implementeer hier de berekening voor de afvoer op basis van neerslag en afvoercoëfficiënt
        pass





#%%



class demand_model:
    def __init__(self):
        pass
    
    def read_demand:
        
        pass
    
    def soil_moisture_model:
        
        pass
    
    def setup_demand:
        
        pass
    
    def demand_model:
        
        pass
    
    








#%%















class tankmodel:
    def __init__(self, oppervlakte, neerslag, afvoer_coefficient):
        self.oppervlakte = oppervlakte
        self.neerslag = neerslag
        self.afvoer_coefficient = afvoer_coefficient

    def bereken_afvoer(self):
        # Implementeer hier de berekening voor de afvoer op basis van neerslag en afvoercoëfficiënt
        pass



#%%

#
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

