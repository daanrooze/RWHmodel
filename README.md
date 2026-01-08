
# RWHmodel | An exploratory model to assess the potential of rainwater harvesting.

# Installation Guide
Use conda/mamba to install the environment for development:

```
conda env create -f environment.yml
```

Then activate the created environment (called ‘RWHmodel’) and perform a developer install with pip:

```
pip install -e .
```

# Introduction
## Global relevance
The availability of fresh water is becoming an increasing challenge worldwide, including in India. With ongoing population growth and urbanization, especially in delta cities, the demand for fresh water is rising while supplies are diminishing. Changing rainfall and hydrology patterns, along with declining groundwater levels, further complicate the situation. Additionally, many water systems are designed to quickly remove rainwater to avoid flooding, rather than storing it for future use. This highlights the need for a new perspective on freshwater availability.

Rainwater harvesting, an ancient technique still embedded in many cultures, offers a potential solution for securing long-term freshwater supplies. It can also help buffer against extreme rainfall events. However, while rainwater harvesting is increasingly recommended in urban development projects, there is limited understanding of the required system sizes and types. Local experience provides some guidelines, but a more quantitative approach is needed. To address this, Deltares developed a conceptual model to better understand the potential of rainwater harvesting in urban areas.

The immediate driver for this model was the drought conditions in recent summers, during which drinking water suppliers reported difficulties in meeting the water demands for new homes and businesses. Using Vitens as a case study, Deltares developed a methodology to assess whether local climate conditions and building types are suitable for rainwater harvesting .

## Intended beneficiaries and impact
This tool aims to provide a practical way for users to evaluate the potential for rainwater harvesting in urban environments, empowering them to make informed decisions about water management.

The primary beneficiaries are people who need access to freshwater, as well as drinking water suppliers and businesses transitioning to more sustainable water sources. The tool is designed for users with a hydrological background who are familiar with Python scripting, as it is a Python-based package and not a fully polished tool (though this could change in the future).

It is also important to note that the tool should not be used to promote rainwater harvesting as a universal solution. In some areas, local climate or building types may require different approaches, so the model should provide an honest evaluation of rainwater harvesting's potential effectiveness.
## Unique approach to modelling rainwater harvesting
The methodology behind the tool uses a statistical approach to quantify both water demand and supply. The tool is a Python package that includes scripts for hydrological calculations and analysis.

What sets this tool apart is its ability to model not just a single “average” year but the variability of reservoir dynamics. The definition of an “average” or “dry” year (the model input) is also not the most interesting figure – it’s about the reliability of using rainwater (the model output). Rather than assuming a fixed demand pattern, it explores different reservoir sizes and demand patterns to determine the minimum required tank size for a given service level. This method accounts for the likelihood of a reservoir being too small or over-sized, providing more realistic predictions. The model also allows users to view individual runs, including climate data, reservoir capacity, overflow, and deficit information.

To make the model broadly applicable, it is designed to work with any hard surface (e.g., a roof) and any type of reservoir. Both reservoir capacity and demand are measured in units of height (mm), making hydrological calculations more straightforward. Results are then converted back into their original units (e.g., cubic meters).



# Quick-Start and Example Use Cases
## Getting started
For basic usage of the tool, follow these steps:
1. Download or clone the repository containing the model to your local machine.
2. Install the model using the instructions provided in the Installation Guide section of this document.
3. Create a project folder (e.g., for a case study in India) and provide the required input data, including local climate data (precipitation and potential evapotranspiration), reservoir characteristics, and area-specific details. If local climate data is unavailable, users can use the [hydromt_uwbm](https://github.com/Deltares-research/hydromt_uwbm) plugin for the [hydromt framework](https://deltares.github.io/hydromt/latest/) to generate timeseries based on global climate datasets such as ERA5.
4. Run the model for your specific case study by using the installed model from the project directory. Model results, figures, and statistics will be saved in the /output folder in the project directory.

## Use cases
*Disclaimer: This model has not been thoroughly tested. For developers, not all tests are operational.*
### Example: single model run
If the user wants to run a single simulation, to instance to assess the dynamics of a particular irrigation system, the first step is to correctly set up the **setup.toml** file with general characteristics.

The receiving surface area has a size of 60 m2 and the interception capacity of the receiving surface is estimated to be 2 mm. The *specific* reservoir capacity is estimated to be 15 mm (900 liters / 60 m2 surface area = 15 mm). The **setup.toml** file should contain the following parameters:
```
srf_area = 60
int_cap = 2
reservoir_cap = 15
```

The next step is to initiate the model with the given parameters as arguments:
- **root**: Root folder for the model case.
- **name**: Unique name to identify model runs.
- **mode**: Run mode. In this case: “single”.
- **setup_fn**: Path to setup.toml file that specifies generic characteristics of this single model run.
- **forcing_fn**: Path to forcing.csv file containing precipitation and PET timeseries.
- **demand_fn**: Path to demand.csv file.
- **reservoir_initial_state**: To prevent a dry reservoir at the start of the simulation, the reservoir is set to be filled to 75% of its capacity.
- **timestep**: Model timestep is 1 day (86400 seconds).
- **unit**: Calculation unit for the reservoir size and demand per timestep, in this case ‘mm’.

```python
import RWHmodel

model = RWHmodel.Model(
        root = r"C:\Users\example_case",
        name = "irrigation_simulation",
        mode = "single",
        setup_fn = r"C:\Users\example_case\input\setup.toml",
        forcing_fn = r"C:\Users\example_case\input\forcing_daily.csv",
        demand_fn = r"C:\Users\example_case\input\demand_daily.csv",
        reservoir_initial_state = 0.75,
        timestep = 86400,
        unit = "mm"
)
```

Call the run function using the following command:
```python
model.run()
```
Model run results are stored in: **root/output/runs**

Plot model results using the following command:
```python
model.plot(plot_type="run")
```
Figure plots are stored in: **root/output/figures**
### Example: using user-provided runoff

If the user wants to run the model using observed or externally simulated runoff, instead of calculating it from net precipitation, they can provide a runoff file. Suppose we have a runoff timeseries file **user_runoff.csv** with the following format:

```
date,runoff
2000-01-01,1.2
2000-01-02,0.0
2000-01-03,0.8
...
```

The runoff column must have the same number of timesteps as the forcing file. The `date` column is optional if using a pandas Series directly.

The model can then be initiated as follows:

```python
import pandas as pd
import RWHmodel

# Load user runoff
user_runoff = pd.read_csv(r"C:\Users\example_case\input\user_runoff.csv")["runoff"]

# Initiate model with user-supplied runoff
model = RWHmodel.Model(
        root = r"C:\Users\example_case",
        name = "irrigation_user_runoff",
        mode = "single",
        setup_fn = r"C:\Users\example_case\input\setup.toml",
        forcing_fn = r"C:\Users\example_case\input\forcing_daily.csv",
        demand_fn = r"C:\Users\example_case\input\demand_daily.csv",
        runoff_source = "user",       # Use user-provided runoff
        user_runoff = user_runoff,    # Supply the runoff Series
        reservoir_initial_state = 0.75,
        timestep = 86400,
        unit = "mm"
)
```

Call the run function:

```python
model.run()
```

Results are stored in: **root/output/runs**

Plot the results:

```python
model.plot(plot_type="run")
```

Figures are stored in: **root/output/figures**
### Example: batch model run
A batch run allows the user to run a “scenario-space” of possible demand patterns and reservoir sizes. Individual model results can be saved to .csv; the ‘coverage’ table showing how much of the *total* water demand can be supplied for each reservoir-demand combination is saved to .csv automatically.

In order to run a batch run, for instance to assess the potential of rainwater harvesting for an industrial complex, the first step is to correctly set up the **setup.toml** file with general characteristics.

The threshold (‘norm’) for which the industry has to obtain water from other sources than the rainwater harvesting tank is set at 48 consecutive hours. The probabilities of exceeding the threshold are requested for the following return periods: 1, 2, 5, 10, 20, 50 and 100 years. The receiving surface area has a size of 3500 m2 the interception capacity of the receiving surface is estimated to be 3 mm. The **setup.toml** file should contain the following parameters:
```
threshold = 48
T_return_list = [1,2,5,10,20,50,100]
srf_area = 3500
int_cap = 3
```

The next step is to initiate the model with the given parameters as arguments:
- **root**: Root folder for the model case.
- **name**: Unique name to identify model runs.
- **mode**: Run mode. In this case: “batch”.
- **setup_fn**: Path to setup.toml file that specifies generic characteristics of this single model run.
- **forcing_fn**: Path to forcing.csv file containing precipitation and PET timeseries.
- **demand_fn**: Range of hourly demands to be considered, minimum is 0.1 m3/h, maximum is 1 m3/h, with 150 equal steps in between.
- **reservoir_range**: Range of reservoir sizes to be considered, minimum is 10 m3, maximum is 1000 m3, with 150 equal steps in between.
- **reservoir_initial_state**: To prevent a dry reservoir at the start of the simulation, the reservoir is set to be filled to 75% of its capacity.
- **timestep**: Model timestep is 1 hour (3600 seconds).
- **unit**: Calculation unit for the reservoir size and demand per timestep, in this case ‘m3’.

```python
import RWHmodel

model = RWHmodel.Model(
        root = r"C:\Users\example_case",
        name = "industrial_reuse",
        mode = "batch",
        setup_fn = r"C:\Users\example_case\input\setup_batch_run.toml",
        forcing_fn = r"C:\Users\example_case\input\forcing_hourly.csv",
        demand_fn = [0.1, 1, 150],
        reservoir_range = [10, 1000, 150],
        reservoir_initial_state = 0.75,
        timestep = 3600,
        unit = "m3"
)
```

For this exercise, we are interested in the amount of consecutive days that the reservoir does not suffice for the given demand. This is the ‘threshold’ specified in the **setup_batch_run.toml** file.

Call the batch run function using the following command,  :
```python
model.batch_run(method="consecutive_timesteps")
```
The ‘coverage summary’ results are stored in: **root/output/runs/summary**
The ‘system characteristics’, using an extreme value approach and the specified return period,  are stored in: **root/output/runs/statistics**

Plot the coverage summary and system curve using the following commands:
```python
model.plot(plot_type="run_coverage")
model.plot(plot_type="system_curve")
```
To plot the and potential savings curves, call the plot function and specify the typologies to be represented in the graph. This can be the main typology outlined in the  setup_batch_run.toml file, but the user can experiment with different total demands and surface areas.
```python
model.plot(
        plot_type="saving_curve",
        T_return_list=[1,2,5,10],
        unit='m3',
        typologies_name = ['Industry hall'],
        typologies_demand = [1500],
        typologies_area = [3500],
        reservoir_max = 1000
    )
```

Figure plots are stored in: **root/output/figures**
### Plotting
The tool provides basic plotting of input forcing, model results, and some analyses. Based on the saved model output, users are free to tailor plotting functions according to their needs.

After initiating a **model** instance, users can call various plotting functions:
- **meteo**: plotting the input forcing (precipitation and potential evapotranspiration (PET)).
- **run**: Basic timeseries plot of a single run, showing the reservoir storage, reservoir overflow, demand and deficit.
- **run_coverage**: For batch-run results. Shows for the calculated “scenario-space” how much of the *total* water demand can be supplied using the reservoir during the simulated time period.
- **run_system_curve**: For batch-run results. Shows for the selected “exceedance threshold” and associated return time periods which water demand can be supplied from the reservoir.
- **run_saving_curve**: For batch-run results. Shows for the selected  “exceedance threshold” and associated return time periods what percentage of water savings can be achieved with various reservoir sizes.

**Plotting meteo**

To plot the entire forcing timeseries:
```python
model.plot(plot_type="meteo")
```
To plot a clipped part of the forcing timeseries:
```python
model.plot(plot_type="meteo", t_start="2000-06-01", t_end="2000-05-31")
```
To plot the forcing timeseries, monthly aggregated:
```python
model.plot(plot_type="meteo", aggregate=True)
```

**Plotting run results**

To plot the full run results:
```python
model.plot(plot_type="run")
```
To plot a clipped part of the run results:
```python
model.plot(plot_type="run", t_start="2000-06-01", t_end="2000-05-31")
```

**Plotting run coverage**

To plot the coverage of reservoir(s) compared to demand(s):
```python
model.plot(plot_type="run_coverage")
```

**Plotting system curve**

To plot the “system-curve”:
```python
model.plot(plot_type="system_curve")
```
To plot the “system-curve”, using a specified return period list and enabling raw data plotting for validation purposes:
```python
model.plot(plot_type="system_curve", validation=True, T_return_list=[1,2,5,10,20])
```
**Plotting savings-curve**

To plot the “savings-curve” for a selection of typologies:
```python
model.plot(
        plot_type="saving_curve",
        typologies_name = ['Apartment', 'Townhouse', 'Villa'],
        typologies_demand = [6912, 2267, 892],
        typologies_area = [850, 48, 122],
)
```
To plot the “savings-curve”, using specified return period list, plotting unit as m3, maximum reservoir capacity and (vertical) ambition lines for 15%, 30% and 65% reduction:
```python
model.plot(
        plot_type="saving_curve",
        T_return_list=[1,2,5,10],
        unit='m3',
        reservoir_max=20,
        ambitions=[15,30,65],
        typologies_name = ['Apartment', 'Townhouse', 'Villa'],
        typologies_demand = [6912, 2267, 892],
        typologies_area = [850, 48, 122],
)
```

## Function documentation
This section describes all mandatory and optional user arguments for the Python interface of the model.
### Initializing the model
The model is initialized by calling the **.Model** class within the RWHmodel package.
```python
import RWHmodel
model = RWHmodel.Model()
```

The initialization function takes the following arguments:
- **root**: Required. Root folder for the model case.
- **name**: Required. Unique name to identify model runs.
- **mode**: Required. Run mode of the model. Can be either ‘single’ or ‘batch’.
- **setup_fn**: Required. Path to setup.toml file that specifies generic characteristics. Single runs and batch runs require different parameters in the setup.toml file in order to run.
- **forcing_fn**: Required. Path to forcing.csv file containing precipitation and PET timeseries.
- **demand_fn**: Required. Choose between 1) path to demand.csv file, 2) singular value [unit/timestep] or 3) list with Numpy.linspace arguments (for batch run).
- **demand_transform**: Optional, default is False. Boolean to specify whether non-timeseries demand argument should be transformed with a generic, sinusoid seasonal dynamic.
- **reservoir_range**: Required for batch run only. List with Numpy.linspace arguments to specify range of reservoirs to be modelled.
- **reservoir_initial_state**: Optional, default is 0. Fraction of the reservoir capacity that is filled on the first timestep.
- **timestep**: Optional, default is derived from forcing timeseries. Timestep length in seconds: 3600 or 86400 seconds.
- **t_start**: Optional, default is first timestep of forcing timeseries. Start time to clip the timeseries for modelling.
- **t_end**: Optional, default is final timestep of forcing timeseries. End time to clip the timeseries for modelling.
- **unit**: Optional, default is ‘mm’. Calculation unit for the reservoir size and demand per timestep: ‘mm’ or ‘m3’. Ensure that both reservoir and demand timeseries are in the same and correct unit.
- **runoff_source**: Optional, default is "model". Determines whether the runoff is calculated internally using the HydroModel ("model") or provided by the user ("user").
- **user_runoff**: Optional, default is None. User-supplied runoff timeseries (pandas Series, NumPy array, or list) that must match the length of the forcing data. Only used if `runoff_source="user"`.

### Single run
To perform a single model run, call the **run** function.
```python
model.run()
```

The **run** function takes the following argument:
- **save**: Optional, default = True. Boolean to specify whether the run output should be saved to .csv.

The function saves the following output:
- **run timeseries**: The model run timeseries contains the date range, stored volume in the reservoir, reservoir overflow, demand, demand deficit and the timesteps where there was a deficit (all in their original unit). The timeseries is saved as .csv format in: **root/output/runs/summary/{*name*}_run_res-cap={*reservoir capacity in mm*}_yr-dem={*yearly demand in mm*}.csv**
### Batch run
To perform a batch model run, call the **batch_run** function.
```python
model.batch_run()
```

The **batch_run** function takes the following arguments:
- **method**: Optional, default = None. Specifies the desired threshold type to be considered in the statistical calculations. Choose between 1) ‘consecutive_days’ or 2) ‘total_days’.
- **log**: Optional, default = False. Boolean to toggle terminal text printing for each iteration.
- **save**: Optional, default = False. Boolean to toggle individual saving of every model iteration (to .csv).

The function saves the following output:
- **coverage summary**: table showing the fraction of the *total* water demand that can be supplied for each reservoir-demand combination. Reservoir and demand values are saved in mm and mm/year, respectively. The table is saved as .csv format in: **root/output/runs/summary/{*name*}_batch_run_coverage_summary.csv**
- **system characteristics**: table in indicating the (maximum) yearly demand [mm/year] that can be sustained for various reservoir sizes [mm] for the specified return periods and threshold value. These values are obtained using the Peak over Threshold approach. The table is saved as .csv format in: **root/output/runs/statistics/{*name*}_batch_run.csv**.
# Model Documentation
The computational model used consists of two sub-models that operate sequentially: a **rainfall-runoff model** and a **reservoir model**. The rainfall-runoff model firstly calculates the potential runoff from the specified surface. Afterwards, the reservoir model iterates over all timesteps to simulate the reservoir capacity given all inputs and outputs.

The hydrological models calculate all fluxes in a unit of height (mm) per timestep. If other unit formats (such as m3/timestep) are supplied, they are first converted to mm/timestep. If either demand or reservoir capacity are expressed in unit of height, they are called *specific* demand or reservoir capacity.
## Model 1: Rainfall-Runoff Model
To determine the runoff from the specified surface to the reservoir, a simplified runoff model is used. The model concepts are based on the [Urban Water Balance Model](https://publicwiki.deltares.nl/display/AST/Urban+Water+balance+model), developed by Deltares.

The model takes the provided forcing file containing precipitation and potential evaporation (PET) data as input. For each time step, the model simulates the amount of precipitation, interception losses (emptied only by evaporation), and the amount of runoff that ultimately flows into the reservoir. In this model, interception storage is defined as a loss component. The first portion of precipitation remains on the roof and will evaporate if possible. If space is available in the interception storage during a subsequent rain event, it will be filled first before runoff occurs. The precipitation captured as interception will never flow into the reservoir but will only evaporate.

The general hydrological process is as follows:

***runoff = precipitation – evaporation – interception***

The available rainfall-runoff model currently only supports simple, hard surface areas for which the runoff process can be represented by interception losses. As all calculations are made in unit of height, the built-in rainfall-runoff model can be easily substituted by a more advanced rainfall-runoff model, for instance one that simulates runoff from a green roof, blue (polder) roof, or hilly terrain. The current limitation is that the rainfall-runoff model runs entirely before the reservoir model (model 2), hence there can be no interaction between the reservoir and the potential inflow. This could be a limitation for modelling a blue (polder) roof, where the runoff towards the reservoir would be influenced by the actual reservoir storage capacity.

Assumptions for the Rainfall-Runoff model:
- It is assumed that 100% of the runoff from the specified surface drains into the reservoir. If this is not realistic, the total surface area size can be replaced by a smaller proxy surface area to compensate for non-contributing runoff.
- It is assumed that there are no losses in the transport of runoff from the surface to the reservoir.
- Interception capacity is assumed to be constant.
- Since water quality is not within the scope of this model, it is assumed that all harvested rainwater can be used. In practice, there may be a preference to discard the first few millimeters of rainfall (first flush) to prevent the most polluted water from entering the reservoir.
## Model 2: Reservoir Model
The reservoir model takes as input the runoff (inflow) from Model 1 (Rainfall-Runoff Model) and the water demand (outflow). The model outputs are the reservoir capacity, reservoir overflow, supplied demand and demand deficit.  The initial state of the reservoir can be specified when running the model.

The Reservoir model iterates for each timestep the steps outlined below.

**Step 1: Update Reservoir Storage**

$$
S' = S + R - D
$$

Where:
- S′ = updated reservoir storage
- S = previous reservoir storage
- R = runoff (inflow)
- D = water demand

**Step 2: Check for Overflow**

$$
O = \max(0, S' - C)
$$

$$
S' = \min(S', C)
$$

Where:
- O = reservoir overflow
- C = reservoir capacity

**Step 3: Check for Deficit**

$$
\text{If } S' > D, \quad \text{Deficit} = 0
$$

$$
\text{If } 0 < S' \leq D, \quad \text{Deficit} = D - S'
$$

$$
\text{If } S' \leq 0, \quad \text{Deficit} = D, \quad S' = 0
$$

Where:
- Deficit represents the unmet water demand.

Assumptions for the Reservoir model:
- There are no losses from the reservoir, other than the specified demand. For instance, there is no evaporation from the reservoir.
- The model assumes that there is a backup source of water from water mains. The demand is not impacted by the reservoir state – there is no feedback which results in reduced or adjusted demand when the reservoir falls dry. However, the model tracks the water deficit.
## Statistical analysis
The purpose of this modelling package is to assess the potential of rainwater harvesting, given local climatological, physical and demand characteristics. Hence, analyzing a range of modelling results (*scenario space*) is a key part of the methodology.
The potential of rainwater harvesting is determined using a Peak Over Threshold (POT) methodology. In order to apply this method, the user needs to provide three additional inputs compared to a regular model run:
- **method**: The type of threshold the user is interested in. Currently, the model supports both *consecutive timesteps* and *total timesteps*. When using the “consecutive timesteps” method, the model performs an analysis on the amount of consecutive timesteps the reservoir failed to meet the required demand. When using the “total timesteps” method, the model counts the total timesteps per year that the reservoir failed to meet the required demand within the modelling timeframe.
- **threshold**: The threshold (number of timesteps) for which the POT method should be applied. It can be regarded as a ‘norm’ – the minimum level of service.
- **T_return_list**: A list of return times for which the threshold is exceeded. This is used for assessing the exceedance probability.

Assumptions for the analysis:
- By definition, the threshold in this model is defined as an event at which the reservoir fails to fully supply the required demand (storage = 0). It is not possible to implement a threshold that occurs when the stored volume falls beneath a certain percentage of the reservoir, such as 10%.
- This study analyzes consecutive days when the reservoir is empty, but in reality, a minimum threshold (e.g., 5%) is maintained with mains water to ensure availability, while avoiding complete refilling to preserve rainwater storage capacity.

The steps below outline the approach for the POT analysis.
### Step 1 – tracking reservoir failures
The reservoir model simulates the water storage in the reservoir as a continuous time series based on the provided inflow and demand range. The combined list of demands and reservoir sizes form the scenario space. During the simulation, the model tracks when the reservoir fails to supply the required demand.
### Step 2 – identifying drought events
The model analyses all instances where the reservoir was empty and calculates the number of consecutive or total timesteps the reservoir failed to supply the required demand (based on the specified *method*). These are the "drought events"—periods when mains water must be used or when demand is no longer possible.
### Step 3 – determining return periods
The model ranks the drought events from longest to shortest: the longest period without water in the reservoir is at the top, followed by the second longest, etc. Using the exponential function below, the return period for all drought events is determined.

$$
y = a \cdot \ln(x) + b
$$
### Step 4 – determining minimum reservoir size
For each return period (list specified earlier) and demand scenario, the model determines the minimum reservoir size required to prevent exceeding the threshold of timesteps where the reservoir does not suffice.
### Step 5 – generating system curves
Plotting the minimum reservoir size versus the demand yields "system curves"—the relationship between water demand and the minimum required reservoir size (both in unit of height).

The system curves show characteristics of an asymptote: to accommodate an ever-increasing specific consumption, an infinitely large reservoir would be required. An asymptotic curve can be fitted through the data points to describe the system curve. The formula below generally first the data closely.

$$
y = \frac{a \cdot x^n}{x^n + b}
$$

As an asymptotic function, the supported demand (y) approaches a limiting value as x increases. At very small reservoir capacities (x→0), the available water y is also small. At larger reservoir capacities (x→∞), the available water y reaches a maximum limit (a), meaning the reservoir is full and additional rain cannot increase storage beyond its capacity. For moderate rainfall, the function captures how water availability increases with rainfall but at a diminishing rate due to storage constraints or losses (evaporation, overflow).
