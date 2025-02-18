
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
### Example: single model run
If the user wants to run a single simulation, to instance to assess the dynamics of a particular irrigation system, the first step is to correctly set up the **setup.toml** file with general characteristics.

The next step is to initiate the model with the given parameters as arguments:
- **root**: Root folder for the model case.
- **name**: Unique name to identify model runs.
- **setup_fn**: Path to setup.toml file that specifies generic characteristics of this single model run.
- **forcing_fn**: Path to forcing.csv file containing precipitation and PET timeseries.
- **demand_fn**: Path to demand.csv file.
- **reservoir_initial_state**: To prevent a dry reservoir at the start of the simulation, the reservoir is set to be filled to 75% of its capacity.
- **timestep**: Model timestep is 1 day (86400 seconds).
- **unit**: Calculation unit for the reservoir size and demand per timestep, in this case ‘mm’.

```
import RWHmodel

model = RWHmodel.Model(
        root = r"C:\Users\example_case",
        name = "irrigation_simulation",
        setup_fn = r"C:\Users\example_case\input\setup_single_run.toml",
        forcing_fn = r"C:\Users\example_case\input\forcing_daily.csv",
        demand_fn = r"C:\Users\example_case\input\demand_daily.csv",
        reservoir_initial_state = 0.75,
        timestep = 86400,
        unit = "mm"
        )
```

Call the run function using the following command:
```
model.run()
```
Model run results are stored in: **root/output/runs**

Plot model results using the following command:
```
model.plot(plot_type="run")
```
Figure plots are stored in: **root/output/figures**
### Example: batch model run
A batch run allows the user to run a “scenario-space” of possible demand patterns and reservoir sizes. Individual model results can be saved to .csv; the ‘coverage’ table showing how much of the *total* water demand can be supplied for each reservoir-demand combination is saved to .csv automatically.

In order to run a batch run, for instance to assess the potential of rainwater harvesting for an industrial complex, the first step is to correctly set up the **setup.toml** file with general characteristics.

The next step is to initiate the model with the given parameters as arguments:
- **root**: Root folder for the model case.
- **name**: Unique name to identify model runs.
- **setup_fn**: Path to setup.toml file that specifies generic characteristics of this single model run.
- **forcing_fn**: Path to forcing.csv file containing precipitation and PET timeseries.
- **demand_fn**: Range of hourly demands to be considered, minimum is 0.1 m3/h, maximum is 1 m3/h, with 150 equal steps in between.
- **reservoir_range**: Range of reservoir sizes to be considered, minimum is 10 m3, maximum is 1000 m3, with 150 equal steps in between.
- **reservoir_initial_state**: To prevent a dry reservoir at the start of the simulation, the reservoir is set to be filled to 75% of its capacity.
- **timestep**: Model timestep is 1 hour (3600 seconds).
- **unit**: Calculation unit for the reservoir size and demand per timestep, in this case ‘m3’.

```
import RWHmodel

model = RWHmodel.Model(
        root = r"C:\Users\example_case",
        name = "industrial_reuse",
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
```
model.batch_run(method=’consecutive_timesteps’)
```
The ‘coverage summary’ results are stored in: **root/output/runs/summary**
The ‘system characteristics’, using an extreme value approach and the specified return period,  are stored in: **root/output/runs/statistics**

Plot the coverage summary, system curve and potential savings curve using the following commands:
```
model.plot(plot_type="run_coverage")
model.plot(plot_type="system_curve")
model.plot(plot_type="saving_curve")
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

To plot the entire forcing timeseries:
```
model.plot(plot_type="meteo")
```
To plot a clipped part of the forcing timeseries:
```
model.plot(plot_type="meteo", t_start="2000-06-01", t_end="2000-05-31")
```
To plot the forcing timeseries, monthly aggregated:
```
model.plot(plot_type="meteo", aggregate=True)
```

To plot the full run results:
```
model.plot(plot_type="run")
```
To plot a clipped part of the run results:
```
model.plot(plot_type="run", t_start="2000-06-01", t_end="2000-05-31")
```

To plot the coverage of reservoir(s) compared to demand(s):
```
model.plot(plot_type="run_coverage")
```

To plot the “system-curve”:
```
model.plot(plot_type="system_curve")
```
To plot the “system-curve”, using a specified return period list and enabling raw data plotting for validation purposes:
```
model.plot(plot_type="system_curve", validation=True, T_return_list=[1,2,5,10,20])
```

To plot the “savings-curve”:
```
model.plot(plot_type="saving_curve")
```
To plot the “savings-curve”, using specified return period list, plotting unit as m3, maximum reservoir capacity and (vertical) ambition lines for 15%, 30% and 65% reduction:
```
model.plot(plot_type="saving_curve", T_return_list=[1,2,5,10], unit='m3', reservoir_max=20, ambitions=[15, 30, 65])
```

## Function documentation
This section describes all mandatory and optional user arguments for the Python interface of the model.
### Initializing the model
The model is initialized by calling the **.Model** class within the RWHmodel package.
```
import RWHmodel
model = RWHmodel.Model()
```

The initialization function takes the following arguments:
- **root**: Required. Root folder for the model case.
- **name**: Required. Unique name to identify model runs.
- **setup_fn**: Required. Path to setup.toml file that specifies generic characteristics. Single runs and batch runs require different parameters in the setup.toml file in order to run.
- **forcing_fn**: Required. Path to forcing.csv file containing precipitation and PET timeseries.
- **demand_fn**: Required. Choose between 1) path to demand.csv file, 2) singular value [unit/timestep] or 3) list with Numpy.linspace arguments (for batch run).
- **demand_transform**: Optional, default is False. Boolean to specify whether non-timeseries demand argument should be transformed with a generic, sinusoid seasonal dynamic.
- **reservoir_range**: Required for batch run only. List with Numpy.linspace arguments to specify range of reservoirs to be modelled.
- **reservoir_initial_state**: Optional, default is 0. Fraction of the reservoir capacity that is filled on the first timestep.
- **timestep**: Optional, default is derived from the forcing timeseries. Timestep length in seconds: 3600 or 86400 seconds.
- **t_start**: Optional, default is first timestep of forcing timeseries. Start time to clip the timeseries for modelling.
- **t_end**: Optional, default is final timestep of forcing timeseries. End time to clip the timeseries for modelling.
- **unit**: Optional, default is ‘mm’. Calculation unit for the reservoir size and demand per timestep: ‘mm’ or ‘m3’. Ensure that both reservoir and demand timeseries are in the same and correct unit.
### Single run
To perform a single model run, call the **run** function.
```
model.run()
```

The **run** function takes the following argument:
- **save**: Boolean to specify whether the run output should be saved to .csv (optional, default = True)
### Batch run
To perform a batch model run, call the **batch_run** function.
```
model.batch_run()
```

The **batch_run** function takes the following arguments:
- **method**: Optional, default = None. Specifies the desired threshold type to be considered in the statistical calculations. Choose between 1) ‘consecutive_days’ or 2) ‘total_days’.
- **log**: Optional, default = False. Boolean to toggle terminal text printing for each iteration.
- **save**: Optional, default = False. Boolean to toggle individual saving of every model iteration (to .csv).

# Model Documentation

The computational model used consists of two sub-models that operate sequentially in the process.
## Model 1: Runoff Model
To determine the runoff from the roof to the reservoir, a simplified runoff model was used. The model concepts are based on the [Urban Water Balance Model](https://publicwiki.deltares.nl/display/AST/Urban+Water+balance+model), developed by Deltares.

Based on a long series of precipitation and evaporation data, the model calculates, for each time step (1 day), how much precipitation falls on the roof, how much remains due to interception storage, and the amount of runoff that ultimately flows into the reservoir. The general hydrological process is as follows:

***runoff = precipitation – evaporation – interception***

In this model, interception storage is defined as a loss component. The first portion of precipitation remains on the roof and will evaporate if possible. If space is available in the interception storage during a subsequent rain event, it will be filled first before runoff occurs. The precipitation captured as interception will never flow into the reservoir but will only evaporate.

Several assumptions have been made in this model to simplify the calculation:
- It is assumed that 100% of the roof surface drains into the rainwater reservoir. In reality, this is often not the case.
- It is assumed that there are no losses in the transport of rainwater from the roof to the reservoir.
- A standard interception storage of 1 mm is used. Other roof types, such as flat or green roofs, may have a higher interception capacity than 1 mm.
- Since water quality is not a primary focus of this study, it is assumed that all harvested rainwater can be used. In practice, there may be a preference to discard the first few millimeters of rainfall (first flush) to prevent the most polluted water from entering the reservoir.
## Model 2: Reservoir Model
The reservoir model takes as input the runoff from Model 1 (Runoff Model) and the daily varying water demand. The output is a statistical analysis of reservoir dynamics. The model follows these five steps for each scenario:

Step 1
The reservoir model simulates the water storage in the reservoir as a continuous time series based on the given precipitation and evaporation data. During the simulation, the model tracks when the reservoir is empty. To avoid an initial dry period at the beginning of the simulation, it is assumed that the reservoir is 25% full at the start.

Step 2
The model analyzes all instances where the reservoir was empty and calculates the number of consecutive days when there was insufficient water to meet the daily demand. These are the "drought events"—periods when mains water must be used.

Step 3
The model ranks the drought events from longest to shortest: the longest period without water in the reservoir is at the top, followed by the second longest, etc. Using an exponential function, the return period for all drought events is determined. In this report, we are interested in the return periods T1, T2, T5, and T10.

Step 4
For each return period and consumption scenario, the model determines the minimum reservoir size required to prevent exceeding the predefined maximum number of consecutive days with an empty reservoir (3, 7, or 14 days). This principle is illustrated in Figure 13.
Figure 13: Indicative diagram for selecting the smallest possible reservoir size based on the maximum number of consecutive dry days for different consumption levels (example: maximum of 3 consecutive dry days).

Step 5
Plotting the minimum reservoir size versus specific drinking water consumption yields "system curves"—the relationship between specific drinking water demand and the minimum required reservoir size. These data points form an asymptote: to accommodate an ever-increasing specific consumption, an infinitely large reservoir would be required. An asymptotic curve can be fitted through the data points to describe the system curve (see Appendix 7D).

An important note is that this study performs a statistical analysis of consecutive days when the reservoir is empty. In reality, the reservoir will never be entirely empty. When the water level falls below a certain minimum threshold (e.g., 5%), mains water will be used to maintain a constant level, ensuring that users always have access to water. However, the reservoir is not completely refilled with mains water, as this would reduce the efficiency of rainwater reuse. A reservoir filled with mains water would have no space left to capture a significant rainfall event.



 

