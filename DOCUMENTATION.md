
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



# Quick-start and example Use Cases
## Getting started
For basic usage of the tool, follow these steps:
1. Download or clone the repository containing the model to your local machine.
2. Install the model using the instructions provided in the Installation Guide section of this document.
3. Create a project folder (e.g., for a case study in India) and provide the required input data, including local climate data (precipitation and potential evapotranspiration), reservoir characteristics, and area-specific details. If local climate data is unavailable, users can use the ['hydromt_uwbm'](https://github.com/Deltares-research/hydromt_uwbm) plugin for the [hydromt framework](https://deltares.github.io/hydromt/latest/) to generate timeseries based on global climate datasets such as ERA5.
4. Run the model for your specific case study by using the installed model from the project directory. Model results, figures, and statistics will be saved in the /output folder in the project directory.

## Use cases
### Single, simple model run
If the user wants to run a single simulation, to instance to assess the performance of a particular system, the first step is to initiate the model:
```
import RWHmodel

model = RWHmodel.Model(
        root = r"C:\Users\example_case",
        name = "example_case",
        setup_fn = r"C:\Users\example_case\input\setup_single_run.toml",
        forcing_fn = r"C:\Users\example_case\input\forcing_test.csv",
        demand_fn = r"C:\Users\example_case\input\demand_test.csv",
        demand_transform = False,
        reservoir_initial_state = 0.75,
        timestep = 86400,
        unit = "mm"
        )
```
Arguments:
- **root**: Root folder for the model case.
- **name**: Unique name to identify model runs.
- **setup_fn**: Path to setup.toml file that specifies generic characteristics.
- **forcing_fn**: Path to forcing.csv file containing precipitation and PET timeseries.
- **demand_fn**: Path to demand.csv file, singular value or list with Numpy.linspace arguments.
- **demand_transform**: Boolean to specify whether non-timeseries demand argument should be transformed with seasonal dynamics.
- **reservoir_range**: List with Numpy.linspace arguments to specify range of reservoirs to be modelled.
- **reservoir_initial_state**: Fraction of the reservoir capacity that is filled on the first timestep.
- **timestep**: Timestep length in seconds: 3600 or 86400 seconds (optional, default is taken from forcing timeseries).
- **t_start**: Start time to clip the timeseries for modelling (optional).
- **t_end**: End time to clip the timeseries for modelling (optional).
- **unit**: Calculation unit for the reservoir size and demand per timestep: ‘mm’ or ‘m3’ (optional, default = ‘mm’). Ensure that both reservoir and demand timeseries are in the same and correct unit.


Call the run function using the following command:
```
model.run()
```
Arguments:
- **save**: Boolean to specify whether the run output should be saved to .csv (optional, default = True)

Model run results are stored in: **root/output/runs**


### Batch model run
A batch run allows the user to run a “scenario-space” of possible demand patterns and reservoir sizes. 
```
model = RWHmodel.Model(
        root = r"C:\Users\example_case",
        name = "example_case",
        setup_fn = r"C:\Users\example_case\input\setup_batch_run.toml",
        forcing_fn = r"C:\Users\example_case\input\forcing_test.csv",
        demand_fn = [0.5, 5, 10],
        demand_transform = False,
        reservoir_range = [10, 200, 10],
        reservoir_initial_state = 0.75,
        timestep = 86400,
        unit = "mm"
        )
```


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


# Model Description

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



 

