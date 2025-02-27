""" Plotting functions. """

from RWHmodel.analysis import func_system_curve, func_system_curve_inv, func_fitting
from RWHmodel.utils import convert_mm_to_m3, colloquial_date_text

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

######### COLOR MAPS #########
# Default color map
cmap = ['#080c80', '#0ebbf0', '#00b389', '#ff960d', 
        '#e63946', '#6a4c93', '#f4a261', '#2a9d8f']
# Gradient color maps
cmap_g1 = ['#080c80', '#5f6199', '#9395b9', '#c9cadc']
cmap_g2 = ['#00b389', '#5bc1a4', '#90d1c0', '#c6e7de']
cmap_g3 = ['#ff960d', '#f8b05b', '#fbcb8f', '#fde4c6']
cmap_g4 = ['#0ebbf0', '#62ccf1', '#96d9f2', '#c9ebf7']
##############################

def plot_meteo(
        root,
        name,
        forcing_fn,
        t_start,
        t_end,
        aggregate = False
    ) -> None:
    """
    Meteo plotting funtion.
    
    Parameters
    ----------
    root : str
        Folder location of model root.
    name : str
        Unique name of the model instance.
    forcing_fn : pd.DataFrame
        Pandas DataFrame containing forcing timeseries data.
    t_start : str
        Start time of plotting interval.
    t_end : str
        End time of plotting interval.
    aggregate : bool, optional
        Boolean to specify whether meteo plotting should be
        aggregrated to monthly values.
    """
    df = forcing_fn
    # Clip DataFrame based on t_start and t_end
    mask = (df.index > t_start) & (df.index <= t_end)
    df = df.loc[mask]
    # Obtain number of years
    num_years = (df.index.max()-df.index.min())/(np.timedelta64(1, 'W')*52)
    # Resample in case of mean monthly sum aggregation
    if aggregate:
        df = df.groupby(df.index.month).sum()/num_years
        plot_name = "mean_monthly_sum"
    else:
        plot_name = "full"
    
    # Create plot
    fig, ax1 = plt.subplots(1, figsize=(14,6))
    ax2 = ax1.twinx()
    
    # Drawing lines and filled area
    ax2.plot(df.index, df['precip'], linewidth=1, linestyle='-',
             label = 'Precipitation', color='#080c80') 
    ax1.fill_between(df.index, y1=df['pet'], y2=0, label = 'Potential Evapotranspiration' , color='#c6c6c6')
    
    # Axes labels
    ax1.set_ylabel('Potential Evaporation [mm]')
    ax1.set_xlabel('Date')
    ax2.set_ylabel('Precipitation [mm]')
    
    # Axes limits
    ax1.set_ylim([0, df['pet'].max()])
    ax2.set_ylim([0, df['precip'].max()])
    if aggregate:
        ax1.set_xticks(np.arange(1,13))
        ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    else:
        ax1.set_xlim([t_start, t_end])
    
    # Layout and grid
    ax1.spines.top.set_visible(False)
    ax2.spines.top.set_visible(False)
    plt.grid(visible=True, which="major", color="black", linestyle="-", alpha=0.2)
    plt.grid(visible=True, which="minor", color="black", linestyle="-", alpha=0.1)
    
    # Legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    
    # Export
    fig.savefig(f"{root}/output/figures/{name}_forcing_{plot_name}_{t_start.year}_{t_start.month}-{t_end.year}_{t_end.month}.png", dpi=300, bbox_inches='tight')
    #fig.savefig(f"{root}/output/figures/{name}_forcing_{plot_name}_{t_start.year}_{t_start.month}-{t_end.year}_{t_end.month}.svg", dpi=300, bbox_inches='tight')

def plot_run(
        root,
        name,
        run_fn, # Path to run output file
        unit,
        t_start,
        t_end,
        reservoir_cap,
        yearly_demand
    ) -> None:
    """
    Run plotting funtion.
    
    Parameters
    ----------
    root : str
        Folder location of model root.
    name : str
        Unique name of the model instance.
    run_fn : pd.DataFrame
        Pandas DataFrame containing run results timeseries.
    unit : str
        Unit for plotting axis. Choose between 'mm' or 'm3'. 
    t_start : str
        Start time of plotting interval.
    t_end : str
        End time of plotting interval.
    reservoir_cap : float
        Model reservoir capacity.
    yearly_demand : float
        Average yearly demand.
    """
    df_run = run_fn

    # Create plot
    fig, ax1 = plt.subplots(1, figsize=(14,6))
    ax2 = ax1.twinx()
    
    # Drawing lines and filled area
    ax2.plot(df_run.index, df_run['reservoir_overflow'], linewidth=1, linestyle='-',
             label = 'Reservoir overflow', color='#00b389')
    ax2.plot(df_run.index, df_run['reservoir_stor'], linewidth=1, linestyle='-',
             label = 'Reservoir storage', color='#080c80') 
    ax1.fill_between(df_run.index, y1=df_run['deficit'], y2=0, label = 'Deficit' , color='#be1e2d', alpha=0.35, edgecolor='none')
    ax1.plot(df_run.index, df_run['demand'], linewidth=1, linestyle='-',
             label = 'Demand', color='#ff960d')
    
    # Axes labels
    ax1.set_ylabel(f'Demand and deficit [{unit}]')
    ax1.set_xlabel('Date')
    ax2.set_ylabel(f'Storage and overflow [{unit}]')
    
    # Axes limits
    ax2.set_ylim([0, df_run['reservoir_stor'].max() * 1.1])
    ax2.set_xlim([t_start, t_end])
    ax1.set_ylim([0, df_run['demand'].max() * 1.1])
    ax1.set_xlim([t_start, t_end])
    
    # Layout and grid
    ax1.spines.top.set_visible(False)
    plt.grid(visible=True, which="major", color="black", linestyle="-", alpha=0.2)
    plt.grid(visible=True, which="minor", color="black", linestyle="-", alpha=0.1)
    
    # Legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
    
    # Export
    fig.savefig(f"{root}/output/figures/{name}_run_reservoir={reservoir_cap}_yr-demand={yearly_demand}.png", dpi=300, bbox_inches='tight')
    #fig.savefig(f"{root}/output/figures/{name}_run_reservoir={reservoir_cap}_yr-demand={yearly_demand}.svg", dpi=300, bbox_inches='tight')


def plot_run_coverage(
        root,
        name,
        run_fn,  # Path to run summary output file
        unit,
        class_boundaries=[0, 20, 40, 60, 80, 100]  # Default boundaries for 5 classes
    ) -> None:
    """
    Run coverage plotting funtion.
    
    Parameters
    ----------
    root : str
        Folder location of model root.
    name : str
        Unique name of the model instance.
    run_fn : pd.DataFrame
        Pandas DataFrame containing run coverage results.
    unit : str
        Unit for plotting axis. Choose between 'mm' or 'm3'. 
    class_boundaries : list, optional
        A list defining the interval boundaries for the coverage categories. 
        This defines how coverage values are grouped into different intervals.
        Default value is [0, 20, 40, 60, 80, 100], which represents the following 
        coverage intervals: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%.
    """
    df = run_fn.set_index('reservoir_cap')
    df.index.name = None
    
    try:
        x_labels = np.round(pd.to_numeric(df.columns, errors='coerce'), 2)
    except ValueError:
        x_labels = np.round(df.columns.astype(float), 1)
    
    if len(x_labels) > 20:
        x_labels = x_labels[::len(x_labels) // 20]
    
    cmap = ['#e63946', '#ff960d', '#00b389', '#0ebbf0', '#080c80']  # Inverted color map
    cmap = mcolors.ListedColormap(cmap)
    norm = mcolors.BoundaryNorm(class_boundaries, cmap.N)
    
    fig, ax1 = plt.subplots(1, figsize=(14, 6))

    if len(df.columns) == 1:  # Single demand case
        c = ax1.plot(df.index, df.values.flatten() * 100, marker='o', linestyle='-', color='#080c80', label=f'Demand {df.columns[0]} {unit}/year')
        plt.xlabel(f'Specific reservoir capacity [{unit}/year]')
        plt.ylabel(f'Coverage of total demand by reservoir [%]')
        ax1.set_ylim([0, 100])
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05))
    elif len(df.index) == 1:  # Single reservoir case
        c = ax1.plot(df.columns.astype(float), df.values.flatten() * 100, marker='o', linestyle='-', color='#080c80', label=f'Reservoir {df.index[0]} {unit}')
        plt.xlabel(f'Specific demand [{unit}/year]')
        plt.ylabel(f'Coverage of total demand by reservoir [%]')
        ax1.set_ylim([0, 100])
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05))
    else: # Multi-demand and multi-reservoir case
        c = ax1.pcolormesh(df.columns.astype(float), df.index, df.values.astype(float) * 100, cmap=cmap, norm=norm)
        cbar = plt.colorbar(c, label='Yearly demand coverage by reservoir (%)')
        cbar.set_ticks=np.linspace(0, 100, 6)
        plt.xticks(ticks=x_labels, labels=x_labels)
        plt.xlabel(f'Specific demand [{unit}/year]')
        plt.ylabel(f'Specific reservoir capacity [{unit}]')
    
    plt.grid(visible=True, which="major", color="white", linestyle="-", alpha=0.2)
    
    fig.savefig(f"{root}/output/figures/{name}_run_coverage.png", dpi=300, bbox_inches='tight')
    #fig.savefig(f"{root}/output/figures/{name}_run_coverage.svg", dpi=300, bbox_inches='tight')


def plot_system_curve(
        root,
        name,
        system_fn,
        threshold,
        timestep,
        T_return_list = [1,2,5,10],
        validation = False
    ) -> None:
    """
    System curve plotting funtion.
    
    Parameters
    ----------
    root : str
        Folder location of model root.
    name : str
        Unique name of the model instance.
    system_fn : pd.DataFrame
        Pandas DataFrame containing minimum reservoir size for
        a set of demand values.
    threshold : int
        Threshold to be used for the Peak Over Threshold data fitting. 
    timestep : int
        Model timestep to be used for axis labels.
    T_return_list : list, optional
        A list of return time periods to be used in the Peak Over Threshold
        data fitting.
    validation : bool, optional
        Boolean to indicate whether raw model results should be plotted over fitted curves
        for the Peak Over Threshold analysis. Default is False.
    """    
    if validation:
        plot_name = "validation_"
    else:
        plot_name = ""

    df_vars = func_fitting(system_fn)
    
    # Define maximum x-range based on the maximum calculated reservoir size (x-axis bounds)
    x_max = np.max(system_fn['reservoir_cap'].max())
    x_range = np.arange(0.01,x_max,1).astype('float64')
    
    # Create plot
    fig, ax = plt.subplots(1, figsize=(8,6))
    
    # Plot system behavior curves and collect y-values
    y_values = []
    for i, col in enumerate(T_return_list):
        # Plot system behavior curves
        y_data = func_system_curve(x_range, df_vars.loc[str(col),"a"], df_vars.loc[str(col), "b"], df_vars.loc[str(col), "n"])
        plt.plot(x_range, y_data, label=f'T{col}', color=cmap[i % len(cmap)])
        y_values.extend(y_data)  # Collect all y-values
        
        if validation:
            plt.scatter(system_fn['reservoir_cap'], system_fn[str(col)], label=f'Raw data T{col}', color=cmap[i % len(cmap)], alpha=0.4, s=25, marker='x')
    
    y_max = np.max(y_values)
    plt.axis([0, x_max, 0, y_max])  # Set axis limits
    
    # Obtain colloquial timestep for x-axis
    timestep_txt = colloquial_date_text(timestep)
    
    # Axes labels
    ax.set_xlabel('Specific reservoir capacity [mm]')
    ax.set_ylabel(f'Specific water demand [mm/year]') #TODO: change y-axis to mm/timestep?
    
    # Layout and grid
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.grid(visible=True, which="major", color="black", linestyle="-", alpha=0.2)
    
    # Legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(T_return_list))
    
    # Export
    fig.savefig(f"{root}/output/figures/{name}_system_curve_{plot_name}{threshold}timesteps.png", dpi=300, bbox_inches='tight')
    #fig.savefig(f"{root}/output/figures/{name}_system_curve_{plot_name}{threshold}timesteps.svg", dpi=300, bbox_inches='tight')

    

def plot_saving_curve(
        root,
        name,
        unit,
        system_fn, # Path to saved system file
        threshold, # The maximum number of total OR consecutive days
        timestep,
        T_return_list = [1,2,5,10],
        reservoir_max = None,
        ambitions = None, # List of desired reduction lines (in %)
        **kwargs
    ) -> None:
    """
    Savings curve plotting funtion.
    
    Parameters
    ----------
    root : str
        Folder location of model root.
    name : str
        Unique name of the model instance.
    unit : str, optional
        Unit for plotting axis. Choose between 'mm' or 'm3'. Default is taken
        from model class.
    system_fn : pd.DataFrame
        Pandas DataFrame containing minimum reservoir size for
        a set of demand values.
    threshold : int
        Threshold to be used for the Peak Over Threshold data fitting. 
    timestep : int
        Model timestep to be used for axis labels.
    T_return_list : list, optional
        A list of return time periods to be used in the Peak Over Threshold
        data fitting.
    reservoir_max : int, float, optional
            Maximum reservoir size to be used as axis limit.
    ambitions : list, optional
        Vertical ambition lines to be plotted.
    """ 
    if ambitions and type(ambitions)!=list:
        raise ValueError("Provide ambitions as list of percentages")
    
    typologies_name = kwargs['typologies_name']
    typologies_demand = kwargs['typologies_demand']
    typologies_area = kwargs['typologies_area']

    df_vars = func_fitting(system_fn)
    
    cmap_list = [cmap_g1, cmap_g2, cmap_g3, cmap_g4]
    
    if reservoir_max:
        y_max = reservoir_max
    else:
        if unit == 'mm':
            y_max = 100 # Default to maximum reservoir size of 100 mm
        if unit == 'm3':
            y_max = 20 # Default to maximum reservoir size of 20 m3
    
    fig, ax = plt.subplots(1, figsize=(8,6))
    plt.axis([0, 100, 0, y_max]) #setting axis boundaries (xmin, xmax, ymin, ymax)
    
    # Create plot
    for i, typology in enumerate(typologies_name):
        cmap_i = cmap_list[i]
        df_graph = pd.DataFrame(columns = T_return_list)
        df_graph['proc'] = np.arange(0,1.001,0.001)
        df_graph['demand'] = (df_graph['proc'] * typologies_demand[i])# / (365* (86400 / timestep)) # TODO: improve Transform from yearly_demand to timestep_demand
        
        # Filling df_graph with values for tank size i.r.t. yearly demands. Calculating back to m3 using surface area if requested.
        for j, col in enumerate(T_return_list):
            df_graph[col] = func_system_curve_inv(df_graph["demand"],
                                                                  df_vars.loc[str(col), "a"],
                                                                  df_vars.loc[str(col), "b"],
                                                                  df_vars.loc[str(col), "n"])
            if unit == 'm3': # Calculate df_graph back to m3
                df_graph[col] = (df_graph[col] / 1000) * typologies_area[i]
           
        # Plotting curves
        for i, col in enumerate(T_return_list):
            plt.plot(df_graph['proc']*100, df_graph[col], label=f'{typology} T{col}', color=cmap_i[i])
        
        # Plotting ambitions
        if ambitions:
            for ambition in ambitions:
                ax.axvline(x = ambition, color = '#afafaf', linestyle='--', linewidth=1, zorder = 3)
                ax.text(ambition-1, 15, f"{ambition}% reduction", ha="center", va="bottom", color = 'grey', rotation=90)
    
    # Layout and grid
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.grid(visible=True, which="major", color="black", linestyle="-", alpha=0.2)
    plt.xticks(np.arange(0,110,10))
    
    # Axes labels
    ax.set_xlabel('Reduction in demand [%]')
    ax.set_ylabel(f'Required reservoir size [{unit}]')
    
    # Legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(typologies_name))
    
    # Export
    fig.savefig(f"{root}/output/figures/{name}_savings_curve_{threshold}timesteps.png", dpi=300, bbox_inches='tight')
    #fig.savefig(f"{root}/output/figures/{name}_savings_curve_{threshold}timesteps.svg", dpi=300, bbox_inches='tight')