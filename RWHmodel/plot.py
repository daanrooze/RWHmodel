from RWHmodel.analysis import func_system_curve
from RWHmodel.analysis import func_system_curve_inv
from RWHmodel.analysis import func_fitting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


### COLOR MAPS
# Default color map
cmap = ['#080c80', '#0ebbf0', '#00b389', '#ff960d']
# Gradient color maps
cmap_g1 = ['#080c80', '#5f6199', '#9395b9', '#c9cadc']
cmap_g2 = ['#00b389', '#5bc1a4', '#90d1c0', '#c6e7de']
cmap_g3 = ['#ff960d', '#f8b05b', '#fbcb8f', '#fde4c6']
cmap_g4 = ['#0ebbf0', '#62ccf1', '#96d9f2', '#c9ebf7']


    


def plot_meteo(
        root,
        name,
        forcing_fn,
        t_start,
        t_end,
        aggregate = False
    ):
    df = forcing_fn
    # Clip DataFarme based on t_start and t_end
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
    ax2.plot(df.index, df['precip'], linewidth=2, linestyle='-',
             label = 'Precipitation', color='#080c80') 
    ax1.fill_between(df.index, y1=df['pet'], y2=0, label = 'Potential Evapotranspiration' , color='#c6c6c6')
    
    # Axes labels
    ax1.set_ylabel('Potential Evaporation [mm]')
    ax1.set_xlabel('Date')
    ax2.set_ylabel('Precipitation [mm]')
    
    # Axes limits
    pet_max = np.round(df['pet'].max(), 0) + 1
    precip_max = np.round(df['precip'].max(), -1) + 10
    ax1.set_ylim([0, pet_max])
    ax2.set_ylim([0, precip_max])
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
    legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05)) #TODO: maybe different layout for legend?
    
    # Export
    fig.savefig(f"{root}/output/figures/{name}_forcing_{plot_name}_{t_start.year}_{t_start.month}-{t_end.year}_{t_end.month}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{root}/output/figures/{name}_forcing_{plot_name}_{t_start.year}_{t_start.month}-{t_end.year}_{t_end.month}.svg", dpi=300, bbox_inches='tight')

def plot_run(
        root,
        name,
        run_fn, # Path to run output file
        t_start,
        t_end,
        reservoir_cap,
        yearly_demand
    ):
    df = run_fn
    # Create plot
    fig, ax1 = plt.subplots(1, figsize=(14,6))
    
    # Drawing lines and filled area
    ax1.plot(df.index, df['reservoir_stor'], linewidth=2, linestyle='-',
             label = 'Reservoir storage', color='#080c80') 
    ax1.plot(df.index, df['reservoir_overflow'], linewidth=2, linestyle='-',
             label = 'Reservoir overflow', color='#00b389')
    ax1.plot(df.index, df['deficit'], linewidth=2, linestyle='-',
             label = 'Deficit', color='#ff960d')
    
    # Axes labels
    ax1.set_ylabel('Storage/overflow/deficit [mm]')
    ax1.set_xlabel('Date')
    
    # Axes limits
    y_max = np.round(df['reservoir_stor'].max(), -1) + 10
    ax1.set_ylim([0, y_max])
    ax1.set_xlim([t_start, t_end])
    
    # Layout and grid
    ax1.spines.top.set_visible(False)
    plt.grid(visible=True, which="major", color="black", linestyle="-", alpha=0.2)
    plt.grid(visible=True, which="minor", color="black", linestyle="-", alpha=0.1)
    
    # Legend
    legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05)) #TODO: maybe different layout for legend?
    
    # Export
    fig.savefig(f"{root}/output/figures/{name}_run_reservoir={reservoir_cap}_yr_demand={yearly_demand}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{root}/output/figures/{name}_run_reservoir={reservoir_cap}_yr_demand={yearly_demand}.svg", dpi=300, bbox_inches='tight')
    

def plot_system_curve(
        root,
        name,
        system_fn,
        max_num_days, # The maximum number of total OR consecutive days
        demand_range,
        T_return_list = [1,2,5,10,20,50,100],
        validation = False
    ):    
    if validation:
        plot_name = "validation_"
    else:
        plot_name = None

    df_vars = func_fitting(system_fn, T_return_list)
    
    # Define maximum specific water demand (x-axis bounds)
    x_max = max(demand_range)
    x_range = np.arange(0.01,x_max,1).astype('float64')
    
    # Create plot
    fig, ax = plt.subplots(1, figsize=(8,6))
    plt.axis([0, 500, 0, 1250]) #TODO: make variable of axis boundaries?
    
    for i, col in enumerate(T_return_list):
        # Plot system behavior curves
        plt.plot(x_range, func_system_curve(x_range, df_vars["a"][float(col)], df_vars["b"][float(col)], df_vars["n"][float(col)]),
                 label=f'T{col}', color=cmap[i])
        if validation:
            plt.scatter(system_fn['tank_size'], system_fn[str(col)], label=f'Raw data T{col}', color=cmap[i], alpha=0.4, s=75, marker='x')
    
    # Axes labels
    ax.set_xlabel('Specific reservoir capacity [mm]')
    ax.set_ylabel('Specific water demand [mm/year]')
    
    # Layout and grid
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.grid(visible=True, which="major", color="black", linestyle="-", alpha=0.2)
    
    # Legend
    legend = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
    
    # Export
    fig.savefig(f"{root}/output/figures/{name}_system_curve_{plot_name}{max_num_days}dd.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{root}/output/figures/{name}_system_curve_{plot_name}{max_num_days}dd.svg", dpi=300, bbox_inches='tight')

    

def plot_saving_curve(
        root,
        name,
        system_fn, # Path to saved system file
        max_num_days, # The maximum number of total OR consecutive days
        typologies_demand, # List of typologies and yearly demand, from setup_batch.toml file.
        typologies_area, # List of typologies and surface area, from setup_batch.toml file.
        ambitions = None, # List of desired reduction lines (in %)
        T_return_list = [1,2,5,10,20,50,100]
    ):
    if ambitions and type(ambitions)!=list:
        raise ValueError("Provide ambitions as list of percentages")
    
    df_vars = func_fitting(system_fn, T_return_list)
    
    cmap_list = [cmap_g1, cmap_g2, cmap_g3, cmap_g4]
    
    fig, ax = plt.subplots(1, figsize=(8,6))
    plt.axis([0, 100, 0, 20]) #setting axis boundaries (xmin, xmax, ymin, ymax)
    
    typologies = list(typologies_demand)
    # Create plot
    for i, typology in enumerate(typologies):
        cmap_i = cmap_list[i]
        df_graph = pd.DataFrame(columns = system_fn.columns[1:])
        df_graph['proc'] = np.arange(0,1.001,0.001)
        df_graph['demand'] = df_graph['proc'] * typologies_demand[typology]
        
        # Filling df_graph with values for tank size i.r.t. yearly demands. Calculating back to m3 using surface area.
        for i, col in enumerate(T_return_list):
            df_graph[str(col)] = (func_system_curve_inv(df_graph["demand"],
                                                                  df_vars["a"][col],
                                                                  df_vars["b"][col],
                                                                  df_vars["n"][col]) / 1000) * typologies_area[typology]
           
        # Plotting curves
        for i, col in enumerate(T_return_list):
            plt.plot(df_graph['proc']*100, df_graph[str(col)], label=f'{typology} T{col}', color=cmap_i[i])
        
        # Plotting ambitions
        if ambitions:
            for ambition in ambitions:
                ax.axvline(x = ambition, color = '#afafaf', linestyle='--', linewidth=1, zorder = 3)
                ax.text(ambition-1, 15, f"{ambition}% reductie", ha="center", va="bottom", color = 'grey', rotation=90)
    
    # Layout and grid
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.grid(visible=True, which="major", color="black", linestyle="-", alpha=0.2)
    plt.xticks(np.arange(0,110,10))
    
    # Axes labels
    ax.set_xlabel('Reduction in demand [%]')
    ax.set_ylabel('Required reservoir size [m3]')
    
    # Legend
    legend = fig.legend(loc='right')
    
    # Export
    fig.savefig(f"{root}/output/figures/{name}_savings_curve_{max_num_days}dd.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{root}/output/figures/{name}_savings_curve_{max_num_days}dd.svg", dpi=300, bbox_inches='tight')