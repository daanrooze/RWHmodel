import os
import pandas as pd
from io import StringIO
import numpy as np
from datetime import datetime, timedelta

from RWHmodel.plot import (
    plot_meteo,
    plot_run,
    plot_run_coverage,
    plot_system_curve,
    plot_saving_curve
)

# ---------------------------
# Create dummy output folders
# ---------------------------
root = "./tests"
os.makedirs(f"{root}/output/figures", exist_ok=True)
name = "test_model"

# ---------------------------
# Read forcing and demand time series from CSV files
# ---------------------------
forcing_fp = "./tests/input/forcing_test.csv"
forcing_df = pd.read_csv(forcing_fp, index_col=0)
forcing_df.index = pd.to_datetime(forcing_df.index, format = "%d-%m-%Y %H:%M")

demand_fp = "./tests/input/demand_test.csv"
demand_df = pd.read_csv(demand_fp, index_col=0)
demand_df.index = pd.to_datetime(demand_df.index, format = "%d-%m-%Y %H:%M")

# ---------------------------
# Create run_df DataFrame based on demand_df and some random values (or zeros)
# ---------------------------
run_df = pd.DataFrame({
    "reservoir_overflow": np.random.uniform(0, 50, size=len(demand_df.index)),
    "reservoir_stor": np.random.uniform(0, 100, size=len(demand_df.index)),
    "deficit": np.random.uniform(0, 5, size=len(demand_df.index)),
    "demand": np.random.uniform(0, 5, size=len(demand_df.index))
}, index=demand_df.index)

# For coverage: rows = reservoir_cap, cols = demands
reservoir_caps = [10, 20, 30, 40, 50]
demands = ["50", "100", "150"]
coverage_values = np.random.uniform(0, 1, (len(reservoir_caps), len(demands)))
coverage_df = pd.DataFrame(coverage_values, index=reservoir_caps, columns=demands)
coverage_df.reset_index(inplace=True)
coverage_df.rename(columns={"index": "reservoir_cap"}, inplace=True)

# For system curve & saving curve: synthetic data
reservoir_cap = [
    10, 20.52631579, 31.05263158, 41.57894737, 52.10526316, 62.63157895, 73.15789474,
    83.68421053, 94.21052632, 104.7368421, 115.2631579, 125.7894737, 136.3157895,
    146.8421053, 157.3684211, 167.8947368, 178.4210526, 188.9473684, 199.4736842, 210
]

col_1 = [
    209.6, 324.5, 362.8, 401.1, 401.1, 439.4, 439.4,
    439.4, 439.4, 439.4, 439.4, 439.4, 439.4,
    439.4, 439.4, 439.4, 439.4, 439.4, 439.4, 439.4
]

col_2 = [
    171.4, 247.9, 324.5, 324.5, 362.8, 362.8, 362.8,
    362.8, 362.8, 362.8, 362.8, 362.8, 362.8,
    401.1, 401.1, 401.1, 401.1, 401.1, 401.1, 401.1
]

col_5 = [
    133.1, 171.4, 209.6, 247.9, 247.9, 286.2, 286.2,
    286.2, 286.2, 286.2, 286.2, 286.2, 286.2,
    324.5, 324.5, 324.5, 324.5, 324.5, 324.5, 324.5
]

col_10 = [
    94.8, 133.1, 171.4, 209.6, 209.6, 209.6, 247.9,
    247.9, 247.9, 247.9, 247.9, 247.9, 286.2,
    286.2, 286.2, 286.2, 286.2, 286.2, 286.2, 286.2
]

# Build the DataFrame
system_df = pd.DataFrame({
    "reservoir_cap": reservoir_cap,
    "1": col_1,
    "2": col_2,
    "5": col_5,
    "10": col_10
})

# ---------------------------
# Test each plotting function
# ---------------------------
plot_meteo(
    root=root,
    name=name,
    forcing_fn=forcing_df,
    t_start=forcing_df.index.min(),
    t_end=forcing_df.index.max(),
    aggregate=False
)

plot_run(
    root=root,
    name=name,
    run_fn=run_df,
    unit="mm",
    t_start=demand_df.index.min(),
    t_end=demand_df.index.max(),
    reservoir_cap=100,
    yearly_demand=500
)

plot_run_coverage(
    root=root,
    name=name,
    run_fn=coverage_df,
    unit="mm"
)

plot_system_curve(
    root=root,
    name=name,
    system_fn=system_df,
    threshold=5,
    timestep=86400,
    T_return_list=[1, 2, 5, 10],
    validation=True
)

plot_saving_curve(
    root=root,
    name=name,
    unit="mm",
    system_fn=system_df,
    threshold=5,
    timestep=86400,
    T_return_list=[1, 2, 5, 10],
    typologies_name=["Typology A", "Typology B"],
    typologies_demand=[100, 200],
    typologies_area=[50, 100],
    ambitions=[20, 50, 80]
)