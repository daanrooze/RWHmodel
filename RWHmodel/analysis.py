import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def func_log(a, b, x):
    return a * np.log(x) + b


def return_period(
        df,
        T_return_list = [1,2,5,10,20,50,100]
    ): #TODO: variabelen namen nalopen. Deze komen nog uit UWBM
    colnames = df.columns[:-1]
    df_vars = pd.DataFrame(columns=["q", "a", "b"], dtype='float64')
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
 
        df_vars.loc[i] = [float(i), float(a), float(b)]
 
    # Calculate required storage capacity for a set of return periods
    req_storage = pd.DataFrame()
    req_storage["Treturn"] = T_return_list
    for i, key in enumerate(df_vars["q"]):
        req_storage[key] = func_log(df_vars["a"][i], df_vars["b"][i], req_storage["Treturn"])
    req_storage = req_storage.set_index("Treturn")
    req_storage.index.name = None
    req_storage = req_storage.T
    return req_storage


def return_period(
        df,
        T_return_list = [1,2,5,10,20,50,100]
    ):
    colnames = df.columns[:-1]
    df_vars = pd.DataFrame(columns=["q", "a", "b"], dtype='float64')
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
 
        df_vars.loc[i] = [float(i), float(a), float(b)]
 
    # Calculate required storage capacity for a set of return periods
    req_storage = pd.DataFrame()
    req_storage["Treturn"] = T_return_list
    new_columns = {}  # CHANGE
    for i, key in enumerate(df_vars["q"]):  # CHANGE
        new_columns[key] = func_log(df_vars["a"][i], df_vars["b"][i], req_storage["Treturn"])  # CHANGE
    new_columns_df = pd.DataFrame(new_columns)  # CHANGE
    req_storage = pd.concat([req_storage, new_columns_df], axis=1)  # CHANGE
    req_storage = req_storage.set_index("Treturn")
    req_storage.index.name = None
    req_storage = req_storage.T
    return req_storage


def func_fitting(
        system_fn, # Path to saved system file
        T_return_list = [1,2,5,10,20,50,100]
    ):
    # Reset index for df_system
    df_system = system_fn.reset_index(drop=True)
    
    df_vars = pd.DataFrame(columns=["a", "b", "n"])
    #df_vars["Treturn"] = T_return_list
    df_vars["Treturn"] = df_system.columns[1:]
    df_vars = df_vars.set_index("Treturn")
    
    # Loop over different return periods to obtain variables in asymptotic function
    #for i, col in enumerate(T_return_list):
    for col in df_system.columns[1:]:
        # Determine initial conditions
        # Check https://stackoverflow.com/questions/45554107/asymptotic-regression-in-python for assumptions
        a0 = df_system[col].max()
        b0 = df_system.iloc[(df_system[col]-(a0 / 2)).abs().argsort()[:1]].reservoir_size
        n0 = 1.
        p0 = [a0, float(b0.iloc[0]), n0]
        #p0 = [float(a0), float(b0), float(n0)]
        
        # Curve fit using Scipy
        popt, pcov = curve_fit(func_system_curve, df_system['reservoir_size'], df_system[col], p0=p0)
        
        # Store [a, b, n] variables in DataFrame
        df_vars.loc[col] = popt
    return df_vars


def func_system_curve(x, a, b, n):
    y = a * x ** n  / (x ** n + b)
    return y

def func_system_curve_inv(y, a, b, n):
    x = ((a - y)  / (b * y))**(-1/n)
    return x

