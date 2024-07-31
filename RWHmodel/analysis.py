import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def func_log(a, b, x):
    return a * np.log(x) + b


def return_period(df): #TODO: variabelen namen nalopen. Deze komen nog uit UWBM
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
    req_storage["Treturn"] = [1, 2, 5, 10, 20, 50, 100]
    for i, key in enumerate(df_vars["q"]):
        req_storage[key] = func_log(df_vars["a"][i], df_vars["b"][i], req_storage["Treturn"])
    req_storage = req_storage.set_index("Treturn")
    req_storage.index.name = None
    req_storage = req_storage.T
    return req_storage


def func_fitting(
        system_fn, # Path to saved system file
        T_return_list = [1,2,5,10,20,50,100]
    ):
    df_vars = pd.DataFrame(columns=["a", "b", "n"])
    df_vars["Treturn"] = T_return_list
    df_vars = df_vars.set_index("Treturn")
    
    # Loop over different return periods to obtain variables in asymptotic function
    for i, col in enumerate(T_return_list):
        # Determine initial conditions
        # Check https://stackoverflow.com/questions/45554107/asymptotic-regression-in-python for assumptions
        a0 = system_fn[str(col)].max()
        b0 = float(system_fn.iloc[(system_fn[str(col)]-(a0 / 2)).abs().argsort()[:1]].tank_size)
        n0 = 1.
        p0 = [a0, b0, n0]
        
        # Curve fit using Scipy
        popt, pcov = curve_fit(func_system_curve, system_fn['tank_size'], system_fn[str(col)], p0=p0)
        
        # Store [a, b, n] variables in DataFrame
        df_vars.loc[col] = popt
    return df_vars


def func_system_curve(x, a, b, n):
    y = a * x ** n  / (x ** n + b)
    return y

def func_system_curve_inv(y, a, b, n):
    x = ((a - y)  / (b * y))**(-1/n)
    return x

