import numpy as np
import pandas as pd


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