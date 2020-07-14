import pandas as pd
import numpy as np

"""
Into this file, we have stored different simple useful functions.

"""

# Python module.
#
#
# Pietro Foini
#
# Year: 2020
    
def find_multiple_sets(df):
    """
    This function allows to obtain a list of all sets starting from a dataframe (or serie). These sets are obtained by splitting the 
    dataframe based on the nan values inside it.
    
    Parameters
    ----------
    df: a dataframe (or also a simple serie) containing sets of nan values whereby the dataframe can be split.
    
    Return
    ----------
    sets: a list of dataframes (or series).
    
    """
    if isinstance(df, pd.DataFrame):
        grps = df.isna().all(axis = 1).cumsum()
    elif isinstance(df, pd.Series):
        grps = df.isna().cumsum()
    # Find all the sets.
    dfs = [df.dropna() for _, df in df.groupby(grps) if not df.dropna().empty]
    return dfs
