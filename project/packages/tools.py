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
    events = np.split(df, np.where(np.isnan(df.values))[0])
    # Removing NaN entries.
    events = [ev[~np.isnan(ev.values)] for ev in events if not isinstance(ev, np.ndarray)]
    # Removing empty DataFrames.
    sets = [ev for ev in events if not ev.empty]
    return sets
