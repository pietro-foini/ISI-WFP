from rolling_window import rolling_window
import numpy as np
import pandas as pd

# Python module.
#
#
# Pietro Foini
#
# Year: 2020

def LagsCreator(group, lags_dictionary, target, n_out, single_step = True, h = None, return_dataframe = False, 
                feature_time = False, validation = False):
    """
    This module allows to create training/validation/test lag-features for time-series forecasting. It supports several 
    configurations that you can change to get the desired output format. The starting point for using this module is to have
    a dataframe with two levels on axis 1: the level 0 corresponding to the main group and the level 1 corresponding 
    to the time-series.
    
    Parameters
    ----------
    group: a pandas dataframe with hierarchical multi-index on axis 1 where the time-series are stored. The dataframe must 
       have as index a single pandas datetime column with an appropriate frequency set.
    lags_dictionary: a python dictionary containing the lag values for each time-series.
    target: a string of the time-series name that you want to predict.
    n_out: the forecasting horizon ahead in the future.
    single_step: if set, each prediction horizon is predicted independently of the others.
    h: if single_step is set, the independent forecasting horizon to predict.
    return_dataframe: in this mode, the output are returned into pandas dataframe.
    feature_time: if you want to create also the feature time to add as feature in the input samples. This parameter can be use only if
       the single_step and return_dataframe mode are set.
    validation: if you want to create also validation points.
    
    Return
    ----------
    X: the training input samples.
    y: the training output samples.
    X_val: the validation input samples.
    y_val: the validation output samples.
    X_test: the test input sample.
    
    """
    
    # Parameters check.
    if single_step and h == None:
        raise ValueError("If 'single_step' is set, you must provide a value for the 'h' parameter.")
    if feature_time and (not single_step or not return_dataframe):
        raise ValueError("You can use the 'feature_time' only if you are working in the 'single_step' and 'return_dataframe' modes.")
    
    # Define the name of the adminstrata.
    adminstrata = group.columns.get_level_values(0).unique()
    # Remove level 0 from the dataframe.
    group = group.droplevel(level = 0, axis = 1)
    
    # Not consider features whose are not specified into lags_dictionary (the label feature must be always included).
    features_to_remove = list(set(list(group.columns)) - set(list(lags_dictionary.keys())))
    group = group.drop(columns = features_to_remove)
    
    # Define the features.
    features = group.columns
    n_features = len(features)
    # Check static features.
    static_features = [key for (key, value) in lags_dictionary.items() if value == 0]
    
    # Create training samples of shape = (n_samples, window-length, n_features).
    # Rolling a window on dataframe based on the maximum value of the lags dictionary.
    window_length = max(lags_dictionary.values())
    X = rolling_window(group.values, window_length, axes = 0).swapaxes(1, 2)
    
    # Create mask of lags based on lags into lags_dictionary.
    mask = np.full(shape = (window_length, n_features), fill_value = False)    
    for i, feature in enumerate(features):
        lags = lags_dictionary[feature]
        mask[:, i][-lags:] = True
    mask = np.tile(mask, (X.shape[0], 1, 1))

    # Define training samples with defined lags.
    X = np.ma.masked_array(X, mask = ~mask, fill_value = 0).filled(np.nan)
    
    # Create training labels.
    if single_step:
        y = rolling_window(group[target][window_length+h-1:], 1)
    else:
        y = rolling_window(group[target][window_length:], n_out)

    # Return the data into dataframe style.
    if return_dataframe:
        X = np.stack([x.flatten("F") for x in X])
        # Create columns values.
        columns = ["x(t)" if i == 1 else "x(t-%d)" % (i-1) for i in range(window_length, 0, -1)]
        # Create multi-index columns.
        iterables = [features, columns]
        columns = pd.MultiIndex.from_product(iterables, names = ["Features", "Lags"])
        # Create dataframe of training samples.    
        X = pd.DataFrame(X, columns = columns)
        X.dropna(axis = 1, how = "all", inplace = True)
        # Adjust features for static features.
        for feature in static_features:
            X[feature] = X[feature][["x(t)"]]
            # Replace names for static features.
            X.columns = pd.MultiIndex.from_tuples(map(lambda x: (feature, "x") if x == (feature, "x(t)") else x, X.columns), names = X.columns.names)            
        X.dropna(axis = 1, how = "all", inplace = True)
        
        if feature_time:
            days = rolling_window(group[target][window_length:].index.day, n_out)[:, h - 1]
            months = rolling_window(group[target][window_length:].index.month, n_out)[:, h - 1]
            years = rolling_window(group[target][window_length:].index.year, n_out)[:, h - 1]
            # Create feature time.
            dates = np.stack([days, months, years], axis = 1)
            columns = pd.MultiIndex.from_tuples([("Day", "x"), ("Month", "x"), ("Year", "x")], names = ["Features", "Lags"])
            X_dates = pd.DataFrame(dates, columns = columns)
            # Add to the dataframe.
            X = pd.concat([X, X_dates], axis = 1)

        # Create dataframe of labels.
        # Create columns values.
        if single_step:
            columns = ["x(t+%d)" % h]
        else:
            columns = ["x(t+%d)" % (i+1) for i in range(n_out)]
        # Create multi-index columns.
        iterables = [[target], columns]
        columns = pd.MultiIndex.from_product(iterables, names = ["Features", "Prediction horizon"])  
        y = pd.DataFrame(y, columns = columns)

    # Define test sample input.
    X_test = X[-1:]

    # Adjust data if validation mode is set.
    if validation:
        if single_step:
            y_val = y[-n_out:]
            X_val = X[-(n_out+h):][:n_out]
            y = y[:-n_out]
            X = X[:-(n_out+h)]
        else:
            y_val = y[-1:]
            X_val = X[:y.shape[0]][-1:]
            y = y[:-n_out]
            X = X[:-2*n_out]
    else:
        X = X[:y.shape[0]]
        # Define empty validation data.
        X_val = None
        y_val = None

    return X, y, X_val, y_val, X_test