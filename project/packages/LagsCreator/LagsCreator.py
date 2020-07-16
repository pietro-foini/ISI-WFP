from rolling_window import rolling_window
import numpy as np
import pandas as pd

# Python module.
#
#
# Pietro Foini
#
# Year: 2020

class LagsCreator:
    """LagsCreator
    
    This module allows to create training/validation/test lag-features for time-series forecasting. It supports several 
    configurations to get the output into several format. The starting point for using this module is to have
    a dataframe with two levels on axis 1: the level 0 corresponding to the main group and the level 1 corresponding 
    to the time-series. An advantage of this module is that it is possible to visualize the samples created during the precess through 
    an highlighting of the cells of the dataframe.
    
    """
    def __init__(self, group, lags_dictionary, target):
        """
        ***Initialization function***
 
        Initialization of the LagsCreator class.
        
        Parameters
        ----------
        group: a pandas dataframe with hierarchical multi-index on axis 1 (more precisely two levels) where the time-series are stored. 
           The dataframe must have as index a single pandas datetime column with an appropriate frequency set. 
        lags_dictionary: a python dictionary containing the lag values corresponding to each time-series (the names of the time-series 
           will be the keys of the dictionary).
        target: a python string containing the name of the time-series that you want to predict. The target variable must always present
           also into the 'lags_dictionary' parameter.
           
        """
        # Define the name of the group.
        group_name = group.columns.get_level_values(0).unique()
        # Remove level 0 from the dataframe.
        group = group.droplevel(level = 0, axis = 1)
        
        # Check of the 'lags_dictionary' parameter.
        if target not in lags_dictionary.keys():
            raise ValueError("The target feature must be always included in the 'lags_dictionary' parameter.")
        # The features whose are not specified into 'lags_dictionary' are removed.
        features_to_remove = list(set(list(group.columns)) - set(list(lags_dictionary.keys())))
        group = group.drop(columns = features_to_remove)
        # Define the features (the names of the time-series).
        features = group.columns
        # Define static features among the features (features, i.e. time-series, with lag value set to 0).
        static_features = [key for (key, value) in lags_dictionary.items() if value == 0]
        
        # Define the boolean mask for the creation of feature-lags for each time-series.
        # Define the reference size of the window (timestep dimension).
        window_size = max(lags_dictionary.values())
        # Create mask based on lags into 'lags_dictionary' to pass over the input samples.
        mask = np.full(shape = (window_size, len(features)), fill_value = False)    
        for i, feature in enumerate(features):
            lags = lags_dictionary[feature]
            mask[:, i][-lags:] = True

        # Define some attributes of the class.
        self.group_name = group_name
        self.group = group
        self.features = features
        self.static_features = static_features
        self.target = target
        self.window_size = window_size
        self.mask = mask
    
    def to_dataframe(self, X, y):
        """
        ***Sub-function***
 
        This function allows to convert the outputs (input and output samples) into dataframes format.
        
        Parameters
        ----------
        X: the input samples with the column of temporal information of shape (n_samples, timesteps, n_features).
        y: the output samples with the row of temporal information of shape (n_samples, 2, n_out]).

        """
        # Create dataframe of output samples.
        if y is not None:
            # Consider the temporal information.
            dates = y[:, 0, :].flatten()
            # Not consider temporal information.
            y = y[:, 1, :]
            # Create columns values.
            if self.single_step:
                columns = ["x(t+%d)" % self.h]
            else:
                columns = ["x(t+%d)" % (i+1) for i in range(self.n_out)]
            # Create multi-index columns.
            iterables = [[self.target], columns]
            columns = pd.MultiIndex.from_product(iterables, names = ["Features", "Prediction horizon"])  
            y = pd.DataFrame(y, columns = columns)
        else:
            y = None
            
        # Create dataframe of input samples.
        if X is not None:
            # Not consider temporal information for the input samples.
            X = X[:, :, 1:]
            # Flatten the lags of each sample over the rows.
            X = np.stack([x.flatten("F") for x in X])
            # Create columns values.
            columns = ["x(t)" if i == 1 else "x(t-%d)" % (i-1) for i in range(self.window_size, 0, -1)]
            # Create multi-index columns.
            iterables = [self.features, columns]
            columns = pd.MultiIndex.from_product(iterables, names = ["Features", "Lags"])
            # Create dataframe of input samples.    
            X = pd.DataFrame(X, columns = columns)
            X.dropna(axis = 1, how = "all", inplace = True)
            # Adjust features for static features.
            for feature in self.static_features:
                X[feature] = X[feature][["x(t)"]]
                # Replace names for static features.
                X.columns = pd.MultiIndex.from_tuples(map(lambda x: (feature, "x") if x == (feature, "x(t)") else x, X.columns), names = X.columns.names) 
                # Change the variable X regarding the lags to show for the static features.
                self.mask[:, self.features.get_loc(feature)][:-1] = False
                mask = np.tile(self.mask, (self.X.shape[0], 1, 1))
                self.X[:, :, 1:] = np.ma.masked_array(self.X[:, :, 1:], mask = ~mask, fill_value = 0).filled(np.nan)
            X.dropna(axis = 1, how = "all", inplace = True)

            # Add the temporal information to the input samples.
            if self.feature_time:
                if y is not None:
                    days = [date.day for date in dates]
                    months = [date.month for date in dates]
                    years = [date.year for date in dates]
                    # Create feature time.
                    dates = np.stack([days, months, years], axis = 1)
                    columns = pd.MultiIndex.from_tuples([("Day", "x"), ("Month", "x"), ("Year", "x")], names = ["Features", "Lags"])
                    dates = pd.DataFrame(dates, columns = columns)
                    # Add to the dataframe.
                    X = pd.concat([X, dates], axis = 1)
                else:
                    # For the test samples.
                    X.loc[0, ("Day", "x")] = (self.group.index[-1] + (self.h-1)*self.group.index.freq).day
                    X.loc[0, ("Month", "x")] = (self.group.index[-1] + (self.h-1)*self.group.index.freq).month
                    X.loc[0, ("Year", "x")] = (self.group.index[-1] + (self.h-1)*self.group.index.freq).year
        else:
            X = None

        return X, y
        
    def to_supervised(self, n_out, single_step = False, h = None, return_dataframe = False, validation = False, 
                      feature_time = False, return_single_level = False, dtype = object):
        """
        ***Main function***
 
        This function allows to create training/validation/test samples to use for time-series forecasting purposes.
        The main output format difference is determined by the 'return_dataframe' parameter. If set, the outputs are 
        rearranged into pandas dataframes otherwise the output are returned as numpy arrays.
        
        Parameters
        ----------
        n_out: the maximum forecasting horizon ahead in the future. If 'single_step' is set, the parameter 'n_out' indicates 
           the size of the validation set.
        single_step: if set, each prediction horizon is predicted independently from the others.
        h: the independent forecasting horizon to predict for the 'single_step' mode. If 'single_step = False', the 'h' parameter
           is not taken into account.
        return_dataframe: the modality to set in order to have the outputs returned as pandas dataframes.
        validation: if you want to create validation points.
        feature_time: if you want to create a feature time to add as feature in the input samples. This parameter can be use 
           only if the 'single_step' and 'return_dataframe' modes are set.
        return_single_level: if 'return_dataframe' is set, this parameter allows to have as output dataframes with a single level on
           axis 1 merging column names levels. If 'return_dataframe = False', the 'return_single_level' parameter is not taken 
           into account.
           
        Return
        ----------
        X_train: the training input samples.
        y_train: the training output samples.
        X_val: the validation input samples.
        y_val: the validation output samples.
        X_test: the test input sample.
    
        """
        # Check parameters.
        if single_step and h is None:
            raise ValueError("If 'single_step' is set, you must provide a value for the 'h' parameter.")
        if not single_step and h is not None:
            h = None
        if h is not None and h > n_out:
            raise ValueError("The 'h' parameter must be not greater than 'n_out' parameter.")      
        if feature_time and (not single_step or not return_dataframe):
            raise ValueError("You can use the 'feature_time' only if you are working in the 'single_step' and 'return_dataframe' modes.")
    
        # Define attributes of the class.
        self.n_out = n_out
        self.single_step = single_step
        self.h = h
        self.return_dataframe = return_dataframe
        self.validation = validation
        self.feature_time = feature_time

        # Rolling a no masked window over the dataframe based on the maximum value of the 'lags_dictionary'.
        # Create input samples.
        X = rolling_window(self.group.reset_index().values, self.window_size, axes = 0).swapaxes(1, 2)
        # Add the mask to the input samples based on lags.
        # Add the temporal information to the mask in order to always mantain the temporal information.
        temporal_mask = np.expand_dims(np.array([True]*self.window_size), 1)
        mask = np.concatenate([temporal_mask, self.mask], axis = 1)
        # Expand the mask to all the samples.
        mask = np.tile(mask, (X.shape[0], 1, 1))
        # Define input samples with defined lags (with also temporal information).
        X = np.ma.masked_array(X, mask = ~mask, fill_value = 0).filled(np.nan)
        
        # Create output samples.
        if single_step:
            y = rolling_window(self.group[self.target].reset_index().values[self.window_size + h-1:], 1, axes = 0)
        else:
            y = rolling_window(self.group[self.target].reset_index().values[self.window_size:], n_out, axes = 0)
        
        # Splitting of the input X samples and the output y samples into training/validation/test.
        # Define the test sample input.
        X_test = X[-1:]
        # Define the training and validation samples input and outputs.
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
            X_val, y_val = None, None
            
        # Samples arrays created until here with also temporal information: X, y, X_val, y_val, X_test.    
        self.X = X
        self.y = y
        self.X_val = X_val # It could be None if 'validation = False'.
        self.y_val = y_val # It could be None if 'validation = False'.
        self.X_test = X_test
        
        # In this last phase, the output format is changed if desired: array or dataframe outputs.
        if return_dataframe:
            # Define input and output samples training dataframes.
            X_train, y_train = self.to_dataframe(X, y)
            # Change the type of the dataframe.
            X_train = X_train.astype(dtype)
            y_train = y_train.astype(dtype)
            # Define input and output samples validation dataframes.
            X_val, y_val = self.to_dataframe(X_val, y_val)
            # Change the type of the dataframe.
            if validation:
                X_val = X_val.astype(dtype)
                y_val = y_val.astype(dtype)
            # Define input samples test dataframes.
            X_test, _ = self.to_dataframe(X_test, None)
            # Change the type of the dataframe.
            X_test = X_test.astype(dtype)
            # Save the name of the adminstrata as index for the test input sample.
            X_test.index = [self.group_name]
            # Flatten of the hierarchical multi-index on axis 1.
            if return_single_level:
                X_train.columns = X_train.columns.map(lambda x: " | ".join([str(i) for i in x]))
                y_train.columns = y_train.columns.map(lambda x: " | ".join([str(i) for i in x]))
                if validation:
                    X_val.columns = X_val.columns.map(lambda x: " | ".join([str(i) for i in x]))
                    y_val.columns = y_val.columns.map(lambda x: " | ".join([str(i) for i in x]))
                X_test.columns = X_test.columns.map(lambda x: " | ".join([str(i) for i in x]))
        else:
            # Define input samples training arrays removing the temporal information.
            X_train = X[:, :, 1:].astype(dtype)
            # Define output samples training arrays removing the temporal information.
            y_train = y[:, 1, :].astype(dtype)
            if validation:
                # Define input samples validation arrays removing the temporal information.
                X_val = X_val[:, :, 1:].astype(dtype)
                # Define output samples validation arrays removing the temporal information.
                y_val = y_val[:, 1, :].astype(dtype) 
            else:
                X_val, y_val = None, None
            # Define input samples test arrays removing the temporal information.
            X_test = X_test[:, :, 1:].astype(dtype)
            
        return X_train, y_train, X_val, y_val, X_test
    
    def highlight_cells(self, x, y):
        """
        ***Sub-function***
 
        This function draws the cells of the dataframe that belongs to the lag features for the current input sample x and
        output sample y.

        """
        # Define a group for style where the values inside the dataframe are converted to strings.
        group_style = self.group.round(4).astype(str)

        # Pandas mask.
        m = pd.DataFrame(self.mask, index = x[:, 0], columns = self.features)
        # Getting (index, column) pairs for True elements of the boolean DataFrame.
        cells_to_color_input = m[m == True].stack().index.tolist()
        if not self.return_dataframe:
            cells_to_color_input_extra = m[m == False].stack().index.tolist()
        if y is not None:
            cells_to_color_output = y[0]

        def draw(x):
            df_styler  = group_style.copy()
            if self.validation:
                df_styler.loc[-self.n_out:, self.target] = "color: red"
            # Set particular cell colors for the input.
            for location in cells_to_color_input:
                df_styler.loc[location[0], location[1]] = "background-color: RGB(0,131,255)"
            # If the return dataframe is not set, the extra cells of the input filled with nan values are opaque colored.
            if not self.return_dataframe:
                for location in cells_to_color_input_extra:
                    df_styler.loc[location[0], location[1]] = "background-color: RGBA(0,131,255,0.44)"
            # Set particular cell colors for the output.
            if y is not None:
                for location in cells_to_color_output:
                    df_styler.loc[location, self.target] = "background-color: RGB(255,154,0)"
            return df_styler 
        
        # Highlight the following sample into the dataframe.
        sample = group_style.style.apply(lambda x: draw(x), axis = None)
        return sample
        
    def visualization(self, boundaries = True):
        """
        ***Sub-function***
 
        This function allows to visualize the training/validation/test input and output samples created by the process.
        
        Parameters
        ----------
        boundaries: if you want to visualize only the first two an the last two sample points created.
        
        Return
        ----------
        train_dataframes: a list of dataframes that underline each training sample created.
        validation_dataframes: a list of dataframes that underline each validation sample created.
        test_dataframes: a list of dataframes that underline each test sample created.

        """
        # Create dataframes for visualization.
        if boundaries:
            # Keep only boundaries samples.
            self.X = np.concatenate([self.X[:2], self.X[-2:]])
            self.y = np.concatenate([self.y[:2], self.y[-2:]])
            train_dataframes = [self.highlight_cells(x, y) for x, y in zip(self.X, self.y)]
            if self.X_val is not None:
                # Keep only boundaries samples.
                self.X_val = np.concatenate([self.X_val[:2], self.X_val[-2:]])
                self.y_val = np.concatenate([self.y_val[:2], self.y_val[-2:]])
                validation_dataframes = [self.highlight_cells(x, y) for x, y in zip(self.X_val, self.y_val)]
            else:
                validation_dataframes = None
            test_dataframes = [self.highlight_cells(x, None) for x in self.X_test]
        else:
            train_dataframes = [self.highlight_cells(x, y) for x, y in zip(self.X, self.y)]
            if self.X_val is not None:
                validation_dataframes = [self.highlight_cells(x, y) for x, y in zip(self.X_val, self.y_val)]
            else:
                validation_dataframes = None
            test_dataframes = [self.highlight_cells(x, None) for x in self.X_test]
            
        return train_dataframes, validation_dataframes, test_dataframes
    