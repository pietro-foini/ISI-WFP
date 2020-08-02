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
    
    This module allows to create training/validation/test lag-features samples for time-series forecasting purposes. 
    The module supports different configurations to get the outputs into several formats. An advantage of this module is 
    to visualize the lag-features samples created through an highlighting of the cells of the dataframe.
    
    """
    def __init__(self, group, lags_dictionary, target, n_out, return_dataframe = False, row_output = False, 
                 feature_time = False, single_step = False):
        """
        ***Initialization function***
 
        Initialization of the LagsCreator class.
        
        Parameters
        ----------
        group: a pandas dataframe with two hierarchical multi-index on axis 1: the level 0 corresponding to a single main group and 
           the level 1 corresponding to single/multiple time-series. The dataframe must have as index a pandas datetime column 
           with an appropriate frequency set. 
        lags_dictionary: a dictionary containing the lag values corresponding to each time-series (the names of the time-series 
           will be the keys of the dictionary). If you dont't want to use a time-series, the corresponding value in the 
           dictionary must be set to 'None'.
        target: a string containing the name of the time-series that you want to predict. The target variable must be present
           into the 'lags_dictionary'.
        n_out: the maximum forecasting horizon ahead in the future.
        return_dataframe: the modality to set in order to have the outputs returned as pandas dataframes.
        row_output: the modality to set in order to arrange the lag-features of each sample over an unique row. 
        feature_time: if you want to create a feature time to add as feature in the input samples. This parameter can be use 
           only if the 'single_step' and 'row_output' modes are set.
        single_step: if set, each prediction horizon is predicted independently from the others.
           
        """        
        # Define the name of the group (name of the column on axis 1 and level 0).
        group_name = group.columns.get_level_values(0).unique()
        # Remove level 0 of the dataframe on axis 1.
        group = group.droplevel(level = 0, axis = 1)
        
        # Check parameters.
        if feature_time and (not single_step or not row_output):
            raise ValueError("You can use the 'feature_time' only if you are working in the 'single_step' and 'row_output' modes.")
        if return_dataframe and not row_output:
            raise ValueError("If 'return_dataframe' is set, you must work using the 'row_output' mode.")
        if target not in lags_dictionary.keys():
            raise ValueError("The target feature must be always included in the 'lags_dictionary' parameter in order to be a predictor.")
        if set(lags_dictionary.keys()) != set(group.columns):
            raise ValueError("You have to provide a lag value for each time-series stored in the input dataframe. Check the 'lags_dictionary' parameter.")
            
        # The features whose are specified into 'lags_dictionary' with None values are removed (not considered as predictors).        
        features_to_remove = [k for k,v in lags_dictionary.items() if v is None]
        # Define the features that are kept.
        lags_dictionary = {k: v for k,v in lags_dictionary.items() if v is not None}
        # Define static features among the features (features, i.e. time-series, with lag value set to 0).
        static_features = [k for k,v in lags_dictionary.items() if v == 0]
        # Delete unused features.
        group = group.drop(columns = features_to_remove)        
        # Define the features (the names of the time-series).
        features = group.columns
        
        # Define the boolean mask for the creation of lag-features for the time-series.
        # Define the reference size of the window (time-step dimension).
        window_size = max(lags_dictionary.values())
        # Create mask based on lags into 'lags_dictionary' to pass over the input samples.
        mask = np.full(shape = (window_size, len(features)), fill_value = False)    
        for i, feature in enumerate(features):
            if row_output and feature in static_features:
                mask[:, i][-1] = True
            else:
                lags = lags_dictionary[feature]
                mask[:, i][-lags:] = True     
                
        # Create input samples.
        # Rolling a no masked window over the dataframe based on the maximum value of the 'lags_dictionary'.
        X = rolling_window(group.reset_index().values, window_size, axes = 0).swapaxes(1, 2)
        # Add the mask to the input samples based on lags.
        # Add the temporal information to the mask in order to always mantain the temporal information.
        mask_with_time = np.concatenate([np.expand_dims(np.array([True]*window_size), 1), mask], axis = 1)
        # Expand the mask to all the samples.
        mask_with_time = np.tile(mask_with_time, (X.shape[0], 1, 1))
        # Define input samples with defined lags (with also temporal information).
        X = np.ma.masked_array(X, mask = ~mask_with_time, fill_value = 0).filled(np.nan)
        
        # Create output samples.
        if single_step:
            y = rolling_window(group[target].reset_index().values[window_size:], 1, axes = 0)
        else:
            y = rolling_window(group[target].reset_index().values[window_size:], n_out, axes = 0)                        
     
        # Create column names for the features lags for dataframe output format if selected.
        if return_dataframe:
            # Input columns.
            columns_input = list()
            for feature_mask, feature in zip(mask.T, features):
                # Create columns values.
                if feature in static_features:
                    columns = ["%s" % feature]
                    columns_input.extend(columns)
                else:
                    columns = ["%s(t)" % feature if i == 1 else "%s(t-%d)" % (feature,i-1) for i in range(sum(feature_mask), 0, -1)]
                    columns_input.extend(columns)
            if feature_time:
                columns_input.extend(["Day", "Month", "Year"])    
            # Output columns.
            columns_output = ["x(t+%d)" % (i+1) for i in range(n_out)]
        else:
            columns_input = None
            columns_output = None  
        
        # Define some attributes of the class.
        self.X = X
        self.y = y
        self.group_name = group_name
        self.group = group
        self.n_out = n_out
        self.target = target
        self.mask = mask
        self.features = features
        self.return_dataframe = return_dataframe
        self.single_step = single_step
        self.feature_time = feature_time
        self.row_output = row_output
        self.columns_input = columns_input
        self.columns_output = columns_output
    
    def to_dataframe(self, X, y):
        """
        ***Sub-function***
 
        This function allows to convert the input and output samples into dataframes format.
        
        Parameters
        ----------
        X: the input samples.
        y: the output samples.

        """
        # Create dataframe of output samples.
        if y is not None:
            if self.single_step:
                columns_output = [self.columns_output[self.h-1]]
            else:
                columns_output = self.columns_output
            y = pd.DataFrame(y, columns = columns_output)
            y.columns.name = "Target|Prediction horizon"
        else:
            y = None
            
        # Create dataframe of input samples.
        if X is not None:
            # Create dataframe of input samples. 
            X = pd.DataFrame(X, columns = self.columns_input)                 
            X.columns.name = "Features|Lags"
        else:
            X = None

        return X, y
    
    def to_row_output(self, X, y):
        """
        ***Sub-function***
 
        This function allows to convert the input and output samples into row output format.
        
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
        else:
            y = None
            
        # Create dataframe of input samples.
        if X is not None:
            # Not consider temporal information for the input samples.
            X = X[:, :, 1:]
            # Flatten the lags of each sample over the rows.
            X = np.stack([x.flatten("F") for x in X])
            delnan_mask = np.frompyfunc(lambda i: i is np.nan, 1, 1)(X).astype(bool)
            X = np.ma.masked_array(X, mask = delnan_mask)
            X = np.ma.compress_rows(X.T).T

            # Add the temporal information to the input samples.
            if self.feature_time:
                if y is not None:
                    days = [date.day for date in dates]
                    months = [date.month for date in dates]
                    years = [date.year for date in dates]
                    # Create feature time.
                    dates = np.stack([days, months, years], axis = 1)
                    # Add to the data.
                    X = np.concatenate([X, dates], axis = 1)
                else:
                    day = (self.group.index[-1] + (self.h-1)*self.group.index.freq).day
                    month = (self.group.index[-1] + (self.h-1)*self.group.index.freq).month
                    year = (self.group.index[-1] + (self.h-1)*self.group.index.freq).year 
                    dates = np.array([[day], [month], [year]]).transpose()
                    X = np.concatenate([X, dates], axis = 1)
        else:
            X = None

        return X, y
        
    def to_supervised(self, h = None, validation = False, dtype = object):
        """
        ***Main function***
 
        This function allows to create training/validation/test samples to use for time-series forecasting purposes.
        
        Parameters
        ----------   
        h: the independent forecasting horizon to predict for the 'single_step' mode. If 'single_step = False', the 'h' parameter
           is not taken into account.      
        validation: if you want to create validation samples.
        dtype: the type of the output elements.
           
        Return
        ----------
        X_train: the training input samples.
        y_train: the training output samples.
        X_val: the validation input samples.
        y_val: the validation output samples.
        X_test: the test input sample.
    
        """
        # Check parameters.
        if self.single_step and h is None:
            raise ValueError("If 'single_step' is set, you must provide a value for the 'h' parameter.")
        if h is not None and h > self.n_out:
            raise ValueError("The 'h' parameter must be not greater than 'n_out' parameter.")      
        
        # Define some attributes of the class.
        self.h = h
        self.validation = validation

        # Splitting of the input X samples and the output y samples into training/validation/test.
        # Define the test sample input.
        X_test = self.X[-1:]
        # Define the training and validation samples input and outputs.
        if validation:
            if self.single_step:
                y_val = self.y[-self.n_out:]
                X_val = self.X[-(self.n_out+h):][:self.n_out]
                y = self.y[h-1:-self.n_out]
                X = self.X[:-(self.n_out+h)]
            else:
                y_val = self.y[-1:]
                X_val = self.X[:self.y.shape[0]][-1:]
                y = self.y[:-self.n_out]
                X = self.X[:-2*self.n_out]
        else:
            if self.single_step:
                y = self.y[h-1:]
                X = self.X[:y.shape[0]]
                X_val, y_val = None, None
            else:
                y = self.y
                X = self.X[:y.shape[0]]
                X_val, y_val = None, None
            
        # Samples arrays created until here with also temporal information: X, y, X_val, y_val, X_test.    
        self.X_draw = X
        self.y_draw = y
        self.X_val_draw = X_val # It could be None if 'validation = False'.
        self.y_val_draw = y_val # It could be None if 'validation = False'.
        self.X_test_draw = X_test
        
        if self.row_output:
            # Define input and output samples training dataframes.
            X_train, y_train = self.to_row_output(X, y)
            # Change the type of the dataframe.
            X_train = X_train.astype(dtype)
            y_train = y_train.astype(dtype)
            # Define input and output samples validation dataframes.
            X_val, y_val = self.to_row_output(X_val, y_val)
            # Change the type of the dataframe.
            if validation:
                X_val = X_val.astype(dtype)
                y_val = y_val.astype(dtype)
            # Define input samples test dataframes.
            X_test, _ = self.to_row_output(X_test, None)
            # Change the type of the dataframe.
            X_test = X_test.astype(dtype)            
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

        # In this last phase, the output format is changed if desired: array or dataframe outputs.
        if self.return_dataframe:
            # Define input and output samples training dataframes.
            X_train, y_train = self.to_dataframe(X_train, y_train)
            # Define input and output samples validation dataframes.
            X_val, y_val = self.to_dataframe(X_val, y_val)
            # Define input samples test dataframes.
            X_test, _ = self.to_dataframe(X_test, None)
            # Change the type of the dataframe.
            X_test = X_test
            # Save the name of the adminstrata as index for the test input sample.
            X_test.index = [self.group_name]
            
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
        if not self.row_output:
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
            if not self.row_output:
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
            self.X_draw = np.concatenate([self.X_draw[:2], self.X_draw[-2:]])
            self.y_draw = np.concatenate([self.y_draw[:2], self.y_draw[-2:]])
            train_dataframes = [self.highlight_cells(x, y) for x, y in zip(self.X_draw, self.y_draw)]
            if self.X_val_draw is not None:
                # Keep only boundaries samples.
                self.X_val_draw = np.concatenate([self.X_val_draw[:2], self.X_val_draw[-2:]])
                self.y_val_draw = np.concatenate([self.y_val_draw[:2], self.y_val_draw[-2:]])
                validation_dataframes = [self.highlight_cells(x, y) for x, y in zip(self.X_val_draw, self.y_val_draw)]
            else:
                validation_dataframes = None
            test_dataframes = [self.highlight_cells(x, None) for x in self.X_test_draw]
        else:
            train_dataframes = [self.highlight_cells(x, y) for x, y in zip(self.X_draw, self.y_draw)]
            if self.X_val_draw is not None:
                validation_dataframes = [self.highlight_cells(x, y) for x, y in zip(self.X_val_draw, self.y_val_draw)]
            else:
                validation_dataframes = None
            test_dataframes = [self.highlight_cells(x, None) for x in self.X_test_draw]
            
        return train_dataframes, validation_dataframes, test_dataframes
    