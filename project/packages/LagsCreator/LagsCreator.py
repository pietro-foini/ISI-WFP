from IPython.display import display, Image
import dataframe_image as dfi
from itertools import chain
import numpy as np
import pandas as pd
import imageio
import shutil
import os

# Python module.
#
#
# Pietro Foini
#
# Year: 2020

class LagsCreator:
    """LagsCreator.
    
    This module allows to create lag-features for time-series forecasting purposes. It supports different configurations 
    to get the outputs into several formats. It is also possible to visualize the lag-features through 
    an highlighting of the cells of the dataframe (see visualization method).
    
    """
    def __init__(self, group, lags_dictionary, target, return_dataframe = False, feature_time = None):
        """
        Initialization of the LagsCreator class.
        
        Parameters
        ----------
        group: a pandas dataframe with single/multiple columns representing the time-series. The dataframe must have as 
           index a pandas datetime column with an appropriate frequency set. 
        lags_dictionary: a dictionary containing the lag values (array object) corresponding to each time-series (the names of the 
           time-series must be the keys of the dictionary which will be associated with the corresponding lag values). If you don't want 
           to use a time-series, the corresponding value in the dictionary must be set to 'None'. Each column must have set a 
           specified value.
        target: a string containing the name of the time-series that you want to predict. The target variable must have 
           a lag value different from 'None' into the 'lags_dictionary'.
        return_dataframe: the modality to set in order to have the outputs returned as pandas dataframes.
        feature_time: if you want to create a feature time to add as feature in the input samples. This parameter has to be a 
           list containing the time information you want to extract from data.
           
        """        
        # Check parameters.
        symmetric_difference = set(lags_dictionary.keys()).symmetric_difference(set(group.columns))
        if symmetric_difference != set():
            raise ValueError("You have to provide a lag value for each time-series stored in the input dataframe. Please check the 'lags_dictionary' parameter. More precisely, check the features %s." % str(symmetric_difference))

        # The features whose are specified into the 'lags_dictionary' with None values are not considered as predictors.        
        features_to_remove = [k for k,v in lags_dictionary.items() if v is None]
        # Update the 'lags_dictionary' not considering the features with None values.
        lags_dictionary = {k: v for k,v in lags_dictionary.items() if v is not None}
        # Define the names of the static features among all the features (features, i.e. time-series, with lag value set to 0).
        static_features = [k for k,v in lags_dictionary.items() if type(v) == int and v == 0]
        # Delete unused features to the input dataframe.
        group = group.drop(columns = features_to_remove) 
        # Define all the remaining features.
        features = group.columns
        
        # Define the last temporal index of the first expanding window such that it is possible to start collecting valid data for each time-series.
        index = max(group[group.columns.difference(static_features)].apply(lambda x: x.dropna().iloc[:np.max(lags_dictionary[x.name])].index[-1]).values)
        subgroup = group.loc[:index]
        
        # Check if dataframe contains some nan values. If not, the procedure to create the samples is build to be faster.
        nans = group.isnull().sum().sum()

        # Create input samples using two procedures.
        if nans > 0:
            # Create all the expanding windows using the numpy roll function until the end of the input dataframe.
            difference = len(group) - len(subgroup)
            group_np = group.reset_index().values
            expanding_windows = np.stack([np.roll(group_np, difference-i, axis = 0) for i in range(difference+1)])

            # Create mask based on lags into 'lags_dictionary' to pass over the input samples.
            mask = np.full(shape = (expanding_windows.shape[1], len(features)+1), fill_value = False) 
            # Set the value for the time column.
            mask[:, 0] = True

            # Create mask values for the features of each sample.
            def get_(x):
                mask_x = mask.copy()
                for i, feature in enumerate(features):
                    if feature in static_features:
                        mask_x[:, i+1][-1] = True
                    else:
                        lags = np.argwhere(~np.isnan(x[:,i+1].astype(float))).flatten()[-lags_dictionary[feature]]
                        mask_x[:, i+1][lags] = True          
                # Create input sample using a mask.
                x = np.ma.masked_array(x, mask = ~mask_x, fill_value = 0).filled(np.nan)
                return x

            # Input samples.
            X = np.stack(list(map(get_, expanding_windows)))
        else:
            # Define the boolean mask for the creation of lag-features for the time-series.
            # Define the reference size of the window (time-step dimension).
            window_size = max(map(np.max, list(lags_dictionary.values())))
            # Create mask based on lags into 'lags_dictionary' to pass over the input samples.
            mask = np.full(shape = (window_size, len(features)), fill_value = False)   
            for i, feature in enumerate(features):
                if feature in static_features:
                    mask[:, i][-1] = True 
                else:
                    lags = lags_dictionary[feature]
                    mask[:, i][-lags] = True 

            # Create input samples.
            # Rolling a no masked window over the dataframe based on the maximum value of the 'lags_dictionary'.
            X = self.rolling_window(group.reset_index().values, window_size)
            # Add the mask to the input samples based on lags.
            # Add the temporal information to the mask in order to always mantain the temporal information.
            mask_with_time = np.concatenate([np.expand_dims(np.array([True]*window_size), 1), mask], axis = 1)
            # Expand the mask to all the samples.
            mask_with_time = np.tile(mask_with_time, (X.shape[0], 1, 1))
            # Define input samples with defined lags (with also temporal information). 
            # Shape: (n_samples, max lag, n_features + 1). +1 is referred to the temporal information. 
            X = np.ma.masked_array(X, mask = ~mask_with_time, fill_value = 0).filled(np.nan) 

        # Create output samples. 
        # Shape: (n_samples, n_prediction_horizons, 1 + 1). +1 is referred to the temporal information. 
        y = self.rolling_window(group[target].reset_index().values[len(subgroup):], 1)

        # Define test point.
        X_test = X[-1:]

        # Create column names for the features lags.
        # Input columns.
        columns_input_0 = list()
        columns_input_1 = list()
        for feature in features:
            # Create columns values.
            if feature in static_features:
                columns_0 = [feature]
                columns_1 = ["x"]
                columns_input_0.extend(columns_0)
                columns_input_1.extend(columns_1)
            else:
                lags = lags_dictionary[feature]
                columns_0 = [feature]*len(lags)
                columns_1 = ["x(t)" if i == 1 else "x(t-%d)" % (i-1) for i in reversed(lags)]
                columns_input_0.extend(columns_0)
                columns_input_1.extend(columns_1)

        if feature_time is not None:
            columns_input_0.extend(feature_time)    
            columns_input_1.extend(["x"]*len(feature_time))    
        # Create multi-index columns for input samples.
        iterables_input = list(zip(*[columns_input_0, columns_input_1]))
        columns_input = pd.MultiIndex.from_tuples(iterables_input, names = ["Features", "Lags"])

        # Define some attributes of the class.
        self.X = X
        self.y = y
        self.X_test = X_test
        self.group = group
        self.target = target
        self.features = features
        self.nans = nans
        self.return_dataframe = return_dataframe
        self.feature_time = feature_time
        self.columns_input = columns_input
        
    def rolling_window(self, x, window):
        """
        This function allows to rolling a window over a numpy array.
        
        Parameters
        ----------
        x: the input array.
        window: the length of the window to slide.

        """
        # Set shape.
        shape = list(x.shape)
        shape[0] = x.shape[0] - window + 1
        shape.insert(len(shape)-1, window)
        # Set strides.
        strides = list(x.strides)
        strides.insert(0, strides[0])
        return np.lib.stride_tricks.as_strided(x, shape = tuple(shape), strides = tuple(strides))
    
    def to_dataframe(self, X, y):
        """
        This function allows to convert the input and output samples into dataframes format.
        
        Parameters
        ----------
        X: the input samples.
        y: the output samples.

        """
        # Create dataframe of output samples.
        if y is not None:
            if self.single_step:
                # Create multi-index columns for output samples.
                iterables_output = list(zip(*[[self.target], ["x(t+%d)" % (self.h)]]))
                columns_output = pd.MultiIndex.from_tuples(iterables_output, names = ["Target", "Prediction horizon"])
            else:
                # Create multi-index columns for output samples.
                iterables_output = list(zip(*[[self.target]*self.h, ["x(t+%d)" % (i+1) for i in range(self.h)]]))
                columns_output = pd.MultiIndex.from_tuples(iterables_output, names = ["Target", "Prediction horizon"])
            y = pd.DataFrame(y, columns = columns_output)
        else:
            y = None
            
        # Create dataframe of input samples.
        if X is not None:
            # Create dataframe of input samples. 
            X = pd.DataFrame(X, columns = self.columns_input)                 
        else:
            X = None

        return X, y
    
    def to_row_output(self, X, y):
        """
        This function allows to convert the input and output samples into row output format.
        
        Parameters
        ----------
        X: the input samples with the column of temporal information of shape (n_samples, timesteps, n_features).
        y: the output samples with the row of temporal information of shape (n_samples, 2, n_out]).

        """
        # Create dataframe of output samples.
        if y is not None:
            # Consider the temporal information.
            dates = y[:, :, 0].flatten()
            # Not consider temporal information.
            y = y[:, :, 1:]
            y = y.reshape(y.shape[0], y.shape[1])
        else:
            y = None
            
        # Create dataframe of input samples.
        if X is not None:
            # Not consider temporal information for the input samples.
            X = X[:, :, 1:]
            # Flatten the lags of each sample over the rows.
            X = X.reshape((X.shape[0], -1), order = "F")
            if self.nans > 0:
                # Delete nan columns.
                X = np.stack(list(map(lambda x: x[~np.isnan(x.astype(float))], X)))
            else:
                # Delete nan columns.
                X = X[:, ~np.isnan(X.astype(float)).any(axis = 0)] 

            # Add the temporal information to the input samples.
            if self.feature_time is not None:
                if self.single_step:
                    if y is not None:
                        temporal_features = list()
                        for feature in self.feature_time:
                            if feature is "Day":
                                days = [date.day for date in dates]
                                temporal_features.append(days)
                            if feature is "Month":
                                months = [date.month for date in dates]
                                temporal_features.append(months)
                            if feature is "Year":
                                years = [date.year for date in dates]
                                temporal_features.append(years)
                            if feature is "Dayofweek":
                                dayofweek = [date.dayofweek for date in dates]
                                temporal_features.append(dayofweek)
                        # Create feature time.
                        dates = np.stack(temporal_features, axis = 1)
                        # Add to the data.
                        X = np.concatenate([X, dates], axis = 1)
                    else:
                        # Create feature time.
                        temporal_features = list()
                        for feature in self.feature_time:
                            if feature is "Day":
                                day = (self.group.index[-1] + (self.h)*self.group.index.freq).day
                                temporal_features.append(day)
                            if feature is "Month":
                                month = (self.group.index[-1] + (self.h)*self.group.index.freq).month
                                temporal_features.append(month)
                            if feature is "Year":
                                year = (self.group.index[-1] + (self.h)*self.group.index.freq).year
                                temporal_features.append(year)
                            if feature is "Dayofweek":
                                dayofweek = (self.group.index[-1] + (self.h)*self.group.index.freq).dayofweek
                                temporal_features.append(dayofweek)

                        dates = np.expand_dims(temporal_features, 0)
                        # Add to the data.
                        X = np.concatenate([X, dates], axis = 1)
        else:
            X = None

        return X, y
        
    def to_supervised(self, h = None, step = None, single_step = False, dtype = object):
        """
        This function allows to create the input X and output y samples to use for time-series forecasting purposes.
        
        Parameters
        ----------   
        h: the forecasting horizon.     
        step: the temporal step/shift between the samples.
        single_step: if set, each prediction horizon is created independently from the others.
        dtype: the type returned for the input X and output y samples.
           
        Return
        ----------
        X: the input samples.
        y: the output samples.
    
        """       
        # Check parameters.
        if self.feature_time and not single_step:
            raise ValueError("You can use the 'feature_time' only if you are working in the 'single_step' mode.")
            
        # Define some attributes of the class.
        self.h = h
        self.single_step = single_step
        
        # Define the input and output samples.
        if self.single_step:
            y = self.y[h-1:self.y.shape[0]]
            X = self.X[:y.shape[0]]
        else:
            y = self.rolling_window(self.y, h)
            y = y.reshape(y.shape[0], y.shape[2], y.shape[3])
            X = self.X[:y.shape[0]]
        
        # Set temporal step between samples.
        if step is not None:
            # Keep the samples with a step between them but keeping always the last sample.
            indx = [i for i in chain(range(0, X.shape[0]-1, step), [X.shape[0]-1])]
            X = X[indx]
            y = y[indx]

        # Save these samples arrays with temporal information to use for the visualization.    
        self.X_draw = X
        self.y_draw = y
        self.X_test_draw = self.X_test

        # Define input and output samples training dataframes.
        X, y = self.to_row_output(X, y)
        # Change the type of the samples.
        X = X.astype(dtype)
        y = y.astype(dtype)    
        # Define the test input.
        X_test, _ = self.to_row_output(self.X_test, None)
        # Change the type of the samples.
        X_test = X_test.astype(dtype)     

        # The output format is changed from array to dataframe.
        if self.return_dataframe:
            # Define input and output samples training dataframes.
            X, y = self.to_dataframe(X, y)
            # Define input samples test dataframes.
            X_test, _ = self.to_dataframe(X_test, None)
            
        return X, y, X_test   
    
    def highlight_cells(self, x, y):
        """
        This function draws the cells of the dataframe that belongs to the lag features for the current input sample x and
        output sample y.

        """
        # Define a group for style where the values inside the dataframe are converted to strings.
        group_style = self.group.round(4).astype(str)
        group_style.index = list(map(lambda x: str(x.date()), group_style.index))
        
        # Define the mask of the current sample.
        mask = ~np.isnan(x[:,1:].astype(float))

        # Pandas mask.
        m = pd.DataFrame(mask, index = list(map(lambda x: str(x.date()), x[:, 0])), columns = self.features)
        # Getting (index, column) pairs for True elements of the boolean DataFrame.
        cells_to_color_input = m[m == True].stack().index.tolist()
        if y is not None:
            cells_to_color_output = list(map(lambda x: str(x.date()), y[:, 0]))

        def draw(x):
            df_styler  = group_style.copy()
            # Set particular cell colors for the input.
            for location in cells_to_color_input:
                df_styler.loc[location[0], location[1]] = "background-color: RGB(0,131,255)"
            # Set particular cell colors for the output.
            if y is not None:
                for location in cells_to_color_output:
                    df_styler.loc[location, self.target] = "background-color: RGB(255,154,0)"
            return df_styler 
        
        # Highlight the following sample into the dataframe.
        sample = group_style.style.apply(lambda x: draw(x), axis = None)
        return sample
        
    def visualization(self, boundaries = True, gif = False, fps = 1, width = 150):
        """
        This function allows to visualize the input and output samples created by the process.
        
        Parameters
        ----------
        boundaries: if you want to visualize only the first two an the last two sample points created.
        gif: if you want to create a gif showing the rolling samples.
        fps: frame per second of the created gif.
        width: the with of the created gif
        
        Return
        ----------
        highlight_dataframes: a list of dataframes that underline each sample created.

        """
        # Create dataframes for visualization.
        if boundaries:
            # Keep only boundaries samples.
            self.X_draw = np.concatenate([self.X_draw[:2], self.X_draw[-2:]])
            self.y_draw = np.concatenate([self.y_draw[:2], self.y_draw[-2:]])
            highlight_dataframes_train = [self.highlight_cells(x, y) for x, y in zip(self.X_draw, self.y_draw)]
            highlight_dataframes_test = [self.highlight_cells(x, None) for x in self.X_test_draw]
        else:
            highlight_dataframes_train = [self.highlight_cells(x, y) for x, y in zip(self.X_draw, self.y_draw)]
            highlight_dataframes_test = [self.highlight_cells(x, None) for x in self.X_test_draw]
            
        if gif:
            # Save the samples into figures.
            dir = "./visualization"
            if not os.path.exists(dir):
                os.makedirs(dir)
            else:
                shutil.rmtree(dir)           
                os.makedirs(dir)
            # Save figures.
            images = list()
            for i,train in enumerate(highlight_dataframes_train):
                # Create figure from dataframe styler.
                train.export_png(dir + "/%d.png" % i)
                images.append(imageio.imread(dir + "/%d.png" % i))
                
            # Create gif.
            imageio.mimwrite(dir + "/GIF.gif", images, fps = fps)
            
            with open(dir + "/GIF.gif", "rb") as f:
                display(Image(data = f.read(), format = "png", width = width))
            
        return highlight_dataframes_train, highlight_dataframes_test
    