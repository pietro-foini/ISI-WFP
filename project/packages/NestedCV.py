import pandas as pd
import numpy as np
from IPython.display import display, Image
import matplotlib.pyplot as plt
from rolling_window import rolling_window
import gif
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Python module.
#
#
# Pietro Foini
#
# Year: 2020
    
class NestedCV:
    """NestedCV (Nested Cross-Validation for time-series forecasting).
    
    This function divides a group of time-series into k folds in order to perform a nested cross validation for time-series
    forecasting.

    """
    def __init__(self, k_folds, validation_size = 0.1, test_size = 0.1, gap = 0.7, TimeSeriesSplit = False):
        """
        ***Initialization function***
 
        Initialization of the NestedCV class.
        
        Parameters
        ----------
        k_folds: the the number of folds to obtain.
        validation_size: the fraction of training points used for validation.
        test_size: the fraction of training points used for test.
        gap: the fraction of the training time-series to which generate the size of the rolling window and
           so determining the level of freedom to move the window.
        TimeSeriesSplit: split function according to sklearn package for time-series.

        Notes
        ----------
        If gap is 1 the rolling validation is reduced to be the holdout validation.

        """
        # Define some attributes of the class.
        self.k_folds = k_folds
        self.validation_size = validation_size
        self.test_size = test_size
        self.gap = gap
        self.TimeSeriesSplit = TimeSeriesSplit
    
    def get_splits(self, group, show = False, path = None):
        """
        ***Main function***
 
        This function allows to create training/validation/test samples to use for time-series forecasting purposes.
        
        Parameters
        ----------   
        group: a pandas dataframe with two hierarchical multi-index on axis 1: the level 0 corresponding to a single main group and 
           the level 1 corresponding to single/multiple time-series. The dataframe must have as index a pandas datetime column 
           with an appropriate frequency set. 
        show: if you want to show the splits through a gif; the data will be automatically normalized.
        path: the path where the gif will be saved.
           
        Return
        ----------
        splits: a nested list containg the training/validation/test sets.
    
        """
        freq = group.index.freq

        # Define size all data, validation and test.
        window = int((len(group)*self.gap))
        num_validation = int(window*self.validation_size)
        num_test = int(window*self.test_size)

        # Create the datetime folds.
        datetime_folds = rolling_window(group.index, window, axes = 0)

        # Select k folds evenly spaced including first and last ones.
        idx = np.round(np.linspace(0, len(datetime_folds) - 1, self.k_folds)).astype(int)
        k_datetime_folds = datetime_folds[idx]
        
        # Create folds.
        folds = list()
        for idx in datetime_folds[idx]:
            folds.append(group.loc[idx])

        # Create training, validation and test sets for each fold.
        splits = list()
        for i, fold in enumerate(folds):
            if self.TimeSeriesSplit:
                train = fold[:-(num_validation+num_test)]
                train = pd.concat([group[group.index[0]:train.index[0] - 1*freq], train])
                val = fold[-(num_validation+num_test):-num_test]
                test = fold[-num_test:]
                splits.append((train, val, test))
            else:
                train = fold[:-(num_validation+num_test)]
                val = fold[-(num_validation+num_test):-num_test]
                test = fold[-num_test:]
                splits.append((train, val, test))
        
        # Show the nested cross validation with a gif.
        if show:
            # Normalization.
            min_group, max_group = group.min(), group.max()
            group = group / max_group
            frames = list()
            for train, val, test in splits:
                # Normalization.
                train = train / max_group
                val = val / max_group
                test = test / max_group
                
                @gif.frame
                def plot():
                    f = plt.figure(figsize = (20, 5))
                    plt.plot(group, c = "gray", linestyle = ":", label = "Series")
                    plt.plot(train, color = "#1281FF", label = "Train (%d)" % len(train))
                    plt.plot(val, color = "orange", label = "Validation (%d)" % len(val))
                    plt.plot(test, color = "red", label = "Test (%d)" % len(test))
                    plt.autoscale()
                    # Legend.
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), title = "Set", loc = "center left", 
                               bbox_to_anchor = (1.0, 0.5))
                    
                frames.append(plot())
            gif.save(frames, path + "/validation.gif", duration = 1500)
            
            with open(path + "/validation.gif", "rb") as f:
                display(Image(data = f.read(), format = "png"))

        return splits
    