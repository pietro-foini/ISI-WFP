import pandas as pd
import numpy as np
from IPython.display import display, Image
import matplotlib.pyplot as plt
import gif
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use("seaborn")

# Python module.
#
#
# Pietro Foini
#
# Year: 2020
    
class NestedCV:
    """NestedCV (Nested Cross-Validation for time-series forecasting).
    
    This function splits a pandas dataframe containing single or multiple time-series into n splits (multiple dataframes) in order 
    to perform a nested cross validation for time-series forecasting purposes. The splits is based on the end of the months.
    Consequently, the number of splits corresponds to the number of months you want to split out.

    """
    def __init__(self, n_splits, validation = False):
        """
        ***Initialization function***
 
        Initialization of the NestedCV class.
        
        Parameters
        ----------
        n_splits: the number of splits to obtain from the cross validation technique.
        validation: if you want to perform validation and test phases.

        Notes
        ----------
        If gap is set to 1 the nested cross validation is reduced to be the holdout validation.

        """
        # Define some attributes of the class.
        self.n_splits = n_splits
        self.validation = validation
    
    def get_splits(self, group, show = False, figsize = (20, 5), duration = 1500, dpi = 100):
        """
        ***Main function***
 
        This function allows to create the training/validation/test splits to use for time-series forecasting purposes.
        
        Parameters
        ----------   
        group: a pandas dataframe with as index a pandas datetime column with an appropriate frequency set. 
        show: if you want to show the splits through a gif; if True, the data will be automatically normalized only for the visualization.
        path: the path where the gif will be saved.
           
        Return
        ----------
        splits: a dictionary containg the training/validation/test sets for each split.
    
        """
        # Define the frequency of the group.
        freq = group.index.freq
        features = group.columns
        
        # Define granularity of the time-series.
        granularity_features = group.apply(lambda x: x.loc[x.first_valid_index():x.last_valid_index()].isnull().astype(int).groupby(x.loc[x.first_valid_index():x.last_valid_index()].notnull().astype(int).cumsum()).sum()).max().to_dict()
        # Define as no daily time-series those with some nan values inside their first and last valid index.
        no_daily_features = [k for k,v in granularity_features.items() if v != 0]

        # Create training, validation and test sets for each split.
        splits_dict = dict()
        for i,split_number in enumerate(reversed(range(self.n_splits))):
            index_test = group.last("%dM" % (split_number+1)).index
            train = group[~group.index.isin(index_test)]
            if self.validation:
                val = train.last("1M")
                train = train[~train.index.isin(val.index)]
            else:
                val = None
            test = group[group.index.isin(index_test)].first("1M")
            splits_dict[i+1] = (train, val, test)
        
        # Show the nested cross validation with a gif.
        if show:
            # Normalization.
            # N.B. The static time-series that are normalized to 1.
            min_group, max_group = group.min(), group.max()
            group = group / max_group
            group = group.loc[group.first_valid_index():group.last_valid_index()]
            frames = list()
            for split_number, (train, val, test) in splits_dict.items():
                # Normalization.
                train = train / max_group
                train = train.loc[train.first_valid_index():train.last_valid_index()]
                if self.validation:
                    val = val / max_group
                    val = val.loc[val.first_valid_index():val.last_valid_index()]
                test = test / max_group
                test = test.loc[test.first_valid_index():test.last_valid_index()]
                
                @gif.frame
                def plot():
                    f = plt.figure(figsize = figsize, dpi = dpi)
                    for feature in features:
                        # Check if daily time-series.
                        if feature in no_daily_features:
                            linestyle = ".-"
                        else:
                            linestyle = "-"
                        
                        group[feature].dropna().plot(c = "gray", style = ":", label = "Time-series")
                        train[feature].dropna().plot(c = "#1281FF", style = linestyle, label = "Train (%d)" % len(train))
                        if self.validation:
                            val[feature].dropna().plot(c = "orange", style = linestyle, label = "Validation (%d)" % len(val))
                        test[feature].dropna().plot(c = "red", style = linestyle, label = "Test (%d)" % len(test))
                        plt.title("Split %d" % split_number)
                        plt.autoscale()
                    # Legend.
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), title = "Sets:", loc = "center left", 
                               bbox_to_anchor = (1.0, 0.5))
                    
                frames.append(plot())
            gif.save(frames, "validation.gif", duration = duration)
            
            with open("validation.gif", "rb") as f:
                display(Image(data = f.read(), format = "png"))

        return splits_dict
    