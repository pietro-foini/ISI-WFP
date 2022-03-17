import pandas as pd
import numpy as np

# Python module.
#
#
# Pietro Foini
#
# Year: 2020
    
class NestedCV:
    """NestedCV (Nested Cross-Validation for time-series forecasting).
    
    This class aims to split a pandas dataframe containing single or multiple time-series into 
    'n_splits' dataframes in order to perform a nested cross validation for time-series forecasting purposes. 

    """
    def __init__(self, n_splits, test_size):
        """
        Initialization of the NestedCV class.
        
        Parameters
        ----------
        n_splits: the number of splits to obtain from the nested cross validation technique.
        test_size: the length of the test set for each split.

        """
        # Define some attributes of the class.
        self.n_splits = n_splits
        self.test_size = test_size
    
    def get_splits(self, group):
        """
        This function allows to create the training/test splits to use for time-series forecasting purposes.
        
        Parameters
        ----------   
        group: a pandas dataframe with as index a pandas datetime column with an appropriate frequency set. 
           
        Return
        ----------
        splits: a dictionary containg the training/test sets for each split.
        
        Notes
        ----------
        The splits that contains a number of points in test data less than 'test_size' are not considered.
    
        """
        # Define the frequency of the group.
        freq = group.index.freq

        # Create training and test sets for each split.
        splits_dict = dict()
        for i, split_number in enumerate(reversed(range(self.n_splits))):
            index_test = pd.date_range(group.last("%dM" % (split_number+1)).index[0], periods = self.test_size, freq = freq)
            train = group.loc[:index_test[0] -1*freq]
            test = group[group.index.isin(index_test)]
            if len(test) < self.test_size:
                print(f"Warning. Split {i+1} is discarded because the corresponding number of test points is less than 'test_size'!")
            else:
                splits_dict[i+1] = (train, test)

        return splits_dict
    