import pandas as pd
import numpy as np
import os

# Python module.
#
#
# Pietro Foini
#
# Year: 2020
    
class NestedCV:
    """NestedCV (Nested Cross-Validation for time-series forecasting).
    
    This function splits a pandas dataframe containing single or multiple time-series into n splits (multiple dataframes) in order 
    to perform a nested cross validation for time-series forecasting purposes. The splits is based on the end of the months up 
    to 'test_size' points. Consequently, the number of splits corresponds to the number of months you want to split out.

    """
    def __init__(self, n_splits, test_size):
        """
        ***Initialization function***
 
        Initialization of the NestedCV class.
        
        Parameters
        ----------
        n_splits: the number of splits to obtain from the cross validation technique.
        test_size: the length of the test set for each split.

        Notes
        ----------
        If gap is set to 1 the nested cross validation is reduced to be the holdout validation.

        """
        # Define some attributes of the class.
        self.n_splits = n_splits
        self.test_size = test_size
    
    def get_splits(self, group):
        """
        ***Main function***
 
        This function allows to create the training/validation/test splits to use for time-series forecasting purposes.
        
        Parameters
        ----------   
        group: a pandas dataframe with as index a pandas datetime column with an appropriate frequency set. 
           
        Return
        ----------
        splits: a dictionary containg the training/validation/test sets for each split.
    
        """
        # Define the frequency of the group.
        freq = group.index.freq

        # Create training, validation and test sets for each split.
        splits_dict = dict()
        for i,split_number in enumerate(reversed(range(self.n_splits))):
            index_test = pd.date_range(group.last("%dM" % (split_number+1)).index[0], periods = self.test_size, freq = freq)
            train = group.loc[:index_test[0] -1*freq]
            test = group[group.index.isin(index_test)]
            splits_dict[i+1] = (train, test)

        return splits_dict
    