from tool_datetime import next_datetimes
import pandas as pd
import numpy as np

class SMA:
    """Simple Moving Average method.
    
    The SMA model of order n is a simple technique that performs an arithmetic average of the last n values of 
    time-series to predict the next value.
    
    Parameters
    ----------
    serie: the time-series to predict future values; it must be a pandas serie object with datetime values as index. 
      The frequency of the datatime index must be specified.
    n: the offset indicating the n-ultimate observations to use like predictors for the model. 
    h: an int parameter indicating the prediction horizon.

    Returns
    ----------
    y_hats: a pandas serie object containing the predicted points by the algorithm.
       
    """
    def __init__(self, n, h):
        self.n = n
        self.h = h
        
    def predict(self, serie):
        freq = serie.index.freq
        # Make predictions.
        for i in range(self.h):
            y_hat = np.mean(serie[-self.n:])
            y_hat = pd.Series(y_hat, index = next_datetimes(serie.index[-1], 1, freq), name = serie.name)
            serie = serie.append(y_hat)
            
        y_hats = serie[-self.h:]

        return y_hats