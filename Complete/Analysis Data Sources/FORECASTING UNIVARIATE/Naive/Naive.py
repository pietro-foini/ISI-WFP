from tool_datetime import next_datetimes
import pandas as pd

class Naive:
    """Naive method.
    
    Naive model is a persistence model that simply uses the last observation of the corresponding time-series 
    as prediction of the next time step. This simple approach can be adjusted slightly for seasonal data. In this case, 
    the observation at the same time in the previous cycle may be persisted instead. 
    
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
            y_hat = serie[-self.n]
            y_hat = pd.Series(y_hat, index = next_datetimes(serie.index[-1], 1, freq), name = serie.name)
            serie = serie.append(y_hat)
            
        y_hats = serie[-self.h:]

        return y_hats