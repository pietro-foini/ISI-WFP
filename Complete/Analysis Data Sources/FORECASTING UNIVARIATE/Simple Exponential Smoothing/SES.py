from transformations import stationarity, box_cox, invboxcox
from tool_datetime import next_datetimes
import pandas as pd
import numpy as np

class SES:
    """Simple Exponential Smoothing method.
    
    The SES method is analogous to SMA, except by the fact that each series value receives a different weight. 
    The weights increase exponentially over time so that the most recent observations exert more influence on the 
    calculation of future predictions. 
    
    Parameters
    ----------
    serie: the time-series to predict future values; it must be a pandas serie object with datetime values as index. 
      The frequency of the datatime index must be specified.
    alpha: this parameter controls the rate at which the influence of the observations at prior time steps decay exponentially.  
      alpha is set to a value between 0 and 1.
    h: an int parameter indicating the prediction horizon.
    stationary: a bool parameter, whether or not you want to render the time-series stationary (if it is needed) through a first 
      differencing operation. The stationarity of the time-series is checked using an ADF and KPSS test.
    box_cox: a bool parameter, whether or not you want to apply a box cox transformation to the time-series.

    Returns
    ----------
    y_hats: a pandas serie object containing the predicted points by the algorithm.
       
    """
    def __init__(self, alpha, h, stationary = False, bxcx = False):
        self.alpha = alpha
        self.h = h
        self.stationary = stationary
        self.bxcx = bxcx
        
    def algorithm(self, serie, alpha):
        y_hat = alpha*serie[-1] + (1 - alpha)*self.L
        self.L = y_hat
        return y_hat
        
    def predict(self, serie):
        freq = serie.index.freq
        
        # Apply box cox transformation.
        if self.bxcx and all(serie > 0):
            BOX = True
            serie, lmbda = box_cox(serie)
        else:
            BOX = False
        
        # Apply stationarity (first difference order) if needed.
        if self.stationary:
            serie, correction = stationarity(serie)
        else:
            correction = 0
        
        # Compute L recursive starting from L0.
        L = serie.values[0]        
        for z in serie.values[1:-1]:
            L_recursive = self.alpha*z + (1 - self.alpha)*L
            L = L_recursive
        self.L = L
        
        predictions = list()
        # Make predictions.
        for i in range(self.h):
            y_hat = self.algorithm(serie, self.alpha)
            y_hat = pd.Series(y_hat, index = next_datetimes(serie.index[-1], 1, freq), name = serie.name)
            serie = serie.append(y_hat)
            
            # Return to original scale using a cumsum method.
            predictions.append(correction + serie[-1])
          
            if self.stationary:
                correction = predictions[-1]

        y_hats = pd.Series(predictions, index = serie[-self.h:].index)

        if BOX:
            y_hats = invboxcox(y_hats, lmbda)

        return y_hats