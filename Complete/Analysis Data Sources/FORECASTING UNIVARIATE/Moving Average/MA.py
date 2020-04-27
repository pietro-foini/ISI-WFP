from transformations import stationarity, box_cox, invboxcox
from statsmodels.tsa.arima_model import ARMA
from tool_datetime import next_datetimes
import pandas as pd
import numpy as np

class MA:
    """Moving Average method.
    
    The moving average (MA) method models the next step in the sequence as a linear function of the residual errors from a mean 
    process at prior time steps.
    
    Parameters
    ----------
    serie: the time-series to predict future values; it must be a pandas serie object with datetime values as index. 
      The frequency of the datatime index must be specified.
    q: the offset indicating the q-ultimate observations to use like predictors for the model.
    h: an int parameter indicating the prediction horizon.
    stationary: a bool parameter, whether or not you want to render the time-series stationary (if it is needed) through a first 
      differencing operation. The stationarity of the time-series is checked using an ADF and KPSS test.
    box_cox: a bool parameter, whether or not you want to apply a box cox transformation to the time-series.

    Returns
    ----------
    y_hats: a pandas serie object containing the predicted points by the algorithm.
       
    """
    def __init__(self, q, h, stationary = False, bxcx = False):
        self.q = q
        self.h = h
        self.stationary = stationary
        self.bxcx = bxcx
        
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
        
        # Train autoregression.
        model = ARMA(serie, (0, self.q))
        model_fit = model.fit()
        # Make predictions.
        y_hats = model_fit.predict(start = next_datetimes(serie.index[-1], self.h, freq)[0], 
                                   end = next_datetimes(serie.index[-1], self.h, freq)[-1], dynamic = False)
        
        if self.stationary: 
            y_hats.loc[y_hats.index[0]] = correction + y_hats.loc[y_hats.index[0]]
            y_hats = y_hats.cumsum()
        
        if BOX:
            y_hats = invboxcox(y_hats, lmbda)

        return y_hats
