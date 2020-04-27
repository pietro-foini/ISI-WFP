from transformations import stationarity, box_cox, invboxcox
from tool_datetime import next_datetimes
import pandas as pd
import numpy as np

class HES:
    """Holt's Exponential Smoothing method.
    
    The SES model when applied to temporal data that present increasing (or decreasing) linear behavior, provides predictions 
    which underestimate (or overestimate) the actual values. To avoid this systematic error, we can make use of methods as Holt’s
    Exponential Smoothing (HES). The HES model structure is similar to the SES method. However, besides to use the parameter  
    alpha to soften the level component, the algorithm uses a second smoothing constant beta for modeling the time-series trend.
    
    Parameters
    ----------
    serie: the time-series to predict future values; it must be a pandas serie object with datetime values as index. 
      The frequency of the datatime index must be specified.
    alpha: this parameter controls the rate at which the influence of the observations at prior time steps decay exponentially.  
      alpha is set to a value between 0 and 1.
    beta: this parameter is a smoothing constant beta for modeling the time-series trend. beta is set to a value between 0 and 1. 
    damped_factor: the damping parameter to prevent the forecast “go wild”. damped_factor is set to a value between 0 and 1
      (1 if you don't implement it).
    trend: the type of Holt's model between 'additive' and 'multiplicative'.
    h: an int parameter indicating the prediction horizon.
    stationary: a bool parameter, whether or not you want to render the time-series stationary (if it is needed) through a first 
      differencing operation. The stationarity of the time-series is checked using an ADF and KPSS test.
    box_cox: a bool parameter, whether or not you want to apply a box cox transformation to the time-series.

    Returns
    ----------
    y_hats: a pandas serie object containing the predicted points by the algorithm.
       
    """
    def __init__(self, alpha, beta, damped_factor, h, trend = "additive", stationary = False, bxcx = False):
        self.alpha = alpha
        self.beta = beta
        self.damped_factor = damped_factor
        self.trend = trend
        self.h = h
        self.stationary = stationary
        self.bxcx = bxcx
        
    def sum_damped(self, damp, n): 
        # Compute phi^1 + phi^2 + ... + phi^n.
        total = 0
        for i in range(n): 
            total += ((damp**(i + 1))) 
        return total  
            
    def additive(self, damped_factor, i):      
        y_hat = self.L + (self.sum_damped(damped_factor, i))*self.T
        return y_hat
    
    def multiplicative(self, damped_factor, i):      
        y_hat = self.L*self.T**(self.sum_damped(damped_factor, i))
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
        
        # Initial condition on L.
        L = serie.values[0]
        # Additive model.
        if self.trend == "additive":
            T = serie.values[1] - serie.values[0]
            for z in serie.values[1:]:
                L_recursive = self.alpha*z + (1 - self.alpha)*(L + T*self.damped_factor)
                T_recursive = self.beta*(L_recursive - L) + (1 - self.beta)*T*self.damped_factor
                L = L_recursive
                T = T_recursive
            self.L = L  
            self.T = T
         
        if self.trend == "multiplicative":
            T = serie.values[1]/serie.values[0]
            for z in serie.values[1:]:
                L_recursive = self.alpha*z + (1 - self.alpha)*(L*(T**self.damped_factor))
                T_recursive = self.beta*(L_recursive /L) + (1 - self.beta)*(T**self.damped_factor)
                L = L_recursive
                T = T_recursive
            self.L = L  
            self.T = T
        
        # Make predictions.
        predictions = list()
        for i in range(self.h):
            if self.trend == "additive":
                y_hat = self.additive(self.damped_factor, i + 1)
            if self.trend == "multiplicative":
                y_hat = self.multiplicative(self.damped_factor, i + 1)
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