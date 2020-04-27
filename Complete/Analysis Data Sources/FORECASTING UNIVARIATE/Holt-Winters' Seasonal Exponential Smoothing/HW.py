from transformations import stationarity, box_cox, invboxcox
from tool_datetime import next_datetimes
import pandas as pd
import numpy as np

class HW:
    """Holt-Winters' Seasonal Exponential Smoothing method.
    
    The idea behind triple exponential smoothing (Holt-Winters Method) is to apply exponential smoothing to a third component,
    seasonality. This means we should not be using this method if our time series is not expected to have seasonality. 
    
    Parameters
    ----------
    serie: the time-series to predict future values; it must be a pandas serie object with datetime values as index. 
      The frequency of the datatime index must be specified.
    alpha: this parameter controls the rate at which the influence of the observations at prior time steps decay exponentially.  
      alpha is set to a value between 0 and 1.
    beta: this parameter is a smoothing constant beta for modeling the time-series trend. beta is set to a value between 0 and 1. 
    gamma: this parameter is a smoothing constant beta for modeling the time-series seasonality. gamma is set to a value between 0 and 1. 
    s: an in numer indicating the cycle period of the season, e.g. for a daily dataset with a season of a year, 's' must be set to 365.
    damped_factor: the damping parameter to prevent the forecast “go wild”. damped_factor is set to a value between 0 and 1
      (1 if you don't implement it).
    seasonal: the type of Holt's model between 'additive' and 'multiplicative' for season component.
    h: an int parameter indicating the prediction horizon.
    stationary: a bool parameter, whether or not you want to render the time-series stationary (if it is needed) through a first 
      differencing operation. The stationarity of the time-series is checked using an ADF and KPSS test.
    box_cox: a bool parameter, whether or not you want to apply a box cox transformation to the time-series.

    Returns
    ----------
    y_hats: a pandas serie object containing the predicted points by the algorithm.
       
    """
    def __init__(self, alpha, beta, gamma, s, damped_factor, h, seasonal = "additive", stationary = False, bxcx = False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.s = s
        self.damped_factor = damped_factor
        self.h = h
        self.seasonal = seasonal
        self.stationary = stationary
        self.bxcx = bxcx
        
    def sum_damped(self, damp, n): 
        # Compute phi^1 + phi^2 + ... + phi^n.
        total = 0
        for i in range(n): 
            total += ((damp**(i + 1))) 
        return total  
            
    def AHW(self, damped_factor, i):  
        y_hat = self.L + (self.sum_damped(damped_factor, i))*self.T + self.S[-(self.s - (i - 1)%self.s)]
        return y_hat
    
    def MHW(self, damped_factor, i):  
        y_hat = (self.L + (self.sum_damped(damped_factor, i))*self.T)*self.S[-(self.s - (i - 1)%self.s)]
        return y_hat
        
    def predict(self, serie):
        # Define the frequency of the time-series.
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
            
        # To initialize the HW method we need at least two complete season's data.
        if len(serie) <= self.s*2:
            raise ValueError("To initialize the HW method we need at least two complete season's data.")  
        
        # Define initial conditions.        
        # Additive model (AHW).
        if self.seasonal == "additive":
            # Initial condition on L: average of the first seasonal station.
            L = np.mean(serie[:self.s])
            # Initial condition on T: using two complete seasonal stations.
            T = np.mean((serie[self.s:self.s*2].values - serie[:self.s].values)/self.s)
            # Initial conditions on S (depending on seasonal type).
            S = serie[:self.s].values - L
            S = list(S)
            
            # Compute L, T, S recursively.
            for z in serie.values[self.s:]:
                L_recursive = self.alpha*(z - S[-self.s]) + (1 - self.alpha)*(L + T*self.damped_factor)
                T_recursive = self.beta*(L_recursive - L) + (1 - self.beta)*T*self.damped_factor
                S_recursive = self.gamma*(z - L - self.damped_factor*T) + (1 - self.gamma)*S[-self.s]
                L = L_recursive
                T = T_recursive
                S.append(S_recursive)
            self.L = L  
            self.T = T
            self.S = S
            
        # Multiplicative model (MHW).
        if self.seasonal == "multiplicative":
            # Initial condition on L: average of the first seasonal station.
            L = np.mean(serie[:self.s])
            # Initial condition on T: using two complete seasonal stations.
            T = np.mean((serie[self.s:self.s*2].values - serie[:self.s].values)/self.s)
            # Initial conditions on S (depending on seasonal type).
            S = serie[:self.s].values/L
            S = list(S)
            
            # Compute L, T, S recursively.
            for z in serie.values[self.s:]:
                L_recursive = self.alpha*(z/S[-self.s]) + (1 - self.alpha)*(L + T*self.damped_factor)
                T_recursive = self.beta*(L_recursive - L) + (1 - self.beta)*T*self.damped_factor
                S_recursive = self.gamma*(z/(L + self.damped_factor*T)) + (1 - self.gamma)*S[-self.s]
                L = L_recursive
                T = T_recursive
                S.append(S_recursive)
            self.L = L  
            self.T = T
            self.S = S

        # Make predictions.
        predictions = list()
        for i in range(self.h):
            if self.seasonal == "additive":
                y_hat = self.AHW(self.damped_factor, i + 1)
            if self.seasonal == "multiplicative":
                y_hat = self.MHW(self.damped_factor, i + 1)
                
            y_hat = pd.Series(y_hat, index = next_datetimes(serie.index[-1], 1, freq), name = serie.name)
            serie = serie.append(y_hat)
            
            # Add value to original scale prediction.
            predictions.append(correction + serie[-1])
          
            if self.stationary:
                correction = predictions[-1]

        y_hats = pd.Series(predictions, index = serie[-self.h:].index)

        if BOX:
            y_hats = invboxcox(y_hats, lmbda)
        
        return y_hats
