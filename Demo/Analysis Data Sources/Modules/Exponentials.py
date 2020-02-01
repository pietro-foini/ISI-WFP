"""

This code reproduces the some Exponential Smoothing function explained in the paper [1].

[1]. Evaluation of statistical and machine learning models for time series prediction: Identifying the state-of-the-art and the best conditions for the use of each model


"""

from dateutil import relativedelta
import numpy as np
import pandas as pd
import datetime

def next_datetime_point(date):
    # This function simply returns the next timestamp given a certain date.
    next_date = datetime.date(date.year, date.month, date.day) + relativedelta.relativedelta(months = 1)
    next_date = next_date + relativedelta.relativedelta(day = 31)
    return pd.Timestamp(next_date)        

class SimpleExpSmoothing:
    def __init__(self, serie, alpha):
        self.serie = serie
        self.alpha = alpha
        self.L = None

        L = self.serie.values[0]
        
        for z in self.serie.values[1:-1]:
            L_recursive = self.alpha*z + (1 - self.alpha)*L
            L = L_recursive
        
        self.L = L    
        
    def predict(self, h):
        for i in range(h): 
            y_hat = self.alpha*self.serie[-1] + (1 - self.alpha)*self.L
            self.L = y_hat
            y_hat = pd.Series(y_hat, index = [next_datetime_point(self.serie.index[-1])], name = self.serie.name)
            self.serie = self.serie.append(y_hat)
            
        return self.serie[-h:]  
    
class HoltExpSmoothing:
    def __init__(self, serie, alpha, beta):
        self.serie = serie
        self.alpha = alpha
        self.beta = beta
        self.L = None
        self.T = None

        L = self.serie.values[0]
        T = self.serie.values[1] - self.serie.values[0]
        
        for z in self.serie.values[1:]:
            L_recursive = self.alpha*z + (1 - self.alpha)*(L + T)
            T_recursive = self.beta*(L_recursive - L) + (1 - self.beta)*T
            L = L_recursive
            T = T_recursive
        
        self.L = L  
        self.T = T
        
    def predict(self, h):
        for i in range(h): 
            y_hat = self.L + (i + 1)*self.T
            y_hat = pd.Series(y_hat, index = [next_datetime_point(self.serie.index[-1])], name = self.serie.name)
            self.serie = self.serie.append(y_hat)
            
        return self.serie[-h:] 
    