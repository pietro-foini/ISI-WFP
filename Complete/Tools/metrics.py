from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np

class metrics:   
    """Basic metrics.
    
    In this module are defined some basic metrics to exploit with the purpose of computing the forecast error. For more details 
    regarding the basic metrics implemented in this module see the papers [1][2].
    
    Parameters
    ----------
    y_true: the groundtruth time-series.
    y_pred: the forecast time-series.
    metric: the basic metrics to choose for the evaluation of the forecast error. The allowed metrics are:
                    - MSE
                    - TU
                    - ER
                    - MAPE
                    - MCPM

    Returns
    ----------
    The error of the chosen basic metric.
    
    References
    ----------
    [1]. "Evaluation of statistical and machine learning models for time series prediction: Identifying the state-of-the-art 
          and the best conditions for the use of each model", Antonio Rafael Sabino Parmezan, Vinicius M.A. Souza, 
          Gustavo E.A .P.A . Batista.
    [2]. "A Study of the Use of Complexity Measures in the Similarity Search Process Adopted by kNN Algorithm for 
          Time Series Prediction", Antonio Rafael Sabino Parmezan, Gustavo E. A. P. A. Batista.
       
    """
    
    def __init__(self, y_true, y_pred, metric = "MSE"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.metric = metric      
        
    def MSE(self):
        return mse(y_true = self.y_true, y_pred = self.y_pred)
        
    def TU(self):
        return np.sum((self.y_true - self.y_pred)**2)/np.sum(np.diff(self.y_true)**2)
    
    def ER(self):
        D = 0
        a = np.diff(self.y_pred)
        b = np.diff(self.y_true)
        for i in range(len(a)):
            if a[i]*b[i] > 0:
                D += 1          
        POCID = (D/len(self.y_true)) * 100

        return 100 - POCID
    
    def MAPE(self):
        return np.sum(np.abs((self.y_true - self.y_pred)/self.y_true)*100)/len(self.y_true)
    
    def MCPM(self):
        er = self.ER()
        mse = self.MSE()
        tu = self.TU()
        mape = self.MAPE()
        
        mcpm = er + mse + tu + mape

        return mcpm
    
    def compute(self):    
        if self.metric == "MSE":
            return self.MSE()
        if self.metric == "TU":
            return self.TU()
        if self.metric == "ER":
            return self.ER()
        if self.metric == "MAPE":
            return self.MAPE()
        if self.metric == "MCPM":
            return self.MCPM()