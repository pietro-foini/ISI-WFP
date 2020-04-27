import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from scipy.stats import boxcox
import warnings
warnings.filterwarnings("ignore")

def stationarity(serie):
    """
    The differencing transformation to render the time-series stationary if needed using adfuller test & KPSS test.

    """
    original_serie = serie.copy()
    SignificanceLevel = 0.05
    # Dickey-Fuller test.
    adfTest = adfuller(serie, autolag = "AIC")
    pValueadf = adfTest[1] 
    # KPSS test.
    KPSSTest = kpss(serie)    
    pValueKPSS = KPSSTest[1]

    # Not make the difference only when both test say that the serie is stationary. 
    if pValueadf < SignificanceLevel and pValueKPSS > SignificanceLevel:
        correction = 0
    else:
        serie = serie.diff().dropna()       
        correction = original_serie[-1]
    
    return serie, correction

def box_cox(serie):
    """
    A Box-Cox transformation is a way to stabilize the variance of a time series with non-negative values 
    getting a serie more linear and a distribution more Gaussian or Uniform. Sometimes after applying Box-Cox with 
    a particular value of lambda the process may look stationary. It is sometimes possible that even if after applying 
    the Box-Cox transformation the series does not appear to be stationary.
    
    """
    y_boxcox, lmbda = boxcox(serie)
    y_boxcox = pd.Series(y_boxcox, index = serie.index)
    return y_boxcox , lmbda


def invboxcox(serie, lmbda):
    if lmbda == 0:
        return (np.exp(serie))
    else:
        return (np.exp(np.log(lmbda*serie + 1)/lmbda))
    
def standardize(serie):
    mean = serie.mean()
    std = serie.std()
    serie = (serie - mean)/std
    return serie, mean, std

def inv_standardize(serie, mean, std):
    return serie*std + mean







    
    
    
    
    
    