import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def cao_algorithm(serie, m_min = 2, m_max = 15, epsilon =  0.001, plot = False):
    
    """Cao algorithm.
    
    This code reproduces the Cao algorithm that allows to find the best embedding dimension following the rules dictated by the 
    paper [1] for a given time-series. More precisely, this code allows to obtain the graph of the variable E1 
    depending on a range of embedding dimensions. Moreover, an implementation allows to automatically identify 
    the best embedding dimension from the graph: the curve will be fitted using the function 1 / (1 + np.exp(-(x - a) * b)),
    and the best embedding dimension will be choosen when a not sensitive increment of the curve will be detected (plateau of
    the curve).
    
    Parameters
    ----------
    serie: the serie to which we want to apply the Cao algorithm.
    m_min: the lower limit range for the embedding dimension. The lower value allowed for 'm_min' is 2.
    m_max: the higher limit range for the embedding dimension.
    epsilon: a threshold for the measurement of the curve slope of the fit function using the standard deviation. 
    plot: if you want to be plot the E1 graph depending on embedding dimensions.
    
    Returns
    ----------
    best_m: the best embedding dimension find for the given time-series.
    
    Notes
    ----------
    This code reproduces the Cao algorithm with fixed delay factor tau to 1. Moreover for a completly flat time-series
    the code return as best embedding dimension the value of 2.
    It is necessary to select an appropriate m range in order to find the embedding dimension that allows to reach the 
    E1 plateau.
    
    References
    ----------
    
    [1]. "Practical method for determining the minimum embedding dimension of a scalar time series", Liangyue Cao.

    """
    
    # First of all, I check if the time-series is completly flat. In this case the embedding dimension returning is 2.
    if serie.diff().sum() == 0:
        best_m = 2
        return best_m

    # Define all the embedding dimension to check.
    # N.B if you want to check until for example embedding dimension 8 you also need the embedding dimension 9 for the E1 equation.
    ms = np.arange(m_min, m_max + 2)
    
    # Define the list where all the Em will be stored.
    E = list()
    for m in ms:
        def a_func(yi, serie, m):

            def distances(y, yi):
                if np.array_equal(y, yi):
                    return np.inf
                else:
                    dist = max(np.abs(y - yi))
                    return dist

            dist = serie.rolling(m).apply(distances, args = [yi], raw = True).dropna()
            min_distance = min(dist)

            return min_distance

        num = serie.rolling(m + 1).apply(a_func, args = [serie, m + 1], raw = True).dropna().reset_index(drop = True)
        den = serie.rolling(m).apply(a_func, args = [serie, m], raw = True).dropna().reset_index(drop = True)[:-1]
        
        a = num.divide(den)
        Em = a.mean()
        E.append(Em)
        
    # Compute E1 for the selected range of embedding dimensions.
    E1 = pd.Series(E).rolling(2).apply(lambda x: x[1]/x[0], raw = True).dropna()
    E1.index = ms[:-1]    
    
    # Create function that fit the E1 graph.
    def fit(serie):
        x = serie.index
        y = serie.values

        def l(x, alpha, beta):
            return 1 / (1 + np.exp(-(x - alpha) * beta))  

        param = curve_fit(l, x, y)
        alpha, beta = param[0]
        xnew = np.arange(x[0], x[-1] + 1, 0.1)
        ynew = 1 / (1 + np.exp(-(xnew - alpha) * beta)) 

        return pd.Series(ynew, index = xnew, name = serie.name)
    
    E1_fit = fit(E1)

    # Create function that automatically check best embedding dimension.
    def min_emb(serie):
        def func(pattern):
            if pattern.std() < epsilon:
                return np.ceil(pattern.first_valid_index())
            else:
                return np.nan

        return serie.rolling(10).apply(func, raw = False).dropna().values[0]
    
    best_m = int(min_emb(E1_fit))
    
    if plot:
        plt.figure(figsize = (10, 5))
        # Draw original graph.
        plt.plot(E1, c = "black", linewidth = 1, marker = "x")
        # Draw fitted graph.
        plt.plot(E1_fit, c = "blue", linewidth = 0.8)
        # Draw vertical line that identify best embedding dimension.
        plt.axvline(x = best_m, color = "k", linestyle = "--")
        plt.title(serie.name)
        plt.xlabel("Embedding dimension")
        plt.ylabel("E1")
        plt.show()
    
    return best_m