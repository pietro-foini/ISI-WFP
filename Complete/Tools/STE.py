from custom_correlation import corr_pairwise
from Cao import cao_algorithm
import PermutationEntropy
import pandas as pd
import numpy as np
    
def STE_T_XY(X, Y, m = 3, search_best_m = False, m_min = 2, m_max = 7, method = "Cao", best_ms = None):
    
    """Symbolic Transfer Entropy[1].
    
    This code allows to compute the symbolic transfer entropy between two time-series given an embedding dimension. 
    Furthermore, this implementation permit to compute the best embedding dimensions for both the time-series, exploiting
    or the Cao algorithm or the permutation entropy, and then, selecting as common embedding dimension the larger of the 
    two, compute the symbolic transfer entropy. It is also possible to give as input a serie object containing already
    the best embedding dimension of the two time-series.
    
    Parameters
    ----------
    X: the time-series X.
    Y: the time-series Y.
    m: the embedding dimension.
    search_best_m: a boolean parameter to determine if you to compute the best embedding dimensions for the two time-series.
    m_min: the lower limit range for the embedding dimension. The lower value allowed for 'm_min' is 2.
    m_max: the higher limit range for the embedding dimension.
    method: the method to use for the calculation of the best embedding dimension. 
    best_ms: a serie boject containing already the best embedding dimensions.
    
    Returns
    ----------
    probs: the transfer entropy.
    
    References
    ----------
    
    [1]. "The dynamics of information-driven coordination phenomena: A transfer entropy analysis", Javier Borge-Holthoefer, 
          Nicola Perra, Bruno Gonçalves, Sandra González-Bailón, Alex Arenas, Yamir Moreno, Alessandro Vespignani.

    """

    mx = m
    my = m
    
    # Define function that creates a sorted-index pattern starting from an original pattern of the time-series. E.g. (1, 3, 2) for pattern input (31.8, 38.9, 35.1).
    def sort_pattern_index(pattern):
        return tuple(np.argsort(pattern) + 1)
    
    # If you want to search the own best embedding dimension for time-series X and Y.
    if search_best_m:
        # Cao algorithm.
        if method == "Cao":
            mx = int(cao_algorithm(X, m_min = m_min, m_max = m_max))
            my = int(cao_algorithm(Y, m_min = m_min, m_max = m_max))
        # Permutation entropy algorithm.
        if method == "PE":
            ms = np.arange(m_min, m_max + 1)
            mx = PermutationEntropy.search_best_m(X, ms).idxmin()
            my = PermutationEntropy.search_best_m(Y, ms).idxmin()
    
    # Check if a best embedding dimension list has provided.
    if type(best_ms) != type(None):
        mx = int(best_ms.loc[X.name])
        my = int(best_ms.loc[Y.name])
        
    # In order to mantain consistency during the transfer entropy calculation, select the largest embedding dimension.
    m = max(mx, my)   
        
    # Compute the pattern symbolization of the time-series.
    X_permutations = list()
    X.rolling(m).apply(lambda x: X_permutations.append(sort_pattern_index(x)) or 0, raw = False)   
    Y_permutations = list()
    Y.rolling(m).apply(lambda x: Y_permutations.append(sort_pattern_index(x)) or 0, raw = False)   
        
    X = pd.Series(X_permutations, name = "X")
    Y = pd.Series(Y_permutations, name = "Y")
    Y_p = Y.shift(periods = -1).dropna().reset_index(drop = True)
    Y_p.name = "Y + 1"

    combinations = pd.concat([X, Y, Y_p], axis = 1).dropna().drop_duplicates()

    def probability(row):
        x = row["X"]
        y = row["Y"]
        y_p = row["Y + 1"]        
        # Shannon entropy.
        N = len(X)
        yi = np.array([1 if pattern == y else 0 for pattern in np.array(Y)])
        yi_p = np.roll(np.array([1 if pattern == y_p else 0 for pattern in np.array(Y)]), -1)
        xi = np.array([1 if pattern == x else 0 for pattern in np.array(X)])
        prob1 = (yi + yi_p + xi == 3).astype(int)[0:N-1].sum()/(N-1)    
        prob2 = (yi == 1).astype(int)[0:N].sum()/(N)    
        prob3 = (yi + xi == 2).astype(int)[0:N].sum()/(N)
        prob4 = (yi + yi_p == 2).astype(int)[0:N-1].sum()/(N-1)

        prob = prob1 * np.log2((prob1 * prob2)/(prob3 * prob4))
        return prob
        
    probs = combinations.apply(probability, axis = 1)
    probs = probs.sum()

    return probs

def compute_T(df, m = 3, search_best_m = False, m_min = 2, m_max = 7, method = "Cao", plot_cao = False, epsilon =  0.001):
    
    if search_best_m:
        if method == "PE":
            # Create range to search best embedding dimension.
            ms = np.arange(m_min, m_max + 1)
            # Search best embedding dimension.
            best_ms = df.apply(lambda serie: PermutationEntropy.search_best_m(serie, ms)).idxmin()
        if method == "Cao":
            best_ms = df.apply(lambda serie: cao_algorithm(serie, m_min = m_min, m_max = m_max, plot = plot_cao, epsilon = epsilon))
    else:
        best_ms = None
    
    # Compute the correlation matrix using symbolic transfer entropy.
    T = corr_pairwise(df, lambda x, y: STE_T_XY(x, y, m = m, search_best_m = False, best_ms = best_ms))
    
    return T, best_ms