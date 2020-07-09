from custom_correlation import corr_pairwise
from rolling_window import rolling_window
from itertools import permutations
import pandas as pd
import numpy as np

# Python module.
#
#
# Pietro Foini
#
# Year: 2020

def calc_ste(X, Y, m = 3, h = 1, kx = 1, ky = 1):
    """Symbolic Transfer Entropy[1][2].
    
    This code allows to compute the symbolic transfer entropy between two time-series X and Y given an embedding dimension m. 
    The first step of the algorithm works in order to convert the two time-series into a symbolized representation.
    Let's suppose that X = {x(1), x(2), ..., x(N)} and Y = {y(1), y(2), ..., y(N)} are already the symbolized time-series 
    obtained from the original ones. Using these symbolized time-series, the standard formula of the transfer entropy to measure 
    the information flow from X to Y is the following:
    
    T_XY = ∑ p(y(i+1), y(i), x(i))·log2( p(y(i+1)|x(i),y(i)) / p(y(i+1)|y(i)) )
    
    There also exist other formulations, for example considering a more large histroy lengths. Defining two
    parameters kx and ky, it is possible to select not only the last temporal lags as reference for the two time-series:
    
    T_XY = ∑ p(y(i+1), y(i), ..., y(i-ky-1), x(i), ..., x(i-kx-1))·log2(p(y(i+1)|y(i), ..., y(i-ky-1), x(i), ..., x(i-kx-1)) / p(y(i+1)|y(i), ..., y(i-ky-1)))   

    where kx and ky are respectively the lengths of the histories for the two time-series. In the first example kx=ky=1.
    
    Finally, a further implementation allows to consider a more large future horizon h; for example considering kx=ky=1:
    
    T_XY = ∑ p(y(i+h), y(i), x(i))·log2(p(y(i+h)|x(i),y(i)) / p(y(i+h)|y(i)))
    
    This implies that we are measuring the capacity of a signal to predict not the immediate future of another signal, but the h-th 
    future ahead symbol.
    
    Parameters
    ----------
    X: the time-series X.
    Y: the time-series Y.
    m: the embedding dimension; typical values 3<=m<=7.
    h: the h parameter for the future step; h >= 1.
    kx: the kx parameter for the history length of X time-series; kx >= 1.
    ky: the ky parameter for the history length of Y time-series; ky >= 1.
    
    Returns
    ----------
    T_XY: the symbolic transfer entropy T_XY to measure the information flow from X to Y.
    
    References
    ----------
    
    [1]. "Symbolic transfer entropy", M. Staniek, K. Lehnertz.
    [2]. "The dynamics of information-driven coordination phenomena: A transfer entropy analysis", Javier Borge-Holthoefer, 
          Nicola Perra, Bruno Gonçalves, Sandra González-Bailón, Alex Arenas, Yamir Moreno, Alessandro Vespignani.

    """
    # Compute the symbolization of the two time-series rolling a window of dimension m.
    patterns = list(permutations(np.arange(m) + 1))
    # Get the list of integer numbers associated to the permutation.
    dict_pattern_index = {patterns[i]: i for i in range(len(patterns))}
    
    # Pattern time-series X.
    X = np.argsort(rolling_window(X, m, axes = 0)) + 1
    X = np.array([dict_pattern_index[tuple(x)] for x in X])
    # Pattern time-series Y.
    Y = np.argsort(rolling_window(Y, m, axes = 0)) + 1
    Y = np.array([dict_pattern_index[tuple(y)] for y in Y])

    # Define the length of the two time-series.
    N = X.shape[0]

    # Let's define the future horizon to take into consideration.
    Y_h = np.expand_dims(np.roll(Y, shift = -h, axis = 0), axis = 1)

    # Create histories length of the two time-series.
    Xk = np.stack([np.roll(X, shift = i, axis = 0) for i in range(kx)], axis = 1)
    Yk = np.stack([np.roll(Y, shift = i, axis = 0) for i in range(ky)], axis = 1)

    # Define all the unique combinations/states of these two time-series.
    states = np.unique(np.concatenate([Xk, Yk, Y_h], axis = 1)[max(kx,ky)-1:-h], axis = 0)
    
    # Define the concatenations of all these time-series lags.
    concatenation = np.concatenate([Xk, Yk, Y_h], axis = 1)

    T_XY = 0
    for state in states:
        prob1 = (concatenation == state).all(axis = 1).astype(int)[max(kx,ky)-1:N-h].sum()/(N-(h+max(kx,ky)-1))
        prob2 = (concatenation[:,kx:kx+ky] == state[kx:kx+ky]).all(axis = 1).astype(int)[ky-1:N-h].sum()/(N-(h+ky-1))
        prob3 = (concatenation[:,:-1] == state[:-1]).all(axis = 1).astype(int)[max(kx,ky)-1:N-h].sum()/(N-(h+max(kx,ky)-1))
        prob4 = (concatenation[:,kx:] == state[kx:]).all(axis = 1).astype(int)[ky-1:N-h].sum()/(N-(h+ky-1))

        prob = prob1 * np.log2((prob1 * prob2)/(prob3 * prob4))

        if prob != np.nan:
            T_XY += prob 

    return T_XY

def calc_te(X, Y, h = 1, kx = 1, ky = 1):
    """Transfer Entropy (discrete variables).
    
    This code allows to compute the transfer entropy between two discretized time-series X and Y. 
    The algorithm works similarly to the symbolic transfer entropy after the conversion into a symbolic representation.
    
    Parameters
    ----------
    X: the discretized time-series X.
    Y: the discretized time-series Y.
    h: the h parameter for the future step; h >= 1.
    kx: the kx parameter for the history length of X time-series; kx >= 1.
    ky: the ky parameter for the history length of Y time-series; ky >= 1.
    
    Notes
    ----------
    
    To verify the efficiency of this algorithm you can also use two other packages starting from two discerete array time-series X and Y.

    Python: 
    >>> from pyinform import transfer_entropy
    >>> transfer_entropy(X, Y, k = 2) # N.B. k is only the history length of the target Y, the length of X is fix to 1.

    R:
    >>> library(RTransferEntropy)
    >>> calc_te(X, Y, lx = 1, ly = 1, q = 1, entropy ="Shannon", shuffles = 100, type = "bins", bins = 6)
    N.B. the number of bins must be change based on the number of symbols/discrete numbers into the two time-series.
    
    Returns
    ----------
    T_XY: the transfer entropy T_XY to measure the information flow from X to Y.

    """
    # Define the length of the two time-series.
    N = X.shape[0]

    # Let's define the future horizon to take into consideration.
    Y_h = np.expand_dims(np.roll(Y, shift = -h, axis = 0), axis = 1)

    # Create histories length of the two time-series.
    Xk = np.stack([np.roll(X, shift = i, axis = 0) for i in range(kx)], axis = 1)
    Yk = np.stack([np.roll(Y, shift = i, axis = 0) for i in range(ky)], axis = 1)

    # Define all the unique combinations/states of these two time-series.
    states = np.unique(np.concatenate([Xk, Yk, Y_h], axis = 1)[max(kx,ky)-1:-h], axis = 0)
    
    # Define the concatenations of all these time-series lags.
    concatenation = np.concatenate([Xk, Yk, Y_h], axis = 1)

    T_XY = 0
    for state in states:
        prob1 = (concatenation == state).all(axis = 1).astype(int)[max(kx,ky)-1:N-h].sum()/(N-(h+max(kx,ky)-1))
        prob2 = (concatenation[:,kx:kx+ky] == state[kx:kx+ky]).all(axis = 1).astype(int)[ky-1:N-h].sum()/(N-(h+ky-1))
        prob3 = (concatenation[:,:-1] == state[:-1]).all(axis = 1).astype(int)[max(kx,ky)-1:N-h].sum()/(N-(h+max(kx,ky)-1))
        prob4 = (concatenation[:,kx:] == state[kx:]).all(axis = 1).astype(int)[ky-1:N-h].sum()/(N-(h+ky-1))

        prob = prob1 * np.log2((prob1 * prob2)/(prob3 * prob4))

        if prob != np.nan:
            T_XY += prob 

    return T_XY

def entropy_rate(X, m = 3, h = 1, k = 1):
    """
    This function computes the conditional Shannon entropy (entropy rate) for a time-series, first converted into a symbolized
    time-series. More precisely, it returns the following quantity:
    
            H(x(i+h)|x(i), ..., x(i-kx-1))
    
    Parameters
    ----------
    
    X: the time-series X.
    m: the embedding dimension; typical values 3<=m<=7
    h: the h parameter for the future step; h >= 1.
    k: the k parameter for the lags steps of X time-series; k >= 1.
    
    Notes
    ----------
    This function is equal to the function 'entropy_rate' of the package pyinform for discrete variables.
    
    """
    # Compute the symbolization of the two time-series rolling a window of dimension m.
    patterns = list(permutations(np.arange(m) + 1))
    # Get the list of integer numbers associated to the permutation.
    dict_pattern_index = {patterns[i]: i for i in range(len(patterns))}
    
    # Pattern time-series X.
    X = np.argsort(rolling_window(X, m, axes = 0)) + 1
    X = np.array([dict_pattern_index[tuple(x)] for x in X])
    
    # Define the length of the two time-series.
    N = X.shape[0]

    # Let's define the future horizon to take into consideration.
    X_h = np.expand_dims(np.roll(X, shift = -h, axis = 0), axis = 1)

    # Create histories length of the two time-series.
    Xk = np.stack([np.roll(X, shift = i, axis = 0) for i in range(k)], axis = 1)

    # Define all the unique combinations/states of these two time-series.
    states = np.unique(np.concatenate([Xk, X_h], axis = 1)[k-1:-h], axis = 0)
    
    # Define the concatenations of all these time-series lags.
    concatenation = np.concatenate([Xk, X_h], axis = 1)

    H = 0
    for state in states:
        prob1 = (concatenation == state).all(axis = 1).astype(int)[k-1:N-h].sum()/(N-(h+k-1))
        prob2 = (concatenation[:,:-1] == state[:-1]).all(axis = 1).astype(int)[k-1:N-h].sum()/(N-(h+k-1))

        prob = (prob1) * np.log2(prob1/prob2)

        if prob != np.nan:
            H += prob 

    return -H

def calc_entropy_rates_for_te(X, Y, m = 3, h = 1, kx = 1, ky = 1):
    """
    In order to compute the transfer entropy from X to Y, it is possible to use the conditional Shannon entropies. Let's
    suppose for example to compute T_XY:
    
    T_XY = H(y(i+h)|y(i), ..., y(i-ky-1)) - H(y(i+h)|y(i), ..., y(i-ky-1), x(i), ..., x(i-kx-1)) = H1 - H2
    
    This function will be return the two conditional entropies H1 and H2. 
    
    N.B. The factor H1 is also used as a normalization factor for the transfer entropy T_XY.
    
    """
    
    # Compute the symbolization of the two time-series rolling a window of dimension m.
    patterns = list(permutations(np.arange(m) + 1))
    # Get the list of integer numbers associated to the permutation.
    dict_pattern_index = {patterns[i]: i for i in range(len(patterns))}
    
    # Pattern time-series X.
    X = np.argsort(rolling_window(X, m, axes = 0)) + 1
    X = np.array([dict_pattern_index[tuple(x)] for x in X])
    # Pattern time-series Y.
    Y = np.argsort(rolling_window(Y, m, axes = 0)) + 1
    Y = np.array([dict_pattern_index[tuple(y)] for y in Y])
    
    # Define the length of the two time-series.
    N = X.shape[0]

    # Let's define the future horizon to take into consideration.
    Y_h = np.expand_dims(np.roll(Y, shift = -h, axis = 0), axis = 1)

    # Create histories length of the two time-series.
    Xk = np.stack([np.roll(X, shift = i, axis = 0) for i in range(kx)], axis = 1)
    Yk = np.stack([np.roll(Y, shift = i, axis = 0) for i in range(ky)], axis = 1)

    # Define all the unique combinations/states of these two time-series.
    states = np.unique(np.concatenate([Xk, Yk, Y_h], axis = 1)[max(kx,ky)-1:-h], axis = 0)
    
    # Define the concatenations of all these time-series lags.
    concatenation = np.concatenate([Xk, Yk, Y_h], axis = 1)

    H1 = 0
    H2 = 0
    for state in states:
        # Compute H1 conditional entropy.
        prob1 = (concatenation == state).all(axis = 1).astype(int)[max(kx,ky)-1:N-h].sum()/(N-(h+max(kx,ky)-1))
        prob2 = (concatenation[:,kx:] == state[kx:]).all(axis = 1).astype(int)[ky-1:N-h].sum()/(N-(h+ky-1))
        prob3 = (concatenation[:,kx:-1] == state[kx:-1]).all(axis = 1).astype(int)[ky-1:N-h].sum()/(N-(h+ky-1))

        prob = (prob1) * np.log2(prob2/prob3)

        if prob != np.nan:
            H1 += prob 
            
        # Compute H2 conditional entropy.
        prob1 = (concatenation == state).all(axis = 1).astype(int)[max(kx,ky)-1:N-h].sum()/(N-(h+max(kx,ky)-1))
        prob2 = (concatenation[:,:-1] == state[:-1]).all(axis = 1).astype(int)[max(kx,ky)-1:N-h].sum()/(N-(h+max(kx,ky)-1))

        prob = (prob1) * np.log2(prob1/prob2)

        if prob != np.nan:
            H2 += prob 

    H1 = -H1
    H2 = -H2
            
    return H1, H2

def compute_T(df, m = 3, h = 1, kx = 1, ky = 1):   
    """
    Compute the correlation matrix using symbolic transfer entropy as metric.
    
    """
    
    T = corr_pairwise(df, lambda x, y: calc_ste(x.values, y.values, m = m, h = h, kx = kx, ky = ky))
    
    return T
    