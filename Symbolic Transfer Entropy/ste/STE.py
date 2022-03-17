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
    """Symbolic Transfer Entropy [1,2].
    
    This code allows to compute the symbolic transfer entropy between two time-series X and Y given an embedding dimension m. 
    The first step of the algorithm works in order to convert the two time-series into corresponding symbolized representation.
    
    Let's suppose that X = {x(1), x(2), ..., x(N)} and Y = {y(1), y(2), ..., y(N)} are the symbolized time-series 
    obtained from the original ones. Using these symbolized time-series, the standard formula of the transfer entropy to measure 
    the information flow from X to Y is the following:
    
    T_XY = ∑ p(y(i+1), y(i), x(i))·log2( p(y(i+1)|x(i),y(i)) / p(y(i+1)|y(i)) )
    
    There also exist other formulations, for example considering a more large history lengths. Defining two
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
    X = np.argsort(rolling_window(X, m).T) + 1
    X = np.array([dict_pattern_index[tuple(x)] for x in X])
    # Pattern time-series Y.
    Y = np.argsort(rolling_window(Y, m).T) + 1
    Y = np.array([dict_pattern_index[tuple(y)] for y in Y])

    Xk = rolling_window(np.expand_dims(X, axis = 1), kx).squeeze(axis = -1)
    Yk = rolling_window(np.expand_dims(Y, axis = 1), ky).squeeze(axis = -1)
    Yh = np.expand_dims(Y, axis = 1)

    # Define the concatenations of all these time-series lags.
    concatenation = np.concatenate([Xk[max(kx,ky)-kx:-h], Yk[max(kx,ky)-ky:-h], Yh[max(kx,ky)+h-1:]], axis = 1)
    # Define all the unique combinations/states of these two time-series.
    states = np.unique(concatenation, axis = 0)

    concatenation_y = np.concatenate([Yk[:-h], Yh[ky+h-1:]], axis = 1)

    T_XY = 0
    for state in states:
        prob1 = (concatenation == state).all(axis = 1).sum()/(len(concatenation))
        prob2 = (concatenation_y[:,:-1] == state[kx:-1]).all(axis = 1).sum()/(len(concatenation_y))
        prob3 = (concatenation[:,:-1] == state[:-1]).all(axis = 1).sum()/(len(concatenation))
        prob4 = (concatenation_y == state[kx:]).all(axis = 1).sum()/(len(concatenation_y))

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
    N.B. the number of bins must be changed based on the number of symbols/discrete numbers into the two time-series.
    
    Returns
    ----------
    T_XY: the transfer entropy T_XY to measure the information flow from X to Y.

    """
    Xk = rolling_window(np.expand_dims(X, axis = 1), kx).squeeze(axis = -1)
    Yk = rolling_window(np.expand_dims(Y, axis = 1), ky).squeeze(axis = -1)
    Yh = np.expand_dims(Y, axis = 1)

    # Define the concatenations of all these time-series lags.
    concatenation = np.concatenate([Xk[max(kx,ky)-kx:-h], Yk[max(kx,ky)-ky:-h], Yh[max(kx,ky)+h-1:]], axis = 1)
    # Define all the unique combinations/states of these two time-series.
    states = np.unique(concatenation, axis = 0)

    concatenation_y = np.concatenate([Yk[:-h], Yh[ky+h-1:]], axis = 1)

    T_XY = 0
    for state in states:
        prob1 = (concatenation == state).all(axis = 1).sum()/(len(concatenation))
        prob2 = (concatenation_y[:,:-1] == state[kx:-1]).all(axis = 1).sum()/(len(concatenation_y))
        prob3 = (concatenation[:,:-1] == state[:-1]).all(axis = 1).sum()/(len(concatenation))
        prob4 = (concatenation_y == state[kx:]).all(axis = 1).sum()/(len(concatenation_y))

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
    m: the embedding dimension; typical values 3<=m<=7.
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
    X = np.argsort(rolling_window(X, m).T) + 1
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

def entropy_rate_discrete(X, h = 1, k = 1):
    """
    This function computes the conditional Shannon entropy (entropy rate) for a  discrete time-series. More precisely, it returns 
    the following quantity:
    
            H(x(i+h)|x(i), ..., x(i-kx-1))
    
    Parameters
    ----------
    
    X: the time-series X.
    h: the h parameter for the future step; h >= 1.
    k: the k parameter for the lags steps of X time-series; k >= 1.
    
    Notes
    ----------
    This function is equal to the function 'entropy_rate' of the package pyinform.
    
    """
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

def to_symbolization(X, m = 3, pattern = False):
    """
    Convert scalar time-series into a symbolic representation using an embedding dimension m.
    
    """
    # Compute the symbolization of the two time-series rolling a window of dimension m.
    patterns = list(permutations(np.arange(m) + 1))
    # Get the list of integer numbers associated to the permutation.
    dict_pattern_index = {patterns[i]: i for i in range(len(patterns))}
    
    # Pattern time-series X.
    X = np.argsort(rolling_window(X, m).T) + 1
    if not pattern:
        X = np.array([dict_pattern_index[tuple(x)] for x in X])
    
    return X

def rolling_window(x, window):
    """
    This function allows to rolling a window over a numpy array.

    Parameters
    ----------
    x: the input array.
    window: the length of the window to slide.

    """
    x = np.array(x)
    # Set shape.
    shape = list(x.shape)
    shape[0] = x.shape[0] - window + 1
    shape.insert(len(shape)-1, window)
    # Set strides.
    strides = list(x.strides)
    strides.insert(0, strides[0])
    return np.lib.stride_tricks.as_strided(x, shape = tuple(shape), strides = tuple(strides))
