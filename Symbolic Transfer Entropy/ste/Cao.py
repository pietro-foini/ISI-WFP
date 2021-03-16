import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Python module.
#
#
# Pietro Foini
#
# Year: 2020

def cao_algorithm(serie, m_min = 2, m_max = 15, plot = False):
    """Cao algorithm[1].
    
    This code reproduces the Cao algorithm that allows to find the best embedding dimension of a scalar time-series following the 
    rules dictated by Cao. More precisely, this code allows to obtain E1 depending on a range of embedding dimensions. 
    
    Parameters
    ----------
    serie: the serie to which we want to apply the Cao algorithm.
    m_min: the lower limit range for the embedding dimension. The lower value allowed for 'm_min' is 2.
    m_max: the higher limit range for the embedding dimension.
    plot: if you want to be plot E1 depending on embedding dimensions.
    
    Returns
    ----------
    best_m: the best embedding dimension find for the given time-series.
    
    Notes
    ----------
    In this module, the term 'm' substitutes the term 'd' in the reference paper.
    
    References
    ----------
    [1]. "Practical method for determining the minimum embedding dimension of a scalar time series", Liangyue Cao.

    """
    N = len(serie)

    # Define all the embedding dimension to check.
    # N.B if you want to check until for example embedding dimension 8 you also need the embedding dimension 9 for the E1 equation.
    ms = np.arange(m_min, m_max + 2)

    ###################
    ## CAO ALGORITHM ##
    ###################
    
    # Define the list where all the Em will be stored.
    E = list()
    for m in ms:
        # Compute the embedding vector of dimension m.
        num = rolling_window(np.expand_dims(serie.values, axis = 1), m+1).squeeze(axis = -1)
        # Compute the embedding vector of dimension m+1.
        den = rolling_window(np.expand_dims(serie.values, axis = 1), m).squeeze(axis = -1)[:-1]

        # For each embedding vector i compute the distances with other embedding vectors searching for the nearest neighbor.
        a_num_i, a_den_i = list(), list()
        for i in range(N-m):
            # Compute distance. Note: If the embedding vectors are equals, we take the second nearest neighbor instead of it.
            distances_den = np.ma.masked_equal(np.linalg.norm(den[i] - den, ord = np.inf, axis = 1), 0.0, copy = False)
            # Get n(i,d) --> nearest neighbor.
            argmin_den = distances_den.argmin()
            # Get corresponding distance with nearest neighbor.
            distance_den = distances_den[argmin_den]

            # The 𝑛(𝑖,𝑚) in the numerator is the same as that in the denominator.
            distance_num = np.linalg.norm(num[i] - num[argmin_den], ord = np.inf)

            a_num_i.append(distance_num)
            a_den_i.append(distance_den)

        a_num_i = np.array(a_num_i)
        a_den_i = np.array(a_den_i)

        a_i = a_num_i/a_den_i

        Em = np.mean(a_i)
        E.append(Em)

    # Compute E1 for the selected range of embedding dimensions.
    E1 = rolling_window(np.expand_dims(np.array(E), axis = 1), 2).squeeze(axis = -1)
    E1 = E1[:,1] / E1[:,0]

    E1 = pd.Series(E1, index = ms[:-1])

    if plot:
        fig, ax = plt.subplots(figsize = (10, 5))
        # Draw original graph.
        ax.plot(E1, c = "black", linewidth = 1, marker = "x")
        ax.set_title(serie.name)
        ax.set_xlabel("Embedding dimension")
        ax.set_ylabel("E1")
        ax.set_ylim([0,1])
        plt.show()
        
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