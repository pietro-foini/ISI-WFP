import pandas as pd
import numpy as np
import math

def PE(serie, m, normalization = "obs"):
    """Permutation Entropy (PE).
    
    Permutation entropy is conceptually similar to the well-known Shannon entropy. However, instead of being based on the 
    probability of observing a system in a particular state, it utilizes the frequency of discrete motifs, i.e symbols, 
    associated with the growth, decay, and stasis of a time-series [1][2]. The conversion of the time-series to a symbolic 
    rappresentation is very similar to the concept of Symbolic Transfer Entropy [3].
    
    Parameters
    ----------
    serie: the serie to which we want to compute the permutation entropy.
    m: the embedding dimension with which obtain symbolized series. Typical values are set: 3 <= m <= 7. 
    normalization: allowed values: ["obs", "m!"]. If you want to normalize the permutation entropy based on observed symbols, 'obs', or
      on the factorial of the embedding dimension, 'm!'.
    
    Returns
    ----------
    Hp: the permutation entropy of the chosen serie.
    
    Notes
    ----------
    This code reproduces the Permutation Entropy with fixed delay factor tau to 1.
    
    References
    ----------
    
    [1]. "On the predictability of infectious disease outbreaks", Samuel V. Scarpino & Giovanni Petri.
    [2]. "A Study of the Use of Complexity Measures in the Similarity Search Process Adopted by kNN Algorithm for Time Series Prediction",
          Antonio Rafael Sabino Parmezan, Gustavo E. A. P. A. Batista.
    [3]. "The dynamics of information-driven coordination phenomena: A transfer entropy analysis", Javier Borge-Holthoefer, 
          Nicola Perra, Bruno Gonçalves, Sandra González-Bailón, Alex Arenas, Yamir Moreno, Alessandro Vespignani.

    """
    if m < 2:
        raise ValueError("The chosen 'm' value is not valid. It must be > 1.")       
        
    def sort_pattern_index(pattern):
        return tuple(np.argsort(pattern) + 1)

    # Compute the pattern symbolization of the time-series.
    list_of_sorted_patterns_index = list()
    serie.rolling(m).apply(lambda x: list_of_sorted_patterns_index.append(sort_pattern_index(x)) or 0, 
                           raw = False)        

    list_of_sorted_patterns_index = [i for i in list_of_sorted_patterns_index if i]    

    # We implement the Brandmaier correction that excludes all unobserved symbols when calculating Hp, which acts as a 
    # penalty against higher dimensions and results in a minimum value of Hp for finite length time-series.
    # To control for differences in dimension and for the effect of time-series length on the entropy estimation, 
    # we normalize the entropy by the log number of observed symbols.
    sorted_patterns_index = pd.Series(list_of_sorted_patterns_index)
    # Compute the permutation entropy of the time-series that is given by the Shannon entropy on the permutation orders.
    Hp = 0
    value_counts = sorted_patterns_index.value_counts()
    p_pi = value_counts.divide(len(sorted_patterns_index))
    for p in p_pi:
        Hp += p * math.log(p)
    Hp = -Hp

    if len(sorted_patterns_index.unique()) == 1:
        Hp = 0
    else:
        if normalization == "obs":
            Hp = Hp/math.log(len(sorted_patterns_index.unique())) 
        elif normalization == "m!":
            Hp = Hp/math.log(math.factorial(m)) 

    return Hp   

def search_best_m(serie, ms):
    """
    Searching for the embedding dimension that gives the lower permutation entropy. This
    function, given a time-serie and a range of embedding dimensions, returns a list of entropies associated to the 
    chosen embedding dimensions.
    
    Parameters
    ----------
    serie: the serie to which we want to use for the purpose.
    ms: a range of embedding dimensions to use for finding the best embedding dimension.
    
    """
    entropy = list() 
    for m in ms:
        entropy.append(PE(serie, m, normalization = "obs"))
    return pd.Series(entropy, index = ms, name = serie.name)

def PE_scaling_with_amount_of_data(serie, n_iter, ms, min_window_length = 10, max_window_length = 100):
    """
    Focusing on the predictability over short timescales for a time-serie, we compute the permutation entropy 
    over temporal windows of width up to 'max_window_length' length by selecting 'n_iter' random points and calculating 
    the permutation entropy for windows of length ('min_window_length', ..., 'max_window_length') inside the 
    given time-series.
    
    Parameters
    ----------
    serie: the serie to which we want to use for the purpose.
    n_iter: the number of random points to generate.
    ms: a range of embedding dimensions to use for finding the best embedding dimension for each sub time-series.
    min_window_length: the smallest length of the window to sample the original time-series.
    max_window_length: the greatest length of the window to sample the original time-series.
    
    """
    serie = serie.reset_index(drop = True)
    name = serie.name
    
    summaries = pd.Series([])
    for iteration in range(n_iter):        
        # Define the random starting point in the time-series and its length to generate a subsample time-series.
        start = np.random.randint(len(serie) - min_window_length)
        length = np.random.randint(min_window_length, max_window_length + 1)

        # Check if take the sub time-series behind or above the starting point.
        if len(range(start + 1, len(serie))) < length:
            sub_serie = serie.loc[start:len(serie)].reset_index(drop = True)
        else:
            sub_serie = serie.loc[start:(start + length - 1)].reset_index(drop = True)

        # Length of the sub time-series selected.
        n = len(sub_serie)
        
        # Search for the best m parameter for this sub time-series.
        best_m = search_best_m(sub_serie, ms).idxmin()
        # Compute permutation entropy.
        entropy = PE(sub_serie, best_m, normalization = "m!")
        
        summary = pd.Series([entropy, n, best_m], index = ["PE", "n", "m"], name = name, dtype = "float64")
        
        summaries = pd.concat([summaries, summary], axis = 0)
        
    return summaries
