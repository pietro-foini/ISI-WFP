import matplotlib.pyplot as plt
from tool_datetime import next_datetimes
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from scipy.spatial import distance
import PermutationEntropy as PE
from scipy import stats
import rolling
import pandas as pd
import numpy as np
import itertools
import collections
import operator
import zlib
import sys

class kNN_TSPI:
    """kNN-Time Series Prediction with Invariances (kNN-TSPI).
    
    This code reproduces the kNN-Time Series Prediction with Invariances (kNN-TSPI) [1][2] model, a recent and promising 
    modification of the kNN algorithm for time-series prediction. This proposal differs from the literature by incorporating 
    techniques for amplitude and offset invariance, complexity invariance, and treatment of trivial matches. 
    According to the authors, these three invariances when combined allow more meaningful matching between the reference queries 
    and temporal data subsequences.
    
    Parameters
    ----------
    serie: the time-series to predict future values; it must be a pandas serie object with datetime values as index. 
      The frequency of the datatime index must be specified.
    l: the size of the search window.
    k: the number of nearest neighbors.
    h: an int parameter indicating the prediction horizon.
    complexity_measure: the complexity measures that can be adopted in the algorithm. The allowed functions are the following:
                   - squared difference
                   - absolute difference
                   - compression
                   - edges
                   - zero-crossing
                   - permutation entropy   
    alphabet_size: an int number that represents the number of letters of the alphabet you want to consider for the SAX 
      conversion of the time-series into symbols rapresentation. Max value is 19 and Min value is 2. This parameter is used
      only if the 'complexity_measure' function is set to 'compression'.
    embedding_size: the embedding dimension with which obtain symbolized series if the 'complexity_measure' function is set to
      'permutation entropy'. Typical values are set: 3 <= m <= 7.
    plot_nn_subsequence: whether or not you want to plot the k nearest neighbors sequences found at each iteration of the 
      prediction horizon h.

    Returns
    ----------
    y_hats: a pandas serie object containing the predicted points by the algorithm.
       
    """
    def __init__(self, l, k, h, complexity_measure = "squared difference", alphabet_size = 10, embedding_size = 3, 
                 plot_nn_subsequence = False):
        self.l = l
        self.k = k
        self.h = h
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.plot_nn_subsequence = plot_nn_subsequence
        
        # Define some complexity measures that can be adopted in the algorithm. 
        complexity_measures = {"squared difference": lambda x: np.sqrt(np.sum(np.diff(x)**2)),
                               "absolute difference": lambda x: np.sum(np.abs(np.diff(x))),                                    
                               "compression": lambda x: sys.getsizeof(zlib.compress(ts_to_string(x, cuts_for_asize(alphabet_size)).encode("utf-8"))),
                               "edges": lambda x: len(list(itertools.groupby(np.diff(x), lambda x: x > 0))),
                               "zero-crossing": lambda x: len(np.where(np.diff(np.sign(x)))[0]),
                               "permutation entropy": lambda x: PE.PE(pd.Series(x), embedding_size)}
        
        self.complexity_measure = complexity_measures[complexity_measure]
        
    def kNN_TSPI_predict(self, serie):
        # Define the dictionary where all the subsequences of size l will be stored (with the corresponding distance score).
        self.S = dict()
        # Define the list where the k most similar subsequences series to Q will be stored (with the corresponding distance score).
        self.S_k = list() 
        # Define the query Q.
        Q = serie[-self.l:]
        
        def compute_similarity(x, serie, Q):
            # Get the correspondig S_l+1.
            p_1 = np.array(x)[-1][0]
            subsequence = np.array([i[0] for i in np.array(x)[:-1]])
            subsequence_indeces = np.array([i[1] for i in np.array(x)[:-1]])

            CID = np.inf
            change = 0
            # z-normalization of the subsequence and the query Q.
            subsequence_znorm = stats.zscore(subsequence)
            Q_znorm = stats.zscore(Q)
            # Check if the query or the subsequence are flat: in this case the znorm causes some issues.
            if np.isnan(Q_znorm).any() or np.isnan(subsequence_znorm).any():
                Q_znorm = Q
                subsequence_znorm = subsequence
                change = 1
            # Compute the similarity between two subsequences.
            # Euclidean distance.
            ED = distance.euclidean(Q_znorm, subsequence_znorm)
            # Complexity estimate.
            CE = self.complexity_measure
            if min(CE(Q_znorm), CE(subsequence_znorm)) == 0:
                minimum = 1e-8        
            else:
                minimum = min(CE(Q_znorm), CE(subsequence_znorm))
            # Complex correction factor.
            CF = max(CE(Q_znorm), CE(subsequence_znorm))/minimum
            # Compute the CID.
            CID = ED * CF
            # Get the correspondig S_l+1.
            #p_1 = serie[next_datetimes(subsequence_indeces[-1], 1, self.freq)].values[0]
            
            if change == 0:
                subsequence_p_1 = np.append(subsequence, p_1)
                subsequence_p_1_znorm = stats.zscore(subsequence_p_1)

                Sl1 = np.std(Q.values) * subsequence_p_1_znorm[-1] + np.mean(Q.values)
            else:
                Sl1 = p_1
                change = 0

            self.S[CID] = pd.Series(subsequence, index = subsequence_indeces), Sl1
            
            return CID

        # Combine the values and index datetime in parallel into an array not considering the query (-1 point to get S_l+1) itself during the search of the k nearest neighbors..
        serie_to_roll = np.rec.fromarrays([serie.values, serie.index])[:-(self.l - 1)]
        _ = list(rolling.Apply(serie_to_roll, self.l + 1, operation = lambda x: compute_similarity(x, serie, Q)))

        self.S = sorted(self.S.items())

        # The first k nearest neighbors searching for the subsequences that have no trivial matches, i.e. there is no overlapping with the 
        # reference query Q (preventing rolling from -l) or between each other.
        self.S_k.append(self.S[0])
        if self.k != 1:
            for S in self.S[1:]:
                overlap = 0
                for i in range(len(self.S_k)):                
                    if len(S[1][0].index.intersection(self.S_k[i][1][0].index)) != 0:
                        overlap = 1
                if overlap == 0:    
                    self.S_k.append(S)
                if len(self.S_k) == self.k:
                    break

        S_k_l1 = [x[1][1] for x in self.S_k]
        # Prediction function f.
        fs = (np.array(S_k_l1).sum())/self.k
            
        return fs  
        
    def predict(self, serie):
        self.freq = serie.index.freq
        
        # Make predictions.
        for i in range(self.h):
            y_hat = self.kNN_TSPI_predict(serie)
            y_hat = pd.Series(y_hat, index = next_datetimes(serie.index[-1], 1, self.freq), name = serie.name)

            if self.plot_nn_subsequence:
                # Plot the k nearest neighbors sequences found at each iteration of the prediction horizon h.
                f = plt.figure(figsize = (20, 7))
                plt.title("Prediction horizon h: %d" % (i + 1), color = "black")
                # Plot Z.
                serie.plot(ax = f.gca(), c = "gray", style = ":", label = "Z")
                # Plot Q.
                serie[-self.l:].plot(ax = f.gca(), c = "green", style = "-", label = "Q")
                # Plot the k nearest neighbors sequences S_k.
                j = 1
                for x in self.S_k:
                    S = x[1][0]
                    S.plot(ax = f.gca(), c = "blue", alpha = 0.4, style = "--", label = "S_%d" % j)
                    # Plot the S_l+1.
                    serie[[S.index[-1] + 1*self.freq]].plot(ax = f.gca(), c = "red", style = "s", 
                                                                   fillstyle = "none", label = "_")
                    j += 1
                plt.autoscale()
                plt.legend()
                plt.show()
                plt.close()
            
            serie = serie.append(y_hat)
            
        y_hats = serie[-self.h:]

        return y_hats
    