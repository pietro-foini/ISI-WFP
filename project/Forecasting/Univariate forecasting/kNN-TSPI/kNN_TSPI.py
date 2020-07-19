import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import stats
from rolling_window import rolling_window
import pandas as pd
import numpy as np

class kNN_TSPI:
    """kNN-Time Series Prediction with Invariances (kNN-TSPI).
    
    This code reproduces the kNN-Time Series Prediction with Invariances (kNN-TSPI) [1][2] model, a recent and promising 
    modification of the kNN algorithm for time-series prediction. This proposal differs from the literature by incorporating 
    techniques for amplitude and offset invariance, complexity invariance, and treatment of trivial matches. 
    According to the authors, these three invariances when combined allow more meaningful matching between the reference queries 
    and temporal data subsequences. The algorithm can predict large horizon through a recursive approach; it is not able to
    directly predict h-step ahead in the future.
    
    """
    def __init__(self, l, k, h, complexity_measure = "squared difference", plot_nn_subsequence = False):
        """
        ***Initialization function***
        
        Initialization of the kNN_TSPI class.
        
        Parameters
        ----------
        l: the size of the search window.
        k: the number of nearest neighbors.
        h: an int parameter indicating the prediction horizon.
        complexity_measure: the complexity measures that can be adopted in the algorithm. The allowed functions are the following:
                       - squared difference
                       - absolute difference   
        plot_nn_subsequence: whether or not you want to plot the k nearest neighbors sequences found at each iteration of the 
          prediction horizon h.

        Returns
        ----------
        y_hats: a pandas serie object containing the predicted points by the algorithm.
        
        References
        ----------
        [1]. "A Study of the Use of Complexity Measures in the Similarity Search Process Adopted by kNN Algorithm for Time Series 
              Prediction", Antonio Rafael Sabino Parmezan, Gustavo E. A. P. A. Batista.
        [2]. "Evaluation of statistical and machine learning models for time series prediction Identifying the state-of-the-art and 
              the best conditions for the use of each model", Javier Borge-Holthoefer", Antonio Rafael Sabino Parmezan, 
              Vinicius M.A. Souza, Gustavo E.A .P.A . Batista.

        """
        # Define some attributes of the class.
        self.l = l
        self.k = k
        self.h = h
        self.plot_nn_subsequence = plot_nn_subsequence
        
        # Define some complexity measures that can be adopted in the algorithm. 
        complexity_measures = {"squared difference": lambda x: np.sqrt(np.sum(np.diff(x)**2)),
                               "absolute difference": lambda x: np.sum(np.abs(np.diff(x)))}
        
        # Select the complexity measure.
        self.complexity_measure = complexity_measures[complexity_measure]
        
    def kNN_TSPI_predict(self, serie_sets, index_set_to_predict):
        """
        ***Sub-function***
        
        This function allows to predict a single step ahead in the future of one of time-series provided in the list.
        
        Parameters
        ----------
        serie_sets: a list of time-series to use for predicting one of them; they must be pandas serie objects with datetime 
           values as index. The frequency of the datatime indeces must be specified.
        index_set_to_predict: the index of the time-series stored in the list that you want to predict.
          
        """
        # Define the dictionary where all the subsequences of size l will be stored (with the corresponding distance score).
        self.S = dict()
        # Define the list where the k most similar subsequences series to Q will be stored (with the corresponding distance score).
        self.S_k = list() 
        # Define the query Q (the last l observations of the time-series).
        Q = serie_sets[index_set_to_predict][-self.l:]
        
        def compute_similarity(x, Q):
            # Get the correspondig S_l+1.
            p_1 = np.array(x)[-1][0]
            # Get the subsequence S_l.
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
            
            if change == 0:
                subsequence_p_1 = np.append(subsequence, p_1)
                subsequence_p_1_znorm = stats.zscore(subsequence_p_1)
                Sl1 = np.std(Q.values) * subsequence_p_1_znorm[-1] + np.mean(Q.values)
            else:
                Sl1 = p_1
                change = 0

            self.S[CID] = pd.Series(subsequence, index = subsequence_indeces), Sl1
            
            return CID

        # Combine the values and index datetime in parallel into an array not considering the query (-1 point to get S_l+1) itself during the search of the k nearest neighbors.
        for serie in serie_sets:
            serie_to_roll = np.rec.fromarrays([serie.values, serie.index])[:-(self.l - 1)]
            windows = rolling_window(serie_to_roll, self.l + 1, axes = 0)
            np.apply_along_axis(lambda x: compute_similarity(x, Q), 1, windows);

        self.S = sorted(self.S.items())
        
        # The first k nearest neighbors searching for the subsequences that have no trivial matches, i.e. there is no overlapping with the 
        # reference query Q (we prevented to this issue rolling from -l) or between each other.
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

        # Perform a weighted average based on CID of the k nearest neighbors.
        S_k_l1 = [x[1][1] for x in self.S_k]
        S_k_CID = [x[0] for x in self.S_k]
        # Prediction function f.
        fs = sum(np.array(S_k_l1)*(1/np.array(S_k_CID)))/sum((1/np.array(S_k_CID)))
        # Standard mean.
        #fs = (np.array(S_k_l1).sum())/self.k
            
        return fs  
        
    def predict(self, serie_sets, index_set_to_predict = 0):
        """
        ***Main function***
        
        This function allows to perform a recursive forecasting one of the time-series stored in the list provided up to h-steps ahead 
        in the future.
        
        Parameters
        ----------
        serie_sets: a list of time-series to use for predicting one of them; they must be pandas serie objects with datetime 
           values as index. The frequency of the datatime indeces must be specified.
        index_set_to_predict: the index of the time-series stored in the list that you want to predict.
          
        """
        # Define the frequency of the time-series.
        self.freq = serie_sets[0].index.freq
        
        # Make predictions.
        for i in range(self.h):
            y_hat = self.kNN_TSPI_predict(serie_sets, index_set_to_predict)
            y_hat = pd.Series(y_hat, index = [serie_sets[index_set_to_predict].index[-1] + 1*self.freq], name = serie_sets[index_set_to_predict].name)

            # Plot the k nearest neighbors sequences found at each iteration of the prediction horizon h.
            if self.plot_nn_subsequence:
                # Define the figure.
                fig, ax = plt.subplots(figsize = (20, 7))
                # Set the title.
                ax.set_title("Prediction horizon h: %d" % (i + 1), color = "black")
                for serie in serie_sets:
                    # Plot Z.
                    serie.plot(ax = fig.gca(), c = "gray", style = ":", label = "Z")                
                # Plot the k nearest neighbors sequences S_k on the figure.
                j = 1
                for x in self.S_k:
                    S = x[1][0]
                    S.plot(ax = fig.gca(), c = "blue", alpha = 0.4, style = "--", label = "S_%d" % j)
                    # Plot the S_l+1.
                    pd.concat(serie_sets)[[S.index[-1] + 1*self.freq]].plot(ax = fig.gca(), c = "red", style = "s", fillstyle = "none", label = "_")
                    j += 1                        
                # Plot Q.
                serie_sets[index_set_to_predict][-self.l:].plot(ax = fig.gca(), c = "green", style = "-", label = "Q")
                ax.autoscale()
                ax.legend()
                plt.show()
                plt.close()

            # Add the new prediction to the series in order to perform a recursive forecasting if needed.
            serie_sets[index_set_to_predict] = serie_sets[index_set_to_predict].append(y_hat)
        
        # Return only the predicted series.
        y_hats = serie_sets[index_set_to_predict][-self.h:]

        return y_hats  
    