from transformations import standardize, inv_standardize
from tool_datetime import next_datetimes
from sklearn.svm import SVR
import pandas as pd
import numpy as np

class SVM:
    """Support Vector Machine (SVM).
    
    SVM models have a similar structure to ANN, but differ in how the learning is conducted. While ANN work by minimizing 
    the empirical risk, i.e., the error minimization of the induced model on the training data, SVM are grounded on the 
    principle of structural risk minimization, which seeks the lowest training error while minimizing an upper 
    bound on the generalization error of the model (model error when applied to test data).
    
    Parameters
    ----------
    serie: the time-series to predict future values; it must be a pandas serie object with datetime values as index. 
      The frequency of the datatime index must be specified.
    l: the size of the search window.
    C: the regularization parameter of the model.
    gamma: the Gaussian’s width of the radial basis kernel function.
    h: an int parameter indicating the prediction horizon.
    standardize: a bool parameter, whether or not you want to apply a standardization transformation to the time-series.

    Returns
    ----------
    y_hats: a pandas serie object containing the predicted points by the algorithm.
       
    """
    def __init__(self, l, C, gamma, h, standardization = False):
        self.l = l
        self.C = C
        self.gamma = gamma
        self.h = h
        self.standardize = standardization
        
    # Define the function that split a sequence (time-series) into X and y values (into supervised problem).
    def split_sequence(self, sequence, l):
        X, y = list(), list()
        for i in range(len(sequence)):
            # Find the end of this pattern.
            end_ix = i + l
            # Check if we are beyond the sequence.
            if end_ix > len(sequence) - 1:
                break
            # Gather input and output parts of the pattern.
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
        
    def predict(self, serie):
        freq = serie.index.freq
        
        # Apply box cox transformation.
        if self.standardize:
            serie, self.mean, self.std = standardize(serie)
        
        # Make predictions.
        X, y = self.split_sequence(serie, self.l)
        regressor = SVR(C = self.C, gamma = self.gamma, kernel = "rbf")
        regressor.fit(X, y)
        for i in range(self.h):
            y_hat = regressor.predict(serie.tail(self.l).values.reshape(1, -1))[0]            
            y_hat = pd.Series(y_hat, index = next_datetimes(serie.index[-1], 1, freq), name = serie.name)
            serie = serie.append(y_hat)
            
        y_hats = serie[-self.h:]
        
        if self.standardize:
            y_hats = inv_standardize(y_hats, self.mean, self.std)

        return y_hats
    