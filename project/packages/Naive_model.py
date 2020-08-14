import numpy as np

class Naive_model:
    def __init__(self, row_output = False):
        self.row_output = row_output
    
    def fit(self, X, y, target, features_name = None):
        self.output_dims = y.shape[1]
        if self.row_output:
            self.idx_target = list(features_name).index(target + "(t)")  
        else:
            self.idx_target = list(features_name).index(target) 
    
    def predict(self, X):
        if self.row_output:
            y_hats = X[:,self.idx_target]
        else:
            y_hats = X[:,:,self.idx_target][:,-1]
            y_hats = y_hats.reshape(y_hats.shape[0], 1)
            y_hats = np.repeat(y_hats, self.output_dims, axis = 1)

        return y_hats      