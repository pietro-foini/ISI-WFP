
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import os
import shutil


# In[2]:


# Add the python path to the folder containing some useful custom packages.
import sys
sys.path.insert(0, "../../packages/")
from TsIP.TsIP import TsIP
from tools import find_multiple_sets
from LagsCreator.LagsCreator import LagsCreator


# In[3]:


# Create workspace.
dir = "./output"
if not os.path.exists(dir):
    os.makedirs(dir)
else:
    shutil.rmtree(dir)           
    os.makedirs(dir)


# ## Dataset

# In[4]:


COUNTRY = "Yemen"


# In[5]:


PATH_TO_DATA_FOLDER = "../../Dataset time-series/data/" + COUNTRY + "/"


# In[6]:


# Load the dataset of the smoothed training sets.
train_smooth = pd.read_csv(PATH_TO_DATA_FOLDER + "train_smooth.csv", header = [0, 1], index_col = 0)
train_smooth.index.name = "Datetime"
train_smooth.index = pd.to_datetime(train_smooth.index)
freq = "D"
train_smooth.index.freq = freq


# In[7]:


# Load the dataset of the test sets.
test = pd.read_csv(PATH_TO_DATA_FOLDER + "test_target.csv", header = [0, 1], index_col = 0)
test.index.name = "Datetime"
test.index = pd.to_datetime(test.index)
freq = "D"
test.index.freq = freq


# In[8]:


# Load the dataset of the whole time-series of the fcs indicator.
target = pd.read_csv(PATH_TO_DATA_FOLDER + "all_target.csv", header = [0, 1], index_col = 0)
target.index.name = "Datetime"
target.index = pd.to_datetime(target.index)
freq = "D"
target.index.freq = freq


# In[9]:


TRAIN = train_smooth.copy()


# In[10]:


TEST_SIZE = 30
FREQ = TRAIN.index.freq


# In[11]:


PROVINCES = TRAIN.columns.get_level_values(0).unique()


# In[12]:


PREDICTORS = TRAIN.columns.get_level_values(1).unique()


# In[13]:


# Get the training and test sets.
TRAIN_SETS = find_multiple_sets(TRAIN)
TEST_TARGET_SETS = find_multiple_sets(test)


# ## Training & Validation
# ### Parameters grid search

# In[19]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error


# In[20]:


# Define the LAGS to which perform the grid search.
space1 = {"1 Month Anomaly (%) Rainfall": hp.choice("1 Month Anomaly (%) Rainfall", np.append(np.arange(1, 100, 20), None)), 
          "3 Months Anomaly (%) Rainfall": hp.choice("3 Months Anomaly (%) Rainfall", np.append(np.arange(1, 100, 20), None)), 
          "Cereals and tubers": hp.choice("Cereals and tubers", np.append(np.arange(1, 100, 20), None)), 
          "Exchange rate (USD/LCU)": hp.choice("Exchange rate (USD/LCU)", np.append(np.arange(1, 100, 20), None)), 
          "FCS": hp.choice("FCS", np.arange(1, 100, 20)), 
          "Fatality": hp.choice("Fatality", np.append(np.arange(1, 100, 20), None)), 
          "NDVI Anomaly": hp.choice("NDVI Anomaly", np.append(np.arange(1, 100, 20), None)), 
          "Rainfall (mm)": hp.choice("Rainfall (mm)", np.append(np.arange(1, 100, 20), None)), 
          "rCSI": hp.choice("rCSI", np.append(np.arange(1, 100, 20), None)), 
          "Lat": hp.choice("Lat", np.append(np.arange(0, 1), None)), 
          "Lon": hp.choice("Lon", np.append(np.arange(0, 1), None)), 
          "Population": hp.choice("Population", np.append(np.arange(0, 1), None)), 
          "Ramadan": hp.choice("Ramadan", np.append(np.arange(1, 100, 20), None))}


# In[21]:


# Define the PARAMETERS MODEL to which perform the grid search.
space2 = {"max_depth": hp.choice("max_depth", range(5, 20, 1)),
          "learning_rate": hp.quniform("learning_rate", 0.01, 0.5, 0.01),
          "n_estimators": hp.choice("n_estimators", range(20, 100, 5)),
          "gamma": hp.quniform('gamma', 0, 0.50, 0.01),
          "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
          "subsample": hp.quniform("subsample", 0.1, 1, 0.01),
          "colsample_bytree": hp.quniform("colsample_bytree", 0.1, 1.0, 0.01)}


# In[22]:


# Merge the two dictionary to perform the grid search.
space = dict(space1, **space2)


# In[23]:


import xgboost as xgb


# In[69]:


def hyperparameters(space): 
    try:
        print(space)
        # Select lags.
        lags_dict = {key: space[key] for key in PREDICTORS}

        val_losses_h = list()
        for h in range(TEST_SIZE):
            X_train_list, y_train_list, X_val_list, y_val_list = list(), list(), list(), list()

            for train in TRAIN_SETS:
                # Create training and validation samples.  
                for PROVINCE in PROVINCES:
                    creator = LagsCreator(train[[PROVINCE]], lags_dictionary = lags_dict, target = "FCS", n_out = TEST_SIZE, 
                                          return_dataframe = True)
                    X_train, y_train, X_val, y_val, _ = creator.to_supervised(single_step = True, h = h+1, feature_time = True, 
                                                                              validation = True, dtype = np.float64)
                    X_train_list.append(X_train)
                    y_train_list.append(y_train)
                    X_val_list.append(X_val)
                    y_val_list.append(y_val)  

            X_train = pd.concat(X_train_list).reset_index(drop = True)
            y_train = pd.concat(y_train_list).reset_index(drop = True)


            # Train the model.
            print("Training %s samples for the prediction horizon h: %d" % (str(X_train.shape), h+1))
            model = xgb.XGBRegressor(n_estimators = space["n_estimators"], max_depth = int(space["max_depth"]), 
                                     learning_rate = space["learning_rate"], gamma = space["gamma"], 
                                     min_child_weight = space["min_child_weight"], subsample = space["subsample"], 
                                     colsample_bytree = space["colsample_bytree"], objective = "reg:squarederror", 
                                     tree_method = "gpu_hist", gpu_id = 1)
            model.fit(X_train, y_train)  

            y_hats_train = model.predict(X_train)
            # Compute training error.
            train_loss = mean_squared_error(y_train.values.flatten(), y_hats_train)
            r2_train = model.score(X_train, y_train)

            X_val = pd.concat(X_val_list).reset_index(drop = True)
            y_val = pd.concat(y_val_list).reset_index(drop = True)

            # Validation.
            y_hats_val = model.predict(X_val)
            # Compute validation error.
            val_loss = mean_squared_error(y_val.values.flatten(), y_hats_val)
            r2_val = model.score(X_val, y_val)

            val_losses_h.append(val_loss)

            # Recursive save results.
            results = space.copy()
            results["h"] = h+1
            results["r2_train"] = r2_train
            results["r2_val"] = r2_val
            results["val_loss"] = val_loss
            results["train_loss"] = train_loss
            df_space = pd.DataFrame(results, index = [0], dtype = object)
            filename = dir + "/grid_search.csv"
            df_space.to_csv(filename, index = False, header = (not os.path.exists(filename)), mode = "a")

        # Compute mean error of this 'space' for the various prediction horizions.
        val_loss = np.mean(val_losses_h)
    except:
        val_loss = np.inf     

    return {"loss": val_loss, "status": STATUS_OK}


# In[70]:


trials = Trials()
best = fmin(fn = hyperparameters,
            space = space,
            algo = tpe.suggest,
            max_evals = 500,
            trials = trials)

# Save the trials into a file.
pickle.dump(trials, open(dir + "/hyp_trials.p", "wb"))

