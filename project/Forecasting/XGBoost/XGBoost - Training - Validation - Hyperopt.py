
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import pickle
import os
import shutil
from IPython.display import clear_output


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


PATH_TO_DATA_FOLDER = "../../Dataset time-series/"


# In[5]:


# Load the dataset of the training sets.
train = pd.read_csv(PATH_TO_DATA_FOLDER + "train_smooth.csv", header = [0, 1], index_col = 0)
train.index.name = "Datetime"
train.index = pd.to_datetime(train.index)
freq = "D"
train.index.freq = freq


# In[6]:


# Load the dataset of the test sets.
test = pd.read_csv(PATH_TO_DATA_FOLDER + "test_target.csv", header = [0, 1], index_col = 0)
test.index.name = "Datetime"
test.index = pd.to_datetime(test.index)
freq = "D"
test.index.freq = freq


# In[7]:


# Load the dataset of the whole time-series of the fcs indicator.
target = pd.read_csv(PATH_TO_DATA_FOLDER + "all_target.csv", header = [0, 1], index_col = 0)
target.index.name = "Datetime"
target.index = pd.to_datetime(target.index)
freq = "D"
target.index.freq = freq


# In[8]:


TEST_SIZE = 30
FREQ = train.index.freq


# In[9]:


TRAIN = train.copy()


# In[10]:


PROVINCES = TRAIN.columns.get_level_values(0).unique()


# In[11]:


PREDICTORS = TRAIN.columns.get_level_values(1).unique()


# In[12]:


# Get the training and test sets.
TRAIN_NORMALIZED_SETS = find_multiple_sets(train)
TEST_TARGET_SETS = find_multiple_sets(test)


# ## Training & Validation
# ### Parameters grid search

# In[13]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error

# Define the PARAMETERS MODEL to which perform the grid search.
space = {"1 Month Anomaly (%) Rainfall": hp.randint("1 Month Anomaly (%) Rainfall", 1, 100), 
         "3 Months Anomaly (%) Rainfall": hp.randint("3 Months Anomaly (%) Rainfall", 1, 100), 
         "Cereals and tubers": hp.randint("Cereals and tubers", 1, 100), 
         "Exchange rate (USD/LCU)": hp.randint("Exchange rate (USD/LCU)", 1, 100), 
         "FCS": hp.randint("FCS", 1, 100), 
         "Fatality": hp.randint("Fatality", 1, 100), 
         "NDVI Anomaly": hp.randint("NDVI Anomaly", 1, 100), 
         "Rainfall (mm)": hp.randint("Rainfall (mm)", 1, 100), 
         "rCSI": hp.randint("rCSI", 1, 100), 
         "Lat": hp.randint("Lat", 0, 1), 
         "Lon": hp.randint("Lon", 0, 1), 
         "Population": hp.randint("Population", 0, 1), 
         "Ramadan": hp.randint("Ramadan", 1, 100)}


# In[14]:


import xgboost as xgb


# In[15]:


def hyperparameters(space):  
    try:
        for h in range(TEST_SIZE):
            X_train_list, y_train_list, X_val_list, y_val_list = list(), list(), list(), list()
            for train_normalized in TRAIN_NORMALIZED_SETS:
                # Create training and validation samples.  
                for PROVINCE in PROVINCES:
                    creator = LagsCreator(train_normalized[[PROVINCE]], lags_dictionary = space, target = "FCS")
                    X, y, X_val, y_val, X_test = creator.to_supervised(n_out = TEST_SIZE, single_step = True, h = h+1, 
                                                                       return_dataframe = True, feature_time = True, validation = True, 
                                                                       return_single_level = True, dtype = np.float64)
                    X_train_list.append(X)
                    y_train_list.append(y)
                    X_val_list.append(X_val)
                    y_val_list.append(y_val)  

            X_train = pd.concat(X_train_list).reset_index(drop = True)
            y_train = pd.concat(y_train_list).reset_index(drop = True)

            # Train the model.
            #print("Training for the prediction horizon h: %d" % (h+1))
            model = xgb.XGBRegressor(objective = "reg:squarederror", n_estimators = 100)   
            model.fit(X_train, y_train)  

            y_hats_train = model.predict(X_train)
            # Compute training error.
            train_loss = mean_squared_error(y_train.values.flatten(), y_hats_train)

            X_val = pd.concat(X_val_list).reset_index(drop = True)
            y_val = pd.concat(y_val_list).reset_index(drop = True)

            # Validation.
            y_hats_val = model.predict(X_val)
            # Compute validation error.
            val_loss = mean_squared_error(y_val.values.flatten(), y_hats_val)

            # Recursive save results.
            results = space.copy()
            results["h"] = h+1
            results["val_loss"] = val_loss
            results["train_loss"] = train_loss
            df_space = pd.DataFrame(results, index = [0], dtype = object)
            filename = dir + "/grid_search.csv"
            df_space.to_csv(filename, index = False, header = (not os.path.exists(filename)), mode = "a")
    except:
        val_loss = np.inf     

    return {"loss": val_loss, "status": STATUS_OK}


# In[16]:


trials = Trials()
best = fmin(fn = hyperparameters,
            space = space,
            algo = tpe.suggest,
            max_evals = 2,
            trials = trials)

# Save the trials into a file.
pickle.dump(trials, open(dir + "/hyp_trials.p", "wb"))

