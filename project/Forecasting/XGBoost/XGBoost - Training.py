
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Add the python path to the folder containing some useful custom packages.
import sys
sys.path.insert(0, "../../packages/")
from TsIP.TsIP import TsIP
from tools import find_multiple_sets
from LagsCreator.LagsCreator import LagsCreator


# ## Dataset

# In[3]:


COUNTRY = "Yemen"


# In[4]:


PATH_TO_DATA_FOLDER = "../../Dataset time-series/data/" + COUNTRY + "/"


# In[5]:


# Load the dataset of the smoothed training sets.
train_smooth = pd.read_csv(PATH_TO_DATA_FOLDER + "train_smooth.csv", header = [0, 1], index_col = 0)
train_smooth.index.name = "Datetime"
train_smooth.index = pd.to_datetime(train_smooth.index)
freq = "D"
train_smooth.index.freq = freq


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


TRAIN = train_smooth.copy()


# In[9]:


TEST_SIZE = 30
FREQ = TRAIN.index.freq


# In[10]:


PROVINCES = TRAIN.columns.get_level_values(0).unique()
PROVINCES


# In[11]:


PREDICTORS = TRAIN.columns.get_level_values(1).unique()
PREDICTORS


# In[12]:


# Get the training and test sets.
TRAIN_SETS = find_multiple_sets(TRAIN)
TEST_TARGET_SETS = find_multiple_sets(test)


# ## Forecasting

# In[13]:


lags_dict = dict()
# Define lags for each indicator.
lags_dict["3 Months Anomaly (%) Rainfall"] = 7
lags_dict["1 Month Anomaly (%) Rainfall"] = 7
lags_dict["Cereals and tubers"] = 7
lags_dict["Exchange rate (USD/LCU)"] = 7
lags_dict["FCS"] = 7
lags_dict["Fatality"] = 7
lags_dict["NDVI Anomaly"] = 7
lags_dict["Rainfall (mm)"] = 7
lags_dict["rCSI"] = 7
lags_dict["Lat"] = 0
lags_dict["Lon"] = 0
lags_dict["Population"] = 0
lags_dict["Ramadan"] = 7


# In[14]:


import xgboost as xgb


# In[ ]:


import time


# In[15]:


FORECASTING = test.copy()


# In[17]:


t1=time.time()
# Forecasting for each prediction horizon of each test set.
for h in range(TEST_SIZE):
    X_train_list, y_train_list = list(), list()
    X_test_dict = dict()
    for i, train in enumerate(TRAIN_SETS):
        for PROVINCE in PROVINCES:
            creator = LagsCreator(train[[PROVINCE]], lags_dictionary = lags_dict, target = "FCS", n_out = TEST_SIZE, return_dataframe = True)
            X_train, y_train, _, _, X_test = creator.to_supervised(single_step = True, h = h+1, feature_time = True, validation = False, 
                                                                   dtype = np.float64)
            # Add a list of all the training samples of all the provinces together.
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            
            # Add the test sample of the province of the current set into a dictionary.
            X_test_dict[(PROVINCE, i)] = X_test

    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)

    print("Training %s samples for the prediction horizon h: %d" % (str(X_train.shape), h+1))
    model = xgb.XGBRegressor(n_estimators = 100, objective = "reg:squarederror") # tree_method = "gpu_hist", gpu_id = 0
    model.fit(X_train, y_train)  

    # Prediction.
    for i, test_set in enumerate(TEST_TARGET_SETS):
        for PROVINCE in PROVINCES:
            X_test = X_test_dict[(PROVINCE, i)]
            y_hat = model.predict(X_test)[0]
            FORECASTING[(PROVINCE, "FCS")].loc[test_set.index[h]] = y_hat
            
t2=time.time()
print(t2-t1)


