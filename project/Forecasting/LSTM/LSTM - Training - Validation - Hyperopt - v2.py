
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from keras_tqdm import TQDMCallback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import pickle


# In[ ]:


# Selection of the gpu.
GPU = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in GPU)  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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


# Load the dataset of the training sets.
train = pd.read_csv(PATH_TO_DATA_FOLDER + "train_smooth.csv", header = [0, 1], index_col = 0)
train.index.name = "Datetime"
train.index = pd.to_datetime(train.index)
freq = "D"
train.index.freq = freq


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


TEST_SIZE = 30
FREQ = train.index.freq


# In[10]:


TRAIN = train.copy()


# In[11]:


PROVINCES = TRAIN.columns.get_level_values(0).unique()


# In[12]:


PREDICTORS = TRAIN.columns.get_level_values(1).unique()


# ## Data source transformation
# 
# I decide to normalize the data among the provinces considering indicator by indicator and considering only the training sets.

# In[13]:


global SCALERS

MIN = 0
MAX = 1
SCALERS = dict()
def normalization(group, feature_range):
    min_, max_ = feature_range
    min_group = group.min().min()
    max_group = group.max().max()
    
    # Normalization.
    group_std = (group - min_group) / (max_group - min_group)
    group_scaled = group_std * (max_ - min_) + min_

    # Save the scalers for the various indicators.
    SCALERS[group.name] = (min_group, max_group)

    return group_scaled


# In[14]:


TRAIN_NORMALIZED = TRAIN.groupby(axis = 1, level = 1).apply(lambda x: normalization(x, (MIN, MAX)))


# In[15]:


# Plot time-series.
#TsIP(TRAIN_NORMALIZED).interactive_plot_df(title = "Training sets", matplotlib = False, style = "lines")


# In[16]:


def denormalization(group_scaled, indicator, feature_range, scalers):
    min_, max_ = feature_range
    min_group, max_group = scalers[indicator]

    group_std = (group_scaled - min_) / (max_ - min_)
    group = (group_std * (max_group - min_group)) + min_group
    
    return group


# In[17]:


# Get the training and test sets.
TRAIN_NORMALIZED_SETS = find_multiple_sets(TRAIN_NORMALIZED)
TEST_TARGET_SETS = find_multiple_sets(test)


# ## Training & Validation
# ### Parameters grid search

# In[18]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error

# Define the PARAMETERS MODEL to which perform the grid search.
space = {"lags": hp.choice("lags", np.arange(1, 100, 5)), 
         "batch_size": hp.choice("batch_size", np.array([2**j for j in range(3, 10)])), 
         "LSTM": hp.randint("LSTM", 1, 100), 
         "Conv1": hp.randint("Conv1", 1, 64),
         "Conv2": hp.randint("Conv2", 1, 5),
         "Dense": hp.randint("Dense", 1, 100), 
         "Dropout": hp.uniform("Dropout", 0, 0.5)}


# In[19]:


def network(timesteps, features, n_out, lstm, conv1, conv2, dense, dropout):
    
    inp_seq = Input(shape = (timesteps, features))
    
    x = Bidirectional(LSTM(lstm, return_sequences = True))(inp_seq)
    x = AveragePooling1D(2)(x)
    x = Conv1D(conv1, conv2, activation = "relu", padding = "same", name = "extractor")(x)
    x = Flatten()(x)
    x = Dense(dense, activation = "relu")(x)
    x = Dropout(dropout)(x)
    
    out = Dense(n_out)(x)
    
    model = Model(inp_seq, out)
    
    return model


# In[20]:


N_EPOCHS = 1000


# In[21]:


def hyperparameters(space):      
    try:
        # Define the parameters to grid search.
        LAGS = int(space["lags"])
        BATCH_SIZE = int(space["batch_size"])
        lstm = int(space["LSTM"])
        conv1 = int(space["Conv1"])
        conv2 = int(space["Conv2"])
        dense = int(space["Dense"])
        dropout = space["Dropout"]

        lags_dict = dict()
        # Define lags for each indicator.
        lags_dict["3 Months Anomaly (%) Rainfall"] = LAGS
        lags_dict["1 Month Anomaly (%) Rainfall"] = LAGS
        lags_dict["Cereals and tubers"] = LAGS
        lags_dict["Exchange rate (USD/LCU)"] = LAGS
        lags_dict["FCS"] = LAGS
        lags_dict["Fatality"] = LAGS
        lags_dict["NDVI Anomaly"] = LAGS
        lags_dict["Rainfall (mm)"] = LAGS
        lags_dict["rCSI"] = LAGS
        lags_dict["Lat"] = LAGS
        lags_dict["Lon"] = LAGS
        lags_dict["Population"] = LAGS
        lags_dict["Ramadan"] = LAGS

        # Randomly select only some predictors.
        #predictors = list(np.random.choice(PREDICTORS, size = np.random.randint(len(PREDICTORS) + 1), replace = False))
        #if "FCS" not in predictors:
        #    predictors.append("FCS")      
        #for k,v in lags_dict.items():
        #    if k not in predictors:
        #        lags_dict[k] = None

        X_train_list, y_train_list, X_val_list, y_val_list = list(), list(), list(), list()
        # Create training and validation points starting from the training sets.
        for train_normalized in TRAIN_NORMALIZED_SETS:
            # Create training points and validation points from the training set.
            for PROVINCE in PROVINCES:
                # Initialize lags creator.
                creator = LagsCreator(train_normalized[[PROVINCE]], lags_dictionary = lags_dict, target = "FCS")
                # Get samples.
                X_train, y_train, X_val, y_val, _ = creator.to_supervised(n_out = TEST_SIZE, single_step = False, return_dataframe = False, 
                                                                          feature_time = False, validation = True, dtype = np.float32)

                # Add a list of all the training and validation samples of all the provinces together.
                X_train_list.append(X_train)
                y_train_list.append(y_train)
                X_val_list.append(X_val)
                y_val_list.append(y_val)

        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_val = np.concatenate(X_val_list)
        y_val = np.concatenate(y_val_list)

        N_FEATURES = X_train.shape[2]

        # Model.
        model = network(LAGS, N_FEATURES, TEST_SIZE, lstm, conv1, conv2, dense, dropout)
        # Compile model.
        model.compile(loss = "mse", optimizer = "adam")

        # Patient early stopping.
        es = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 100)
        # Fit model.
        history = model.fit(X_train, y_train, epochs = N_EPOCHS, validation_data = (X_val, y_val), 
                            batch_size = BATCH_SIZE, verbose = 0, shuffle = True, 
                            callbacks = [es, TQDMCallback(outer_description = "Loading:", leave_inner = False, leave_outer = False)])

        # Save the number of epochs at which fit stop due to early stopping.
        number_of_epochs_it_ran = len(history.history["loss"])  
        train_loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]

        # Recursive save results.
        results = space.copy()
        results.update(lags_dict)
        results["epoch"] = number_of_epochs_it_ran
        results["val_loss"] = val_loss
        results["train_loss"] = train_loss
        df_space = pd.DataFrame(results, index = [0], dtype = object)
        filename = dir + "/grid_search.csv"
        df_space.to_csv(filename, index = False, header = (not os.path.exists(filename)), mode = "a")

        # Recursive save best model.
        best_val_loss = pd.read_csv(dir + "/grid_search.csv")
        best_val_loss = best_val_loss.iloc[best_val_loss.val_loss.idxmin()].val_loss
        if val_loss <= best_val_loss:
            model.save(dir + "/best_model.h5")

        K.clear_session()

    except:
        val_loss = np.inf     
        K.clear_session()

    return {"loss": val_loss, "status": STATUS_OK}


# In[22]:


trials = Trials()
best = fmin(fn = hyperparameters,
            space = space,
            algo = tpe.suggest,
            max_evals = 1000,
            trials = trials)

# Save the trials into a file.
pickle.dump(trials, open(dir + "/hyp_trials.p", "wb"))

