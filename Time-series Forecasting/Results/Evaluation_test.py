
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as mse
from ipywidgets import interact, widgets, fixed
from IPython.display import display
from scipy.signal import savgol_filter
import pickle
import ast
import os
import shutil
import itertools


# In[2]:


# Add the python path to the folder containing some useful custom packages.
import sys
sys.path.insert(0, "../../../packages/")
from TsIP.TsIP import TsIP
from LagsCreator.LagsCreator import LagsCreator
from NestedCV.NestedCV import NestedCV


# In[3]:


# Create the folder where to store the information arising from this notebook.
dir_output_test = "./output_test"
if not os.path.exists(dir_output_test):
    os.makedirs(dir_output_test)


# In[4]:


# Define the path of the folder containing the information of the hyperparameter tuning.
dir_hyper_params = "./Yemen2/output_hyperparameter_tuning/"
# Load the values of some global variables.
with open(dir_hyper_params + "global_variables", "rb") as f:
    COUNTRIES_TO_CONSIDER, TEST_SIZE, FRACTION_TRAIN_SET = pickle.load(f)


# In[5]:


# Define variable to predict (according to the variable in the 'creator_train_test' notebook).
TARGET = "FCS"
# Define the total number of splits (according to the splits in the 'creator_train_test' notebook).
TOTAL_SPLITS = 7


# In[6]:


# Define the main folder containing the training and test data.
dir_data = "./data"


# ## Time-series dataset

# In[7]:


# Load the data of the Yemen country.
df_yemen = pd.read_csv("../../../Dataset time-series/data/Yemen/Yemen.csv", header = [0, 1], index_col = 0)
df_yemen.index.name = "Datetime"
df_yemen.index = pd.to_datetime(df_yemen.index)
freq = "D"
df_yemen.index.freq = freq
df_yemen.columns = pd.MultiIndex.from_tuples(map(lambda x: ("Yemen", x[0], x[1]), df_yemen.columns), names = ["Country", "AdminStrata", "Indicator"])


# In[8]:


# Load the data of the Syria country.
df_syria = pd.read_csv("../../../Dataset time-series/data/Syria/Syria.csv", header = [0, 1], index_col = 0)
df_syria.index.name = "Datetime"
df_syria.index = pd.to_datetime(df_syria.index)
freq = "D"
df_syria.index.freq = freq
df_syria.columns = pd.MultiIndex.from_tuples(map(lambda x: ("Syria", x[0], x[1]), df_syria.columns), names = ["Country", "AdminStrata", "Indicator"])


# In[9]:


# Concatenate data.
df = pd.concat([df_yemen, df_syria], axis = 1)
# Delete the feature 'Milk and dairy' not presents for the Yemen country.
df = df.drop("Milk and dairy", axis = 1, level = 2)
# Consider the following dates.
df = df.loc["2018-01-01":"2020-08-31"]


# In[10]:


# Select countries.
df = df[COUNTRIES_TO_CONSIDER]
df.head()


# In[11]:


# Plot time-series.
#TsIP(df).interactive_plot_df(title = "Time-series", matplotlib = False, style = "lines", comparison = False)


# ## Nested cross validation/test

# In[12]:


# Create the nested cross validation.
cv = NestedCV(TOTAL_SPLITS, TEST_SIZE)


# In[13]:


# Total nested cross validation.
SPLITS = cv.get_splits(df, show = False)


# In[14]:


for split_number, (train, test) in SPLITS.items():
    print("Split %d: range of days to predict between %s - %s" % (split_number, str(test.index[0].date()), str(test.index[-1].date())))


# ## Hyper-parameters tuning results

# In[15]:


hyper_params = pd.read_csv(dir_hyper_params + "hyperparameter_tuning.csv")
hyper_params


# In[16]:


# Find the best attempt for each prediction horizon over each split.
metric = "loss_to_minimize"

def fmin(group):
    return group[metric].idxmin()

result = hyper_params.groupby(["split", "h"]).apply(fmin)


# In[17]:


best_result = hyper_params.loc[result].set_index(["split", "h"])
best_result


# In[18]:


def plot_r2_box_plot(data_train, data_test, ax, label_train = None, label_test = None, title = None, table_train = None, 
                     color_train = "#355269", color_test = "#5eb91e"):
    # Define x-ticks.
    x_ticks = np.arange(1, data_train.shape[1] + 1)
    medianprops = dict(linestyle = "-", linewidth = 2.5)
    # Boxplot.
    bptrain = ax.boxplot(data_train, positions = x_ticks*2.0 - 0.4, sym = "", widths = 0.6, medianprops = medianprops)
    bptest = ax.boxplot(data_test, positions = x_ticks*2.0 + 0.4, sym = "", widths = 0.6, medianprops = medianprops)

    # Draw temporary lines for legend.
    ax.plot([], c = color_train, label = label_train)
    ax.plot([], c = color_test, label = label_test)
    ax.legend(loc = "best", prop = {"size": 15})
    
    # Set attributes of the plot.
    ax.set_title(title, fontsize = 20)
    ax.set_xlabel("Prediction horizon", fontsize = 15)
    ax.set_ylabel("R$^2$", fontsize = 15)
    ax.set_ylim([0, 1])
    ax.tick_params(labeltop = False, labelright = True)
    ax.set_xticklabels(x_ticks)  
    
    # Insert information table.
    if table_train is not None:
        ax.table(cellText = table_train.values, rowLabels = table_train.index, colLabels = table_train.columns,
                 bbox = [0.0, -0.55, 1, .28], loc = "bottom")

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], color = color)
        plt.setp(bp["whiskers"], color = color)
        plt.setp(bp["caps"], color = color)
        plt.setp(bp["medians"], color = color)

    # Set the colors of the boxplots.
    set_box_color(bptrain, color_train) 
    set_box_color(bptest, color_test)

    # Add boxplots to the axis.
    ax = bptrain
    ax = bptest

    return ax


# In[19]:


# Create folder where to save results of the hyperparameter tuning.
dir_results_hyperparameter_tuning = dir_output_test + "/hyperparameter_tuning"

if os.path.exists(dir_results_hyperparameter_tuning):
    shutil.rmtree(dir_results_hyperparameter_tuning)
    os.makedirs(dir_results_hyperparameter_tuning)
else:
    os.makedirs(dir_results_hyperparameter_tuning)

with plt.style.context("default"):
    # Get the information of the r2 train for each prediction horizon considering the split information together.
    data_train = best_result[["r2_train"]].unstack().transpose()
    # Get the information of the r2 validation for each prediction horizon considering the split information together.
    data_validation = best_result[["r2_val"]].unstack().transpose()

    # Plot the box-plot.
    fig, axs = plt.subplots(figsize = (20, 4))
    plot_r2_box_plot(data_train.values.T, data_validation.values.T, axs, label_train = "Training", label_test = "Validation", 
                     color_train = "#355269", color_test = "#5eb91e")
    fig.savefig(dir_results_hyperparameter_tuning + "/all_splits.png", bbox_inches = "tight", dpi = 300)

    for i,split_number in enumerate(best_result.index.get_level_values(0).unique()):
        # Create dataframe/table with information about the train and validation shapes.
        table_train = best_result.loc[split_number][["shape_train", "shape_val"]].applymap(lambda x: ast.literal_eval(x))
        table_train["shape"] = table_train.apply(lambda x: (x.shape_train[0], x.shape_val[0], x.shape_train[1]), axis = 1)
        table_train = table_train["shape"].to_dict()
        table_train = pd.DataFrame(table_train, index = ["n° train points", "n° val points", "n° features"])
        # Plot the box-plot.
        fig, axs = plt.subplots(figsize = (20, 4))
        plot_r2_box_plot(data_train[[split_number]].values.T, data_validation[[split_number]].values.T, axs, 
                         label_train = "Training", label_test = "Validation", title = "Split %d" % split_number, 
                         table_train = table_train, color_train = "#355269", color_test = "#5eb91e")
        fig.savefig(dir_results_hyperparameter_tuning + "/split_%d.png" % split_number, bbox_inches = "tight", dpi = 300)


# In[20]:


def recursive_improvement(x):
    list_improvements = list()

    for i,value in enumerate(x):
        if i == 0:
            best_min = value
        else:
            if best_min > value:
                diff = best_min - value
                best_min = value
                list_improvements.append(diff)
            else:
                list_improvements.append(0)
            
    return list_improvements


# In[21]:


def f_recursive_improvement(x):
    x = x.to_list()
    return recursive_improvement(x)

hyper_params_recursive = hyper_params.set_index(["split", "h"])["loss_to_minimize"].groupby(axis = 0, level = [0, 1]).apply(f_recursive_improvement)


# In[22]:


for split_number in hyper_params_recursive.index.get_level_values(0).unique():
    hyper_params_recursive_1 = hyper_params_recursive.loc[split_number]
    hyper_params_recursive_1 = pd.DataFrame(hyper_params_recursive_1.tolist())
    hyper_params_recursive_1 = hyper_params_recursive_1.iloc[:,:200]
    
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize = (30, 18))
        im = ax.imshow(hyper_params_recursive_1.astype(float), cmap = "viridis")

        ax.set_yticks(hyper_params_recursive_1.index)
        ax.grid(False)
        ax.set_yticklabels(hyper_params_recursive_1.index + 1)
        ax.set_xlabel("Configuration", fontsize = 15)
        ax.set_ylabel("Prediction horizon", fontsize = 15)
        ax.set_title("Hyperopt - Split %d" % split_number, fontsize = 20)
        for i in range(hyper_params_recursive_1.shape[0]):
            ax.axhline(i + 0.5, color = "white", lw = 0.0)
        
        # Save the figure.
        fig.savefig(dir_results_hyperparameter_tuning + "/recursive_split_%s.png" % split_number, bbox_inches = "tight", dpi = 300)


# ## Forecasting

# In[23]:


QUANTILES = False
ALPHA_QUANTILES = 0.75
SMOOTH_PREDICTION = True


# In[24]:


# Open the names of the parameters of the model.
with open(dir_hyper_params + "space1", "rb") as fp:
    parameter_names_model = pickle.load(fp)


# In[25]:


# Open the names of the parameters of the features.
with open(dir_hyper_params + "space2", "rb") as fp:
    parameter_names_indicator = pickle.load(fp)


# In[26]:


# Load the lags dictionary.
with open (dir_hyper_params + "lags_dict", "rb") as fp:
    lags_dict = pickle.load(fp)


# In[27]:


def take_lags(x, lags = None, delay = False):
    if lags is not None:
        lags = [(x, "x(t)") if i == 1 else (x, "x(t-%d)" % (i-1)) for i in lags]
        if delay:
            lags.append((x, "delay"))
    else:
        lags = [(x, slice(None))]
    return lags


# In[31]:


# Define the splits to consider to compute the corresponding prediction based on hyperparameters tuning.
SPLITS_TO_USE = best_result.index.get_level_values(0).unique()


# In[32]:


def model(train, test, lags_dict, out, target, quantiles = False, hyper = False):
    """
    This function allows to predict 'out' steps ahead in the future of the 'target' variable of each site in the
    'train' group. The predictions of 'out' steps in the future start from the last date of the 'train' group 
    provided.
    
    """
    # Use the best parameters obtained through a previous hyperparameter tuning.
    if hyper:
        # Define the best parameters for the current split obtained by the hyperparameter tuning.
        parameter_names = parameter_names_model + parameter_names_indicator
        best_parameters = best_result.loc[split_number][parameter_names].astype(float)
        # Model parameters.
        best_parameter_indicator = best_parameters[parameter_names_indicator]
        # Indicators parameters.
        best_parameter_model = best_parameters[parameter_names_model]

    #####################
    ### DATA CREATION ###
    #####################
    
    print("Load data...")

    # Define the first level of multi-sites (countries level).
    countries = train.columns.get_level_values(0).unique()

    # Creation of an unique pot for putting the training points (X, y) for all the multi-sites (countries and provinces) for each prediction horizon.
    training_points = {"X": {h+1: [] for h in range(out)}, 
                       "y": {h+1: [] for h in range(out)}}
    # Creation of the input test points specifically for each site (country and province) and prediction horizon.
    test_input_points = {country: {province: {h+1: None for h in range(out)} for province in train[country].columns.get_level_values(0).unique()} for country in countries}

    for country in countries:
        # Select the subdataframe corresponding to the current country.
        train_country = train[country]
        # Define the second level of multi-sites (provinces level).
        provinces = train_country.columns.get_level_values(0).unique()
        for province in provinces:
            for h in range(out):
                # Training samples.
                X_train = pd.read_csv(dir_data + "/train/%s/%s/X_train_split%d_h%d.csv" % (country, province, split_number, h+1), header = [0, 1], index_col = 0) 
                y_train = pd.read_csv(dir_data + "/train/%s/%s/y_train_split%d_h%d.csv" % (country, province, split_number, h+1), header = [0, 1], index_col = 0) 
                X_test = pd.read_csv(dir_data + "/test/%s/%s/X_test_split%d_h%d.csv" % (country, province, split_number, h+1), header = [0, 1], index_col = 0) 

                # Get the features to keep for the current prediction horizon according to the hyperparameter tuning.
                if hyper:
                    # Select features.
                    # Decide the indicators to keep based on values (0 or 1).
                    space_features = {k: v for k,v in dict(best_parameter_indicator.loc[h+1]).items() if v == 1}
                    # Select the corresponding lags.
                    space_features = {feature: lags_dict[feature] for feature in space_features.keys()}
                    # Flatten list.
                    features = list(itertools.chain(*list(space_features.values())))
                    # Keep features.
                    X_train = pd.concat([X_train.loc[:, feature] for feature in features], axis = 1).sort_index(axis = 1)
                    X_test = pd.concat([X_test.loc[:, feature] for feature in features], axis = 1).sort_index(axis = 1)

                # Store information.
                training_points["X"][h+1].append(X_train)
                training_points["y"][h+1].append(y_train)
                test_input_points[country][province][h+1] = X_test

    # Concatenate training data for each prediction horizon in order to consider them into an unique pot.
    for h in range(out):
        training_points["X"][h+1] = pd.concat(training_points["X"][h+1]).reset_index(drop = True) 
        training_points["y"][h+1] = pd.concat(training_points["y"][h+1]).reset_index(drop = True) 
        
    print("Complete!")
        
    ###################
    ### FORECASTING ###
    ###################
    
    print("Forecasting...")

    # Create the dataframe where to store the predictions of the target.
    c1 = pd.MultiIndex.from_tuples(map(lambda x: (x[0], x[1], "lower_quantile"), train.columns.droplevel(2)), names = ["Country", "AdminStrata", "Target"])
    c2 = pd.MultiIndex.from_tuples(map(lambda x: (x[0], x[1], "Forecast"), train.columns.droplevel(2)), names = ["Country", "AdminStrata", "Target"])
    c3 = pd.MultiIndex.from_tuples(map(lambda x: (x[0], x[1], "upper_quantile"), train.columns.droplevel(2)), names = ["Country", "AdminStrata", "Target"])
    predictions = pd.DataFrame(index = test.index, columns = c1.union(c2).union(c3))   

    # Training model.
    models = {h+1: None for h in range(out)}
    r2_train = {h+1: None for h in range(out)}
    r2_test = {country: {h+1: None for h in range(out)} for country in countries}
    for h in range(out):
        # Train the model for the current prediction horizon.
        X_train, y_train = training_points["X"][h+1], training_points["y"][h+1]
        
        # Get the best model parameters for the current prediction horizon if exist the information about.
        if hyper:
            # Select best model parameters.
            best_parameter_model_h = dict(best_parameter_model.loc[h+1])
            # Convert to int type the float numbers that are integers.
            best_parameter_model_h = {k: int(v) if v.is_integer() else v for k,v in best_parameter_model_h.items()}

        # Model.
        if quantiles:
            # Train model.
            model = GradientBoostingRegressor(n_estimators = 100)
            model.set_params(loss = "ls")
            model.fit(X_train, y_train.flatten())
            # Lower model.
            model_lower = GradientBoostingRegressor(n_estimators = 100)
            model_lower.set_params(loss = "quantile", alpha = 1.-ALPHA_QUANTILES)
            model_lower.fit(X_train, y_train.flatten())
            # Upper model.
            model_upper = GradientBoostingRegressor(n_estimators = 100)
            model_upper.set_params(loss = "quantile", alpha = ALPHA_QUANTILES)
            model_upper.fit(X_train, y_train.flatten())
        else:  
            if hyper:
                model = xgb.XGBRegressor(**best_parameter_model_h, objective = "reg:squarederror")
            else:
                model = xgb.XGBRegressor(n_estimators = 100, objective = "reg:squarederror")
            # Train model.
            model.fit(X_train, y_train)
            # Lower model.
            model_lower = None
            # Upper model.
            model_upper = None
        
        # Save models.
        models[h+1] = (model, model_lower, model_upper, X_train.columns)
        # Save training r2 scores.
        r2_train[h+1] = model.score(X_train, y_train)
       
        # Forecasting.
        for country in countries:
            X_test_list, y_test_list = list(), list()
            # Define the second level multi-sites (provinces).
            provinces = train[country].columns.get_level_values(0).unique()
            for province in provinces:
                X_test = test_input_points[country][province][h+1]
                y_hats = model.predict(X_test)[0]
                
                # Save the true information about this test point.
                X_test_list.append(X_test)
                y_true = test[(country, province, target)].loc[predictions.index[h]] 
                y_test_list.append([y_true])

                # Store the predicted values into the dataframe.
                predictions[(country, province, "Forecast")].loc[predictions.index[h]] = y_hats

                # Prediction for the quantiles.
                if quantiles:
                    y_hats_lower = model_lower.predict(X_test)[0]
                    predictions[(country, province, "lower_quantile")].loc[predictions.index[h]] = y_hats_lower
                    y_hats_upper = model_upper.predict(X_test)[0]
                    predictions[(country, province, "upper_quantile")].loc[predictions.index[h]] = y_hats_upper

            # Compute the r2 test for the current prediction horizon and country.
            r2_test_country = model.score(pd.concat(X_test_list).reset_index(drop = True), np.expand_dims(np.concatenate(y_test_list), 1))
            r2_test[country][h+1] = r2_test_country  
 
    # Define the shape of the training and test points.    
    shape_training_points = training_points.copy()
    shape_test_points = test_input_points.copy()
    for h in range(out):
        shape_training_points["X"][h+1] = shape_training_points["X"][h+1].shape
        shape_training_points["y"][h+1] = shape_training_points["y"][h+1].shape
        for country in countries:
            provinces = train[country].columns.get_level_values(0).unique()
            for province in provinces:
                shape_test_points[country][province][h+1] = shape_test_points[country][province][h+1].shape
                
    print("Complete!")

    return predictions, models, r2_train, r2_test, shape_training_points, shape_test_points


# In[33]:


# Create a dictionary to store forecasting information for each split.
information_to_store = {"shape_train": None, # Save the shape of the training points.
                        "shape_test": None, # Save the shape of the test points.
                        "r2_train": None, # Save the r2 on the training points.
                        "r2_test": None, # Save the r2 on the test points.
                        "models": None, # Save the trained models (one for each prediction horizon) with the corresponding feature names.
                        "prediction_sites": None, # Save the predictions for each site (province).
                        "loss_sites": None, # Save the prediction loss (mse) for each site (province).
                        "loss_overall": None, # Save the overall loss. 
                        "loss_h": None} # Save the loss as function of the prediction horizon among all the sites (province).

TOTAL_RESULTS = {split_number: information_to_store.copy() for split_number in SPLITS_TO_USE}

# Forecasting.
for split_number, (train, test) in SPLITS.items():
    if split_number in SPLITS_TO_USE:
        print("SPLIT %d/%d" % (split_number, len(SPLITS.keys())))

        ## ACTUAL ##
        # Define the test points of the target to predict for each site (country and province).
        true = test.xs(TARGET, axis = 1, level = 2, drop_level = False)
        # Define the number of days to predict.
        test_size = len(true)
        print("Range of days to predict: %s - %s" % (str(true.index[0].date()), str(true.index[-1].date())))

        ## NAIVE ##
        # Define the predictions for the Naive model.
        naive = train.xs(TARGET, axis = 1, level = 2, drop_level = False).iloc[-1].to_frame().transpose()
        naive = naive.loc[naive.index.repeat(test_size)]
        naive = naive.rename(columns = {"FCS": "Naive"})
        naive.index = true.index

        ## MODEL ##
        # Train the model to predict the test_size points for each site (country and province).
        predictions, models, r2_train, r2_test, shape_train, shape_test = model(train, test, lags_dict, TEST_SIZE, TARGET, 
                                                                                quantiles = False, hyper = True)

        # Smooth the output prediction over the prediction horizons.
        if SMOOTH_PREDICTION:
            def smooth_output(serie):
                if serie.isna().sum() > 0:
                    return serie
                else:
                    # Smooth serie.
                    smooth_serie = savgol_filter(serie, 15, 3)
                    return smooth_serie
            predictions = predictions.apply(smooth_output)

        ## ALL ##
        results = pd.concat([true, predictions, naive], axis = 1).sort_index(axis = 1, level = 0)
        
        # Analysis of the forecast results.  
        # Define the total prediction loss for each site (province) not considering quantiles.
        select = results.columns.get_level_values(2).isin(["Forecast", TARGET, "Naive"])
        results_no_quantiles = results.loc[:, select]

        def f_loss(x, level):
            # Model.
            model = mse(x.xs(TARGET, axis = 1, level = level), x.xs("Forecast", axis = 1, level = level))
            # Naive.
            naive = mse(x.xs(TARGET, axis = 1, level = level), x.xs("Naive", axis = 1, level = level))  
            return pd.Series([model, naive], index = ["Model", "Naive"])

        loss_sites = results_no_quantiles.groupby(axis = 1, level = [0, 1]).apply(lambda x: f_loss(x, 2)).transpose()

        # Define the overall loss.
        loss = loss_sites.groupby(axis = 0, level = 0).mean().mean()
        print("Overall loss:")
        print(loss)
        print("Country loss:")
        print(loss_sites.groupby(axis = 0, level = 0).mean())

        # Compute the loss as a function of the prediction horizon among all the sites (provinces).
        loss_h = results_no_quantiles.transpose().unstack(2).groupby(axis = 0, level = 0).apply(lambda x: x.groupby(axis = 1, level = 0).apply(lambda x: f_loss(x, 1)).transpose())

        # Save the results for the current split.
        TOTAL_RESULTS[split_number]["r2_train"] = r2_train 
        TOTAL_RESULTS[split_number]["r2_test"] = r2_test  
        TOTAL_RESULTS[split_number]["shape_train"] = shape_train 
        TOTAL_RESULTS[split_number]["shape_test"] = shape_test 
        TOTAL_RESULTS[split_number]["models"] = models 
        TOTAL_RESULTS[split_number]["prediction_sites"] = results
        TOTAL_RESULTS[split_number]["loss_sites"] = loss_sites
        TOTAL_RESULTS[split_number]["loss_overall"] = loss
        TOTAL_RESULTS[split_number]["loss_h"] = loss_h
