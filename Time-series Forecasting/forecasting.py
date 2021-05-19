import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import argparse
import os
import shutil
import pickle
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error as mse

from _gui import *
from _default import *
from _utils import *
from _model import *

# Add the python path to the folder containing some custom packages.
import sys
sys.path.insert(0, "../packages/")
from NestedCV.NestedCV import NestedCV

###############################
### USER-DEFINED PARAMETERS ###
###############################

parser_user = argparse.ArgumentParser(description = "This file allows to forecast the target time-series at provincial level using the configuration defined during the creation of the dataset. It is possible to use the parameters found through a previous hyperparameter tuning.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

# Example usage: python forecasting.py --folder_path_to_dataset "./Yemen/dataset" --folder_path_to_workspace "./Yemen/out_hyper_1/out_test" --folder_path_to_hyperparameters "./Yemen/out_hyper_1" --n_jobs 3

parser_user.add_argument('--folder_path_to_dataset', type = str, default = "./dataset", help = "The path to the folder containing the dataset (training and test points).")
parser_user.add_argument('--folder_path_to_workspace', type = str, default = "./output_forecasting", help = "The path to the folder where all the results arising from the current analysis will be stored.")
parser_user.add_argument('--folder_path_to_hyperparameters', type = str, help = "The path to the folder containing the results obtained from a previous hyperparameter tuning. If defined, the forecasting is performed using the corresponding best configurations found, otherwise you can define a configuration through a GUI interface. In the last case, the xgboost parameter are set to their default values.")
parser_user.add_argument('--no_smooth_prediction', action = "store_false", help = "If you don't want to smooth the output predictions.")
parser_user.add_argument('--n_jobs', type = int, default = 1, help = "Define the number of 'n_job' of the xgboost model.")
parser_user.add_argument('--importance_type', choices = ["weight", "gain", "cover", "total_gain", "total_cover"], default = "weight", help = "Define the type of xgboost feature importance.")
parser_user.add_argument('--gui_interface', action = "store_true", help = "If you want to select the time features and the lags for each indicator through a GUI interface otherwise the corresponding default values are taken (see *_default*). This option is valid only if you don't use the results from a previous hyperparameter tuning.")

args = parser_user.parse_args()

#################
### WORKSPACE ###
#################

# Create the workspace folder where all the results arising from the current analysis will be stored.
if os.path.exists(args.folder_path_to_workspace): 
    shutil.rmtree(args.folder_path_to_workspace) 
os.makedirs(args.folder_path_to_workspace)
os.makedirs(args.folder_path_to_workspace + "/features_selection")
os.makedirs(args.folder_path_to_workspace + "/features_importance")

# Load the values of some global variables defined during the creation of the dataset.
with open(args.folder_path_to_dataset + "/global_variables", "rb") as f:
    COUNTRIES, TARGET, TEST_SIZE, NUMBER_OF_SPLITS, FEATURES_TIME, FORMAT, _ = pickle.load(f)
    
# Load the time-series dataset defined during the creation of the dataset.
df = pd.read_csv(args.folder_path_to_dataset + "/dataset.csv", header = [0, 1, 2], index_col = 0)
df.index.name = "Datetime"
df.index = pd.to_datetime(df.index)
df.index.freq = "D"

# Create the nested cross validation.
cv = NestedCV(NUMBER_OF_SPLITS, TEST_SIZE)
# Total nested cross validation.
SPLITS = cv.get_splits(df)

# Check if you want to use the results obtained from an hyperparameter tuning analysis.
if args.folder_path_to_hyperparameters is not None:
    # Load the file containing the hyperparameter tuning results.
    hyper_params = pd.read_csv(args.folder_path_to_hyperparameters + "/hyperparameter_tuning.csv") 
    # Get the best parameters for each prediction horizon over each split.
    result = hyper_params.groupby(["split", "h"]).apply(lambda x: x["loss_to_minimize"].idxmin())
    BEST_RESULTS = hyper_params.loc[result].set_index(["split", "h"])
    # Define the splits to analysis based on those used during the hyperparameter tuning.
    SPLITS_TO_CONSIDER = BEST_RESULTS.index.get_level_values(0).unique()
    # Load the lags dictionary defined during the hyperparameter tuning.
    with open (args.folder_path_to_hyperparameters + "/lags_dict", "rb") as fp:
        LAGS_DICT = pickle.load(fp)
    # Load the names of the parameters of the model.
    with open(args.folder_path_to_hyperparameters + "/space1", "rb") as fp:
        PARAMETER_NAMES_MODEL = pickle.load(fp)
        PARAMETER_NAMES_MODEL = list(PARAMETER_NAMES_MODEL.keys())
    # Load the names of the parameters of the indicators.
    with open(args.folder_path_to_hyperparameters + "/space2", "rb") as fp:
        PARAMETER_NAMES_FEATURE = pickle.load(fp)   
        PARAMETER_NAMES_FEATURE = list(PARAMETER_NAMES_FEATURE.keys())
    # Store information regarding hyperparameter tuning.
    HYPER = (BEST_RESULTS, PARAMETER_NAMES_MODEL, PARAMETER_NAMES_FEATURE)
else:
    # Define the splits to consider.
    SPLITS_TO_CONSIDER = np.arange(1,NUMBER_OF_SPLITS+1) 
    # Load the lags dictionary defined during the creation of the dataset.
    with open (args.folder_path_to_dataset + "/lags_dict", "rb") as fp:
        LAGS_DICT = pickle.load(fp)
    # GUI interface (it allows to modify default time features and lags).
    if args.gui_interface:
        interface = gui()
        out = interface.GUI2(FEATURES_TIME, LAGS_DICT, defaultTimes2, defaultLags2)
        # Check parameters.
        if out is None:
            raise ValueError("No values selected. You have to press 'Run' button.") 
        elif out[1] is {}:
            raise ValueError("You have to set a lag value for at least one indicator.") 
    else:
        out = (defaultTimes2.copy(), defaultLags2.copy())        
    # Add time features to lags dictionary.
    LAGS_DICT = out[1].copy()
    for feature in out[0]:
        LAGS_DICT[feature] = None
    # Store information regarding hyperparameter tuning.
    HYPER = None
    
############
### MAIN ###
############
        
# Save forecasting into an Excel file.
datasetsExcel = {TARGET: df.xs(TARGET, axis = 1, level = 2, drop_level = False).dropna()}
# Save training shapes.
training_shapes = list()

# Forecasting.
for split_number, (train, test) in SPLITS.items():
    if split_number in SPLITS_TO_CONSIDER:
        print(f"SPLIT {split_number}/{len(SPLITS.keys())}")
        
        ## ACTUAL ##
        # Define the test points of the target to predict for each site (country and province).
        true = test.xs(TARGET, axis = 1, level = 2, drop_level = False)
        # Define the number of days to predict.
        test_size = len(true)
        print(f"Range of days to predict: {true.index[0].date()} - {true.index[-1].date()}")

        ## NAIVE ##
        # Define the predictions for the Naive model.
        naive = train.xs(TARGET, axis = 1, level = 2, drop_level = False).iloc[-1].to_frame().transpose()
        naive = naive.loc[naive.index.repeat(test_size)]
        naive = naive.rename(columns = {TARGET: "Naive"})
        naive.index = true.index

        ## MODEL ##
        # Train the model to predict the test_size points for each site (country and province).
        predictions, models, r2_train, shape_train = model(train, test, LAGS_DICT, TEST_SIZE, TARGET, split_number, hyper = HYPER, 
                                                           format = FORMAT, dir_data = args.folder_path_to_dataset, 
                                                           importance_type = args.importance_type, n_jobs = args.n_jobs)

        
        # Save the features and features importance.
        with open(args.folder_path_to_workspace + f"/features_importance/importance_type.txt", "w") as fp:
            fp.write("Feature importance type: %s" % (args.importance_type))
        
        for h in models.keys():
            with open(args.folder_path_to_workspace + f"/features_selection/features_split_{split_number}_h_{h}", "wb") as fp:
                pickle.dump(models[h][1], fp)
            with open(args.folder_path_to_workspace + f"/features_importance/features_split_{split_number}_h_{h}", "wb") as fp:
                pickle.dump(models[h][2], fp)
        
        # Smooth predictions.
        if args.no_smooth_prediction:
            predictions = predictions.apply(lambda x: savgol_filter(x, 15, 3))

        ## ALL ##
        results = pd.concat([true, predictions, naive], axis = 1).sort_index(axis = 1, level = 0)
        
        # Store predictions.
        datasetsExcel[f"Split {split_number}"] = results.copy()
        # Store training shape.
        shape_train = pd.DataFrame(shape_train["X"].values(), index = shape_train["X"].keys(), 
                                   columns = pd.MultiIndex.from_product([[f"Split {split_number}"], ["training points", "features"]], 
                                                                        names = ["Split", "Info"]))
        training_shapes.append(shape_train)

        
# Save the predictions into an Excel file. 
excel_filename = args.folder_path_to_workspace + "/forecast.xlsx"
with pd.ExcelWriter(excel_filename) as writer:
    for sheet, data in datasetsExcel.items():
        data.to_excel(writer, sheet_name = sheet)

# Save training points.
training_shapes = pd.concat(training_shapes, axis = 1)
training_shapes.to_csv(args.folder_path_to_workspace + "/training_shapes.csv")


########################
### ANALYSIS RESULTS ###
########################

# Load forecasting results.
xls = pd.ExcelFile(args.folder_path_to_workspace + "/forecast.xlsx")

forecast_splits = dict()
for split in xls.sheet_names[1:]:
    forecast_split = pd.read_excel(xls, split, index_col = 0, header = [0, 1, 2])
    # Reset the index.
    forecast_split.index = np.arange(1, len(forecast_split) + 1)
    forecast_split.index.names = ["Prediction horizon"]
    # Save the predictions.
    forecast_splits[split] = forecast_split

forecast_splits = pd.concat(forecast_splits, axis = 1)

def f_loss(x, level, func):
    # Model.
    model = func(x.xs(TARGET, axis = 1, level = level), x.xs("Forecast", axis = 1, level = level))
    # Naive.
    naive = func(x.xs(TARGET, axis = 1, level = level), x.xs("Naive", axis = 1, level = level))  
    return pd.Series([model, naive], index = ["mse_model", "mse_naive"])


loss_sites_dict = dict()
loss_h_dict = dict()
loss_overall_dict = dict()
# Compute mean squared error.
for split in forecast_splits.columns.get_level_values(0).unique():
    print(f"{split}")
    
    # Compute the total prediction loss for each site (province).
    loss_sites = forecast_splits[split].groupby(axis = 1, level = [0, 1]).apply(lambda x: f_loss(x, level = 2, func = mse)).transpose()
    loss_sites_dict[split] = loss_sites
    
    # Compute the loss as a function of the prediction horizon among all the sites (provinces).
    loss_h = forecast_splits[split].transpose().unstack(2).groupby(axis = 0, level = 0).apply(lambda x: x.groupby(axis = 1, level = 0).apply(lambda x: f_loss(x, level = 1, func = mse)).transpose())
    loss_h_dict[split] = loss_h
    
    # Define the overall loss.
    loss_overall = loss_sites.groupby(axis = 0, level = 0).mean().mean()
    loss_overall_dict[split] = loss_overall
    
    print("Overall loss:")
    print(loss_overall)
    print("Country loss:")
    print(loss_sites.groupby(axis = 0, level = 0).mean())

# Overall loss.
loss_overall = pd.concat(loss_overall_dict)
loss_overall = loss_overall.groupby(axis = 0, level = 1).mean()
loss_overall.to_csv(args.folder_path_to_workspace + "/overall_loss.txt", sep = "\t")
print("Loss overall: \n", loss_overall)

# Compute loss for each site for each split.
loss_sites = pd.concat(loss_sites_dict).unstack(0).reorder_levels([1, 0], axis = 1).sort_index(axis = 1, level = [0, 1])
loss_sites.columns.names = ["Split", "Type"]
loss_sites.to_csv(args.folder_path_to_workspace + "/loss_sites.csv")
# Compute loss for each prediction horizon at country level for each split.
loss_h = pd.concat(loss_h_dict).unstack(0).reorder_levels([1, 0], axis = 1).sort_index(axis = 1, level = [0, 1])
loss_h.columns.names = ["Split", "Type"]
loss_h.to_csv(args.folder_path_to_workspace + "/loss_h.csv")

# Compute r2 for each prediction horizon at country level for each split.
r2_results = dict()
def compute_r2(x):
    def func(row):
        actual = row.xs(TARGET, axis = 0, level = 3).values
        naive = row.xs("Naive", axis = 0, level = 3).values
        forecast = row.xs("Forecast", axis = 0, level = 3).values
        return pd.Series([r2(actual, forecast), r2(actual, naive)], index = ["r2_model", "r2_naive"])
    r2_results[x.name] = x.apply(func, axis = 1)

forecast_splits.groupby(axis = 1, level = [0, 1]).apply(compute_r2)
r2_results = pd.concat(r2_results).unstack([0, 1]).reorder_levels([2, 1, 0], axis = 1).sort_index(axis = 1, level = [0, 1, 2])
r2_results.columns.names = ["Country", "Split", "Type"]
r2_results.to_csv(args.folder_path_to_workspace + "/r2_results.csv")

