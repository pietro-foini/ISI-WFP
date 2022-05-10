import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import shutil
import pickle
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.early_stop import no_progress_loss
import xgboost as xgb
import itertools
import argparse
import click

from _gui import *
from _utils import *
from _default import *

###############################
### USER-DEFINED PARAMETERS ###
###############################

parser_user = argparse.ArgumentParser(description = "This file allows to perform the hyperparameter tuning on the xgboost parameters (the indicator selection as optional hyperparameter tuning) over the validation data of the corresponding splits using a bayesian approach.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

# Example usage: python hyperparameter_tuning.py --folder_path_to_dataset "./Nigeria/dataset" --folder_path_to_workspace "./Nigeria/out_hyper" --splits_to_consider 1 2 3 4 5 --fraction_train_set 0.8 --trial_steps 600 --gui_interface --n_jobs 6 --early_stop_hyperopt 200

parser_user.add_argument('--folder_path_to_dataset', type = str, default = "./dataset", help = "The path to the folder containing the dataset (training and test points).")
parser_user.add_argument('--folder_path_to_workspace', type = str, default = "./output_hyperparameter_tuning", help = "The path to the folder where all the results arising from the current hyperparameter tuning will be stored.")
parser_user.add_argument('--splits_to_consider', type = int, default = [1,2,3,4,5,6,7,8,9,10], nargs = "+", help = "Define on which splits perform the hyperpameter tuning.")
parser_user.add_argument('--fraction_train_set', type = float, default = 0.8, help = "Define the fraction of points to use for training for the time-series of each province. The remaining points (temporally last) are used for validation: the split is performed in a time-order way for each time-series.")
parser_user.add_argument('--start_date', type = str, default = None, help = "The start date of target time-serie with which build the training and validation points (and consequently test points). If None the entire time-series are considered.")
parser_user.add_argument('--n_provinces', type = int, default = -1, help = "Define the number of provinces to consider for each country in the current analysis. If the selected value is -1, all the provinces are considered for each country (standard approach). If the selected value is smaller than the number of provinces (N.B. the 'n_provinces' parameter have to be set equal or less than the number of provinces of each country), 10 different hyperparameter analysis are generated by subsampling 'n_provinces' provinces from each country with 10 different random seeds.")
parser_user.add_argument('--format', type = str, default = "feather", choices = ["csv", "feather", "xlsx"], help = "The file format to store training and validation points.")
parser_user.add_argument('--trial_steps', type = int, default = 600, help = "Define the number of hyeropt hyperparameter configurations to test for each prediction horizon and each split.")
parser_user.add_argument('--early_stop_hyperopt', type = int, default = 150, help = "Stop function that will stop after X iteration if the loss doesn't increase during hyperopt optimization.")
parser_user.add_argument('--n_jobs', type = int, default = 1, help = "Define the number of 'n_job' of the xgboost model.")
parser_user.add_argument('--gui_interface', action = "store_true", help = "If you want to select the time features and the lags for each indicator through a GUI interface otherwise the corresponding default values are taken (see *_default*).")
parser_user.add_argument('--gpu', action = "store_true", help = "If you want use gpu.")

args = parser_user.parse_args()

#################
### WORKSPACE ###
#################

CONTINUE = False

# Define the name of the hyperparameter tuning analysis.
if args.n_provinces == -1:
    SEEDS = ["standard"]
else:
    SEEDS = [0,1,2,3,4,5,6,7,8,9]

# Create the workspace folder where all the results arising from the current hyperparameter tuning will be stored.
if not os.path.exists(args.folder_path_to_workspace): 
    os.makedirs(args.folder_path_to_workspace) 
    for seed in SEEDS:
        os.makedirs(f"{args.folder_path_to_workspace}/{seed}") 
        os.makedirs(f"{args.folder_path_to_workspace}/{seed}/hyperopt") 
else:
    if not click.confirm("The workspace selected already exist. If you continue, the hyperparameter tuning will continue starting from the existing information in the folder otherwise a new session will start. Continue?", default = True):
        shutil.rmtree(args.folder_path_to_workspace) 
        os.makedirs(args.folder_path_to_workspace) 
        for seed in SEEDS:
            os.makedirs(f"{args.folder_path_to_workspace}/{seed}") 
            os.makedirs(f"{args.folder_path_to_workspace}/{seed}/hyperopt") 
    else:
        CONTINUE = True

# Load the values of some global variables defined during the creation of the dataset.
with open(f"{args.folder_path_to_dataset}/global_variables", "rb") as f:
    COUNTRIES, TARGET, TEST_SIZE, NUMBER_OF_SPLITS, FEATURES_TIME, FORMAT = pickle.load(f)
    
# Load the lags dictionary defined during the creation of the dataset.
with open(f"{args.folder_path_to_dataset}/lags_dict", "rb") as fp:
    LAGS_DICT = pickle.load(fp)

# Check if the splits defined by user are allowed with respect the dataset creation.
if not all(i <= NUMBER_OF_SPLITS for i in args.splits_to_consider):
    raise ValueError("Some selected splits are not allowed by the current dataset. Check 'splits_to_consider' parameter.") 
    
# Save start_date parameter into pickle file.
# N.B. The start date have to take into account the lag value corresponding to the target variable during dataset creation.
if args.start_date is not None:
    START_DATE = pd.to_datetime(args.start_date) + pd.DateOffset(days = len(LAGS_DICT[TARGET]) - 1)
else:
    START_DATE = None
# Save the starting date information.    
with open(f"{args.folder_path_to_workspace}/start_date", "wb") as f:
    pickle.dump([START_DATE], f)
    
# Get the 'n_provinces' provinces to consider for each seed.  
for seed in SEEDS:      
    provinces_considered = {}
    for country in COUNTRIES:
        provinces = os.listdir(f"{args.folder_path_to_dataset}/train/{country}/")
        if seed != "standard":
            np.random.seed(seed)            
            p = np.random.choice(provinces, args.n_provinces, replace = False)
        else:
            p = provinces.copy()  
        # Store provinces.
        provinces_considered[country] = p        
    # Save the provinces considered in the current seed.
    with open(f"{args.folder_path_to_workspace}/{seed}/provinces_considered", "wb") as fp:
        pickle.dump(provinces_considered, fp)
        
# Save argparse arguments of the current session.
with open(f"{args.folder_path_to_workspace}/commandline_args.txt", "w") as f:
    print(args.__dict__, file = f)
    
#####################
### GUI INTERFACE ###
#####################
    
# GUI interface (it allows to modify default time features and indicator lags).
if not CONTINUE:
    if args.gui_interface:
        interface = gui()
        out = interface.GUI_lags_2(FEATURES_TIME, LAGS_DICT, defaultTimes, defaultLags_2)
        # Check parameters.
        if out[1] is {}:
            raise ValueError("You have to set a lag value for at least one indicator.") 
    else:
        out = (defaultTimes.copy(), defaultLags_2.copy())

    # Add time features to lags dictionary.
    LAGS_DICT = out[1].copy()
    for feature in out[0]:
        LAGS_DICT[feature] = None
        
    # Save the lags dictionary.
    with open(f"{args.folder_path_to_workspace}/lags_dict", "wb") as fp:
        pickle.dump(LAGS_DICT, fp)
else:
    # Load the lags dictionary defined during in a previous hyperparameter tuning.
    with open(f"{args.folder_path_to_workspace}/lags_dict", "rb") as fp:
        LAGS_DICT = pickle.load(fp)

# Space of configurations: define the space of configurations to which perform the hyperparameter tuning. We define two spaces with different natures.
if not CONTINUE:
    # GUI interface (it allows to modify default feature selection).
    if args.gui_interface:
        # Model.
        SPACE_MODEL = defaultSpace_model.copy()
        # Indicators.
        interface = gui()
        space_indicators = interface.GUI_indicators_2(list(LAGS_DICT.keys()), TARGET)
        SPACE_INDICATORS = {k: hp.choice(k, [True, False]) if v else True for k,v in space_indicators.items()}
    else:
        SPACE_MODEL = defaultSpace_model.copy()
        SPACE_INDICATORS = defaultSpace_indicators.copy()

    # Merge the two dictionary to perform the hyperparameter tuning on both dictionaries.
    space = dict(SPACE_MODEL, **SPACE_INDICATORS)

    # Save the total set of parameters.
    with open(f"{args.folder_path_to_workspace}/space", "wb") as fp:
        pickle.dump(space, fp)
    # Save the first set of parameters.
    with open(f"{args.folder_path_to_workspace}/space_model", "wb") as fp:
        pickle.dump(SPACE_MODEL, fp)
    # Save the second set of parameters.
    with open(f"{args.folder_path_to_workspace}/space_indicators", "wb") as fp:
        pickle.dump(SPACE_INDICATORS, fp)
else:
    # Load the first set of parameters.
    with open(f"{args.folder_path_to_workspace}/space_model", "rb") as fp:
        SPACE_MODEL = pickle.load(fp)       
    # Load the second set of parameters.
    with open(f"{args.folder_path_to_workspace}/space_indicators", "rb") as fp:
        SPACE_INDICATORS = pickle.load(fp)

    # Merge the two dictionary to perform the hyperparameter tuning on both dictionaries.
    space = dict(SPACE_MODEL, **SPACE_INDICATORS) 
    
########################
### CUSTOM FUNCTIONS ###
########################

# Define function in order to load training data and consequently create validation data.
def load_train_validation(country, province, split_number, h):
    # Load data.
    X = load(f"{args.folder_path_to_dataset}/train/{country}/{province}/X_train_split_{split_number}_h_{h+1}", FORMAT) 
    y = load(f"{args.folder_path_to_dataset}/train/{country}/{province}/y_train_split_{split_number}_h_{h+1}", FORMAT) 

    if args.start_date is not None:
        try:
            t = X[[f"Year|x(t+{h+1})", f"Month|x(t+{h+1})", f"Day|x(t+{h+1})"]].copy()
            t.columns = ["year", "month", "day"]
        except:
            raise ValueError(f"Not enough temporal information if you select '{args.start_date}' as starting point.")

        s = np.flatnonzero(pd.to_datetime(t) > START_DATE + pd.DateOffset(days = h+1))

        X = X.loc[s].reset_index(drop = True)
        y = y.loc[s].reset_index(drop = True)

    # Train and validation.
    X_train, X_validation = X.iloc[:int(X.shape[0]*args.fraction_train_set),:], X.iloc[int(X.shape[0]*args.fraction_train_set):,:] 
    y_train, y_validation = y.iloc[:int(y.shape[0]*args.fraction_train_set),:], y.iloc[int(y.shape[0]*args.fraction_train_set):,:] 

    X_validation = X_validation.reset_index(drop = True)
    y_validation = y_validation.reset_index(drop = True) 

    return X_train, y_train, X_validation, y_validation
      
# Define the function that, given a space configuration of hyperparameters and a space of features, computes a validation loss to minimize. 
def objective(space, split_number, h, training, validation, seed): 
    # Split the space parameters based on two dictionaries: SPACE_MODEL and SPACE_INDICATORS.
    # Model parameters.
    sp_model = {k: v for k, v in space.items() if k in SPACE_MODEL.keys()}
    # Indicators parameters.
    sp_indicators = {k: v for k, v in space.items() if k in SPACE_INDICATORS.keys()}
    
    # Select indicators.
    # Decide the indicators to keep based on the current space.
    sp_indicators = [k for k, v in sp_indicators.items() if v]
    # Select all the corresponding lags.
    sp_indicators = {indicator: take_lags(indicator, lags = LAGS_DICT[indicator], h = h) for indicator in sp_indicators}
    # Flat dictionary values.
    indicators = list(itertools.chain(*list(sp_indicators.values())))

    # Get training and validation data.
    Xy_train = (training[0].copy(), training[1].copy())
    Xy_validation = validation.copy()
    # Training.
    X_train = Xy_train[0]
    X_train = X_train[indicators].sort_index(axis = 1)
    y_train = Xy_train[1]
    # Validation.
    for country in COUNTRIES:
        X_validation = Xy_validation[country][0]
        X_validation = X_validation[indicators].sort_index(axis = 1)
        y_validation = Xy_validation[country][1]
        Xy_validation[country] = (X_validation, y_validation)

    # Concatenate validation points among the countries.
    X_validation = pd.concat([Xy_validation[country][0] for country in COUNTRIES]).reset_index(drop = True)
    y_validation = pd.concat([Xy_validation[country][1] for country in COUNTRIES]).reset_index(drop = True)

    # Training.
    if args.gpu:
        model = xgb.XGBRegressor(**sp_model, objective = "reg:squarederror", tree_method = "gpu_hist", n_jobs = args.n_jobs)
    else:
        model = xgb.XGBRegressor(**sp_model, objective = "reg:squarederror", tree_method = "hist", n_jobs = args.n_jobs)
    
    if len(trials.trials) == 0:
        model.fit(X_train, y_train) 
        n_estimators = model.n_estimators
    else:
        model.fit(X_train, y_train, early_stopping_rounds = 50, eval_set = [(X_validation, y_validation)], 
                  eval_metric = "rmse", verbose = False) 
        # Retrieve the number of estimators stopped from early stopping.
        n_estimators = len(model.evals_result()["validation_0"]["rmse"])

    # Compute r2 train.
    r2_train = model.score(X_train, y_train)
    # Compute r2 train for each country.
    r2_validation_list = list()
    for country in COUNTRIES: 
        r2_validation = model.score(Xy_validation[country][0], Xy_validation[country][1])
        r2_validation_list.append(r2_validation)

    # Average of the r2 validation among the countries.
    r2_validation = np.mean(r2_validation_list)
    # Compute the r2 difference.
    r2_difference = np.abs(r2_train - r2_validation)
    
    # Define the loss to minimize during the hyperparameter tuning. We create some penalty if the r2 scores are negatives and if the r2 train is too low.
    if r2_train < 0 or r2_validation < 0:
        penalty = np.inf
    else:
        penalty = (1-r2_train)

    loss_to_minimize = (r2_difference*0.7 + penalty*0.3)

    # Save results.
    results = space.copy()
    results["n_estimators"] = n_estimators
    results["attempt"] = len(trials.trials)
    results["h"] = h
    results["split"] = split_number
    results["r2_train"] = r2_train
    results["r2_val"] = r2_validation
    results["shape_train"] = [X_train.shape]
    results["shape_val"] = [X_validation.shape]
    results["r2_difference"] = r2_difference
    results["loss_to_minimize"] = loss_to_minimize
    results = pd.DataFrame(results, index = [0], dtype = object)
    filename = f"{args.folder_path_to_workspace}/{seed}/hyperparameter_tuning.csv"
    results.to_csv(filename, index = False, header = (not os.path.exists(filename)), mode = "a")

    return {"loss": loss_to_minimize, "status": STATUS_OK}

###############################
### TRAINING AND VALIDATION ###
###############################

# Define the loop for the hyperparameter tuning.
for seed in SEEDS:
    for split_number in args.splits_to_consider:
        for h in range(TEST_SIZE):
            print(f"Seed {seed} -- Split {split_number} -- Prediction horizon {h+1}")

            # Provinces considered from the current seed.
            with open(f"{args.folder_path_to_workspace}/{seed}/provinces_considered", "rb") as fp:
                provinces_considered = pickle.load(fp) 
            
            # Load data.
            Xy_train_list = list()
            Xy_validation_dict = {country: list() for country in COUNTRIES}
            # Training and validation.
            for country in COUNTRIES:
                provinces = provinces_considered[country]
                for province in provinces:
                    X_train, y_train, X_validation, y_validation = load_train_validation(country, province, split_number, h)
                    #  Store data.
                    Xy_train_list.append((X_train, y_train))
                    Xy_validation_dict[country].append((X_validation, y_validation))
                
            # Concatenate training points among the countries and provinces.
            X_train = pd.concat([x[0] for x in Xy_train_list]).reset_index(drop = True)
            y_train = pd.concat([x[1] for x in Xy_train_list]).reset_index(drop = True)
            
            # Concatenate validation points among the provinces but separating for country.
            for country in COUNTRIES:
                Xy_validation_dict[country] = (pd.concat([x[0] for x in Xy_validation_dict[country]]).reset_index(drop = True),
                                               pd.concat([x[1] for x in Xy_validation_dict[country]]).reset_index(drop = True))

            # Try to load an already saved trials object if exist, and continue minimization increasing the max.
            try:  
                trials = pickle.load(open(f"{args.folder_path_to_workspace}/{seed}/hyperopt/hyp_trials_split_{split_number}_h_{h+1}.p", "rb"))
                max_trials = len(trials.trials) + args.trial_steps

                print("Found saved Trials! Loading...")
                print(f"Rerunning from {len(trials.trials)} trials to {max_trials} (+{args.trial_steps}) trials")

                # Hyperopt optimization.
                best = fmin(fn = lambda x: objective(x, split_number, h+1, (X_train, y_train), Xy_validation_dict, seed),
                            space = space,
                            algo = tpe.suggest,
                            max_evals = max_trials,
                            trials = trials, 
                            early_stop_fn = no_progress_loss(iteration_stop_count = args.early_stop_hyperopt),
                            rstate = np.random.RandomState(42))

                # Save the hyperopt trials into a file.
                pickle.dump(trials, open(f"{args.folder_path_to_workspace}/{seed}/hyperopt/hyp_trials_split_{split_number}_h_{h+1}.p", "wb"))
            # Create a new trials object starting a new hyperopt optimization.
            except: 
                # STARTING CONFIGURARION 1.
                # Use the default parameters of the model as starting configuration (see *_default.py*).
                starting_point_space_model = defaultStarting_point_space_model.copy()
                # Use only the target indicator as starting configuration (a.k.a univariate forecasting).
                starting_point_space_features = {k: True if k is TARGET else False for k,v in LAGS_DICT.items()}            
                # Merge two dictionaries.
                starting_point_space_1 = dict(starting_point_space_model, **starting_point_space_features)

                try:
                    # STARTING CONFIGURARION 2.
                    # Try to load the space parameters that obtained the best result on the previous split.
                    results_hyp = pd.read_csv(f"{args.folder_path_to_workspace}/{seed}/hyperparameter_tuning.csv")
                    best_parameters_previous = results_hyp[results_hyp["split"] == split_number-1].groupby("h").apply(lambda x: x.loc[x["loss_to_minimize"].idxmin()]).loc[h+1]
                    # Use this configuration as another starting configuration for the current split.
                    starting_point_space_2 = dict(best_parameters_previous[starting_point_space.keys()])
                    trials = generate_trials_to_calculate([starting_point_space_1, starting_point_space_2])
                except:
                    trials = generate_trials_to_calculate([starting_point_space_1])

                # Hyperopt optimization.
                max_trials = args.trial_steps
                best = fmin(fn = lambda x: objective(x, split_number, h+1, (X_train, y_train), Xy_validation_dict, seed),
                            space = space,
                            algo = tpe.suggest,
                            max_evals = max_trials,
                            trials = trials, 
                            early_stop_fn = no_progress_loss(iteration_stop_count = args.early_stop_hyperopt),
                            rstate = np.random.RandomState(42))

                # Save the hyperopt trials into a file.
                pickle.dump(trials, open(f"{args.folder_path_to_workspace}/{seed}/hyperopt/hyp_trials_split_{split_number}_h_{h+1}.p", "wb"))
