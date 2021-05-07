import numpy as np
import pandas as pd
import os
import shutil
import pickle
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate
import xgboost as xgb
import itertools
import argparse
import glob
import ntpath
import click
import sys
from _gui import *
from _utils import *
from _default import *

#rstate = np.random.RandomState(123) # Put it into 'fmin' of hyperopt.

###############################
### USER-DEFINED PARAMETERS ###
###############################

parser_user = argparse.ArgumentParser(description = "This file allows to perform the hyperpameter tuning (and feature selection) over the validation sets of the corresponding splits using a bayesian approach.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser_user.add_argument('--folder_path_to_dataset', type = str, default = "./dataset", help = "The path to the folder containing the dataset (training and test points).")
parser_user.add_argument('--folder_path_to_workspace', type = str, default = "./output_hyperparameter_tuning", help = "The path to the folder where all the results arising from the current hyperparameter tuning will be stored.")
parser_user.add_argument('--splits_to_consider', type = int, default = [1,2,3,4,5,6,7,8,9,10], nargs = "+", help = "Define on which splits perform the hyperpameter tuning.")
parser_user.add_argument('--fraction_train_set', type = float, default = 0.9, help = "Define the fraction of points to use for training for the time-series of each province. The remaining points are used for validation. The split is performed in a time-order way for each time-series.")
parser_user.add_argument('--format', type = str, default = "feather", choices = ["csv", "feather", "xlsx"], help = "The file format to store training and validation points.")
parser_user.add_argument('--trial_steps', type = int, default = 300, help = "Define the number of hyeropt hyperparameter configurations to try for each prediction horizon of each split.")
parser_user.add_argument('--n_jobs', type = int, default = 1, help = "Define the number of 'n_job' of the xgboost model.")
parser_user.add_argument('--gui_interface', action = "store_true", help = "If you want to select the time features and the lags for each indicator through a GUI interface otherwise the corresponding default values are taken (see *_default*).")
parser_user.add_argument('--gpu', action = "store_true", help = "If you want use gpu.")

args = parser_user.parse_args()

#################
### WORKSPACE ###
#################

CONTINUE = False

# Create the workspace folder where all the results arising from the current hyperparameter tuning will be stored.
if not os.path.exists(args.folder_path_to_workspace): 
    os.makedirs(args.folder_path_to_workspace) # Create folder.
    os.makedirs(args.folder_path_to_workspace + "/hyperopt") # Create subfolder for the hyperopt checkpoints.
else:
    if not click.confirm("The workspace selected already exist. If you continue, the hyperparameter tuning will continue starting from the existing information in the folder. Continue?", default = True):
        exit()
    else:
        CONTINUE = True

# Load the values of some global variables defined during the creation of the dataset.
with open(args.folder_path_to_dataset + "/global_variables", "rb") as f:
    COUNTRIES, TARGET, TEST_SIZE, NUMBER_OF_SPLITS, FEATURES_TIME, FORMAT, _ = pickle.load(f)
    
# Load the lags dictionary defined during the creation of the dataset.
with open(args.folder_path_to_dataset + "/lags_dict", "rb") as fp:
    LAGS_DICT = pickle.load(fp)

# Check if the splits defined by user are allowed with respect the dataset.
if not all(i <= NUMBER_OF_SPLITS for i in args.splits_to_consider):
    raise ValueError("Some selected splits are not allowed by the current dataset. Check 'splits_to_consider' parameter.")   
    
# Save argparse arguments of the current session.
with open(args.folder_path_to_workspace + "/commandline_args.txt", "w") as f:
    f.write("\n".join(sys.argv[1:]))
    
# Creation of training and validation points
dir_data_hyper = args.folder_path_to_workspace + "/dataset_hyper"
if not os.path.exists(dir_data_hyper):
    print("Create training and validation. Please wait.")
    # Create folder for containing data points.
    os.makedirs(dir_data_hyper)
    # Create folder for containing training data.
    os.makedirs(dir_data_hyper + "/train")
    # Create folder for containing validation data.
    os.makedirs(dir_data_hyper + "/validation")
    for country in COUNTRIES:
        os.makedirs(dir_data_hyper + f"/validation/{country}") 
        
    for split_number in args.splits_to_consider:
        for h in range(TEST_SIZE):
            # Get training and validation data.
            Xy_train_list = list()
            Xy_validation_dict = {country: list() for country in COUNTRIES}
            for country in COUNTRIES:
                provinces = glob.glob(args.folder_path_to_dataset + f"/train/{country}/*")
                for province in provinces:
                    province = ntpath.basename(province).split(".")[0]

                    # Load training data.
                    X = load(args.folder_path_to_dataset + f"/train/{country}/{province}/X_train_split_{split_number}_h_{h+1}",
                                   FORMAT) 
                    y = load(args.folder_path_to_dataset + f"/train/{country}/{province}/y_train_split_{split_number}_h_{h+1}",
                                   FORMAT) 

                    # Train and validation.
                    X_train, X_validation = X.iloc[:int(X.shape[0]*args.fraction_train_set),:], X.iloc[int(X.shape[0]*args.fraction_train_set):,:] 
                    y_train, y_validation = y.iloc[:int(y.shape[0]*args.fraction_train_set),:], y.iloc[int(y.shape[0]*args.fraction_train_set):,:] 

                    X_validation = X_validation.reset_index(drop = True)
                    y_validation = y_validation.reset_index(drop = True) 

                    # Store data.
                    Xy_train_list.append((X_train, y_train))
                    Xy_validation_dict[country].append((X_validation, y_validation))

            # Concatenate training points among the countries and provinces.
            X_train = pd.concat([x[0] for x in Xy_train_list]).reset_index(drop = True)
            save(X_train, dir_data_hyper + f"/train/X_train_split_{split_number}_h_{h+1}", args.format)
            y_train = pd.concat([x[1] for x in Xy_train_list]).reset_index(drop = True)
            save(y_train, dir_data_hyper + f"/train/y_train_split_{split_number}_h_{h+1}", args.format)
            
            # Concatenate validation points among the provinces but separating for country.
            for country in COUNTRIES:
                Xy_validation_dict[country] = (pd.concat([x[0] for x in Xy_validation_dict[country]]).reset_index(drop = True), pd.concat([x[1] for x in Xy_validation_dict[country]]).reset_index(drop = True))
                save(Xy_validation_dict[country][0], dir_data_hyper + f"/validation/{country}/X_validation_split_{split_number}_h_{h+1}", args.format)
                save(Xy_validation_dict[country][1], dir_data_hyper + f"/validation/{country}/y_validation_split_{split_number}_h_{h+1}", args.format)
                
    print("Complete!")
    
# GUI interface (it allows to modify default time features and lags).
if not CONTINUE:
    if args.gui_interface:
        interface = gui()
        out = interface.GUI2(FEATURES_TIME, LAGS_DICT, defaultTimes2, defaultLags2)
        # Check parameters.
        if out[1] is {}:
            raise ValueError("You have to set a lag value for at least one indicator.") 
    else:
        out = (defaultTimes2.copy(), defaultLags2.copy())

    # Add time features to lags dictionary.
    lags_dict = out[1].copy()
    for feature in out[0]:
        lags_dict[feature] = None
        
    # Save the lags dictionary.
    with open(args.folder_path_to_workspace + "/lags_dict", "wb") as fp:
        pickle.dump(lags_dict, fp)
    # Save the features name considered during this hyperparameter search.
    with open(args.folder_path_to_workspace + "/features.txt", "w") as output:
        for feature in lags_dict.keys():
            output.write(str(feature) + "\n")
else:
    # Load the lags dictionary defined during in a previous hyperparameter tuning.
    with open(args.folder_path_to_workspace + "/lags_dict", "rb") as fp:
        lags_dict = pickle.load(fp)

# Space of configurations: define the space of configurations to which perform the hyperparameter tuning. We define two spaces with different natures.

if not CONTINUE:
    # GUI interface (it allows to modify default feature selection).
    if args.gui_interface:
        space1 = defaultSpace1.copy()
        interface = gui()
        space2 = interface.GUI3(list(lags_dict.keys()), TARGET)
        space2 = {k: hp.quniform(k, 0, 1, 1) if v else 1. for k,v in space2.items()}
    else:
        space1 = defaultSpace1.copy()
        space2 = defaultSpace2.copy()

    # Merge the two dictionary to perform the hyperparameter tuning on both dictionaries.
    space = dict(space1, **space2)

    # Save the parameter names of the total set of parameters.
    with open(args.folder_path_to_workspace + "/space", "wb") as fp:
        pickle.dump(space, fp)

    # Save the parameter names of the first set of parameters.
    with open(args.folder_path_to_workspace + "/space1", "wb") as fp:
        pickle.dump(space1, fp)

    # Save the parameter names of the second set of parameters.
    with open(args.folder_path_to_workspace + "/space2", "wb") as fp:
        pickle.dump(space2, fp)
else:
    # Load the parameter names of the first set of parameters.
    with open(args.folder_path_to_workspace + "/space1", "rb") as fp:
        space1 = pickle.load(fp)
        
    # Load the parameter names of the second set of parameters.
    with open(args.folder_path_to_workspace + "/space2", "rb") as fp:
        space2 = pickle.load(fp)

    # Merge the two dictionary to perform the hyperparameter tuning on both dictionaries.
    space = dict(space1, **space2)
    
############
### MAIN ###
############
      
# Define the function that, given a space configuration of hyperparameters and a space of features, computes a custom loss to minimize.   
def hyperparameters(space, split_number, h, training, validation): 
    # The float parameters that are integer are converted to be int type (avoid future issues).
    space = {k: int(v) if v.is_integer() else v for k,v in space.items()}
    
    # Split the space parameters based on two dictionaries: space1 and space2.
    # Model parameters (space1).
    space_model = {k: v for k,v in space.items() if k in space1.keys()}
    # Features parameters (space2).
    space_features = {k: v for k,v in space.items() if k in space2.keys()}
    
    # Select features.
    # Decide the features to keep based on values (0 or 1) of the current space.
    space_features = [k for k,v in space2.items() if v == 1]
    # Select all the corresponding lags.
    space_features = {feature: take_lags(feature, lags = lags_dict[feature], h = h) for feature in space_features}
    # Flat dictionary values.
    features = list(itertools.chain(*list(space_features.values())))
    
    # Get training and validation data.
    Xy_train = (training[0].copy(), training[1].copy())
    Xy_validation = validation.copy()
    # Training.
    X_train = Xy_train[0]
    X_train = X_train[features].sort_index(axis = 1)
    y_train = Xy_train[1]
    # Validation.
    for country in COUNTRIES:
        X_validation = Xy_validation[country][0]
        X_validation = X_validation[features].sort_index(axis = 1)
        y_validation = Xy_validation[country][1]
        Xy_validation[country] = (X_validation, y_validation)

    # Concatenate validation points among the countries.
    X_validation = pd.concat([Xy_validation[country][0] for country in COUNTRIES]).reset_index(drop = True)
    y_validation = pd.concat([Xy_validation[country][1] for country in COUNTRIES]).reset_index(drop = True)

    # Training.
    if args.gpu:
        model = xgb.XGBRegressor(**space_model, objective = "reg:squarederror", tree_method = "gpu_hist", n_jobs = args.n_jobs)
    else:
        model = xgb.XGBRegressor(**space_model, objective = "reg:squarederror", tree_method = "hist", n_jobs = args.n_jobs)
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
    filename = args.folder_path_to_workspace + "/hyperparameter_tuning.csv"
    results.to_csv(filename, index = False, header = (not os.path.exists(filename)), mode = "a")

    return {"loss": loss_to_minimize, "status": STATUS_OK}

# Define the loop for the hyperparameter tuning.
for split_number in args.splits_to_consider:
    for h in range(TEST_SIZE):
        print(f"Split {split_number} --- Prediction horizon {h+1}")
        
        # Load data.
        # Training.
        X_train = load(dir_data_hyper + f"/train/X_train_split_{split_number}_h_{h+1}", args.format)
        y_train = load(dir_data_hyper + f"/train/y_train_split_{split_number}_h_{h+1}", args.format)
        # Validation.
        Xy_validation_dict = dict()
        for country in COUNTRIES:
            X_validation = load(dir_data_hyper + f"/validation/{country}/X_validation_split_{split_number}_h_{h+1}", args.format) 
            y_validation = load(dir_data_hyper + f"/validation/{country}/y_validation_split_{split_number}_h_{h+1}", args.format) 
            Xy_validation_dict[country] = (X_validation, y_validation)
 
        # Try to load an already saved trials object if exist, and continue minimization increasing the max.
        try:  
            trials = pickle.load(open(args.folder_path_to_workspace + f"/hyperopt/hyp_trials_split_{split_number}_h_{h+1}.p", "rb"))
            max_trials = len(trials.trials) + args.trial_steps

            print("Found saved Trials! Loading...")
            print(f"Rerunning from {len(trials.trials)} trials to {max_trials} (+{args.trial_steps}) trials")

            # Hyperopt optimization.
            best = fmin(fn = lambda x: hyperparameters(x, split_number, h+1, (X_train, y_train), Xy_validation_dict),
                        space = space,
                        algo = tpe.suggest,
                        max_evals = max_trials,
                        trials = trials)

            # Save the hyperopt trials into a file.
            pickle.dump(trials, open(args.folder_path_to_workspace + f"/hyperopt/hyp_trials_split_{split_number}_h_{h+1}.p", "wb"))
        # Create a new trials object starting a new hyperopt optimization.
        except:  
            # Use the default parameters of the model as starting configuration.
            starting_point_space1 = defaultStarting_point_space1.copy()
            # Use only the target feature as starting configuration.
            starting_point_space2 = {k: 1. if k is TARGET else 0. for k,v in lags_dict.items()}
            
            # Merge the two dictionary to perform the hyperparameter tuning on both dictionaries.
            starting_point_space = dict(starting_point_space1, **starting_point_space2)

            # Try to load the space parameters that obtained the best result on the previous split.
            try:
                results_hyp = pd.read_csv(args.folder_path_to_workspace + "/hyperparameter_tuning.csv")
                best_parameters_previous = results_hyp[results_hyp["split"] == split_number-1].groupby("h").apply(lambda x: x.loc[x["loss_to_minimize"].idxmin()]).loc[h+1]
                # Use this configuration as another starting configuration for the current split.
                starting_point_space_2 = dict(best_parameters_previous[starting_point_space.keys()])
                trials = generate_trials_to_calculate([starting_point_space, starting_point_space_2])
            except:
                trials = generate_trials_to_calculate([starting_point_space])

            # Hyperopt optimization.
            max_trials = args.trial_steps
            best = fmin(fn = lambda x: hyperparameters(x, split_number, h+1, (X_train, y_train), Xy_validation_dict),
                        space = space,
                        algo = tpe.suggest,
                        max_evals = max_trials,
                        trials = trials)

            # Save the hyperopt trials into a file.
            pickle.dump(trials, open(args.folder_path_to_workspace + f"/hyperopt/hyp_trials_split_{split_number}_h_{h+1}.p", "wb"))
