# Hyperparameter tuning.

import numpy as np
import pandas as pd
import os
import shutil
import pickle
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse
import itertools

# Create the workspace folder where all the results arising from the current hyperparameter tuning will be stored.
dir = "./output_hyperparameter_tuning"
if not os.path.exists(dir): # N.B. If the folder already exists, the hyperparameter tuning will recursively continue from the existing results.
    os.makedirs(dir) # Create main folder.
    os.makedirs(dir + "/hyperopt") # Create folder containing the hyperopt checkpoints.

# Define the main folder containing the training and test data.
dir_data = "./data_xgboost"


# ## Time-series dataset

# Load the time-series data of the Yemen country.
df_yemen = pd.read_csv("../../Dataset time-series/output_data/Yemen/Yemen.csv", header = [0, 1], index_col = 0)
df_yemen.index.name = "Datetime"
df_yemen.index = pd.to_datetime(df_yemen.index)
freq = "D"
df_yemen.index.freq = freq
df_yemen.columns = pd.MultiIndex.from_tuples(map(lambda x: ("Yemen", x[0], x[1]), df_yemen.columns), names = ["Country", "AdminStrata", "Indicator"])

df_yemen.drop("Exchange rate", axis = 1, level = 2, inplace = True)

# Load the time-series data of the Nigeria country.
df_nigeria = pd.read_csv("../../Dataset time-series/output_data/Nigeria/Nigeria.csv", header = [0, 1], index_col = 0)
df_nigeria.index.name = "Datetime"
df_nigeria.index = pd.to_datetime(df_nigeria.index)
freq = "D"
df_nigeria.index.freq = freq
df_nigeria.columns = pd.MultiIndex.from_tuples(map(lambda x: ("Nigeria", x[0], x[1]), df_nigeria.columns), names = ["Country", "AdminStrata", "Indicator"])

df = pd.concat([df_yemen, df_nigeria], axis = 1)
# Consider the following dates.
df = df.loc["2018-01-01":"2020-08-31"]


# ## Global variables

# Select the countries to consider for the prediction analysis.
COUNTRIES_TO_CONSIDER = ["Yemen", "Nigeria"]
# Select countries.
df = df[COUNTRIES_TO_CONSIDER]

# Define some global variables.
# Define the number of days we want to learn to predict.
TEST_SIZE = 30
# Define the fraction of points to use for training. The remaining points are used for validation.
FRACTION_TRAIN_SET = 0.9

# Save the global variables.
with open(dir + "/global_variables", "wb") as f:
    pickle.dump([COUNTRIES_TO_CONSIDER, TEST_SIZE, FRACTION_TRAIN_SET], f)


# ## Lags variables

def take_lags(x, lags = None, delay = False):
    if lags is not None:
        lags = [(x, "x(t)") if i == 1 else (x, "x(t-%d)" % (i-1)) for i in lags]
        if delay:
            lags.append((x, "delay"))
    else:
        lags = [(x, slice(None))]
    return lags

# Define lags for each feature.
lags_dict = {"3 Months Anomaly Rainfalls (%)": take_lags("3 Months Anomaly Rainfalls (%)", lags = np.array([1]), delay = False), 
             "1 Month Anomaly Rainfalls (%)": take_lags("1 Month Anomaly Rainfalls (%)", lags = np.array([1]), delay = False), 
             "Rainfalls (mm)": take_lags("Rainfalls (mm)", lags = np.array([1]), delay = False), 
             "Price cereals and tubers": take_lags("Price cereals and tubers", lags = np.array([1]), delay = False), 
             "NDVI Anomaly": take_lags("NDVI Anomaly", lags = np.array([1]), delay = False),
             "Fatalities": take_lags("Fatalities", lags = np.array([1,2,3]), delay = False),
             "FCG": take_lags("FCG", lags = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14]), delay = False), 
             "rCSI": take_lags("rCSI", lags = np.array([1,2,3]), delay = False), 
             "Population": take_lags("Population", lags = np.array([1]), delay = False), 
             "Ramadan": take_lags("Ramadan", lags = np.array([1]), delay = False), 
             "Code": take_lags("Code", lags = np.array([1]), delay = False), 
             "Lat": take_lags("Lat", lags = np.array([1]), delay = False), 
             "Lon": take_lags("Lon", lags = np.array([1]), delay = False), 
             "Day": take_lags("Day"),
             "Month": take_lags("Month"), 
             "Year": take_lags("Year")}

# Save the lags dictionary.
with open(dir + "/lags_dict", "wb") as fp:
    pickle.dump(lags_dict, fp)


# ## Space of configurations

# Define the space of configurations to which perform the hyperparameter tuning. We define two spaces with different natures.
# Define the parameters of the XGBoost model to which perform the hyperparameter tuning.
space1 = {"gamma": hp.quniform("gamma", 0.0, 1., 0.005),
          "n_estimators": 1000.,
          "reg_alpha": hp.quniform("reg_alpha", 0.0, 1., 0.005),
          "reg_lambda": hp.quniform("reg_lambda", 0.0, 1., 0.005),
          "max_depth": hp.quniform("max_depth", 3, 6, 1), 
          "min_child_weight": hp.quniform("min_child_weight", 1., 10., 0.1), 
          "learning_rate": hp.quniform("learning_rate", 0.001, 0.3, 0.001), 
          "subsample": hp.quniform("subsample", 0.6, 1.0, 0.05),
          "colsample_bytree": hp.quniform("colsample_bytree", 0.6, 1.0, 0.05)}

# Define the parameters regarding the features to keep during the hyperparameter tuning: 0 not considers the feature, 1 considers the feature.
space2 = {"3 Months Anomaly Rainfalls (%)": hp.quniform("3 Months Anomaly Rainfalls (%)", 0, 1, 1), 
          "1 Month Anomaly Rainfalls (%)": hp.quniform("1 Month Anomaly Rainfalls (%)", 0, 1, 1), 
          "Rainfalls (mm)": hp.quniform("Rainfalls (mm)", 0, 1, 1),
          "Price cereals and tubers": hp.quniform("Price cereals and tubers", 0, 1, 1),
          "Fatalities": hp.quniform("Fatalities", 0, 1, 1),
          "NDVI Anomaly": hp.quniform("NDVI Anomaly", 0, 1, 1),
          "rCSI": hp.quniform("rCSI", 0, 1, 1),
          "Ramadan": 1.,
          "FCG": 1.,
          "Population": 1.,
          "Code": 1., 
          "Lat": 1., 
          "Lon": 1., 
          "Day": 1.,
          "Month": 1., 
          "Year": 1.}

# Merge the two dictionary to perform the hyperparameter tuning on both dictionaries.
space = dict(space1, **space2)

# Save the parameter names of the total set of parameters.
with open(dir + "/space", "wb") as fp:
    pickle.dump(list(space.keys()), fp)

# Save the parameter names of the first set of parameters.
with open(dir + "/space1", "wb") as fp:
    pickle.dump(list(space1.keys()), fp)

# Save the parameter names of the second set of parameters.
with open(dir + "/space2", "wb") as fp:
    pickle.dump(list(space2.keys()), fp)


# ## Creation of training and validation points

SPLITS_TO_CONSIDER = [1,2,3]

dir_data_hyper = dir + "/data_hyper"
if not os.path.exists(dir_data_hyper):
    print("Create training and validation. Please wait.")
    # Create folder for containing data points.
    os.makedirs(dir_data_hyper)
    # Create folder for containing training data.
    os.makedirs(dir_data_hyper + "/train")
    # Create folder for containing validation data.
    os.makedirs(dir_data_hyper + "/validation")
    for country in COUNTRIES_TO_CONSIDER:
        os.makedirs(dir_data_hyper + "/validation/%s" % country) 
        
    for split_number in SPLITS_TO_CONSIDER:
        for h in range(TEST_SIZE):
            # Get training and validation data.
            Xy_train_list = list()
            Xy_validation_dict = {country: list() for country in COUNTRIES_TO_CONSIDER}
            for country in COUNTRIES_TO_CONSIDER:
                provinces = df[country].columns.get_level_values(0).unique()
                for province in provinces:
                    X = pd.read_csv(dir_data + "/train/%s/%s/X_train_split%d_h%d.csv" % (country, province, split_number, h+1), header = [0, 1], index_col = 0) 
                    y = pd.read_csv(dir_data + "/train/%s/%s/y_train_split%d_h%d.csv" % (country, province, split_number, h+1), header = [0, 1], index_col = 0) 
                    # Train and validation.
                    X_train, X_validation = X.loc[:int(X.shape[0]*FRACTION_TRAIN_SET),:], X.loc[int(X.shape[0]*FRACTION_TRAIN_SET):,:] 
                    y_train, y_validation = y.loc[:int(y.shape[0]*FRACTION_TRAIN_SET),:], y.loc[int(y.shape[0]*FRACTION_TRAIN_SET):,:] 

                    X_validation = X_validation.reset_index(drop = True)
                    y_validation = y_validation.reset_index(drop = True) 

                    # Store data.
                    Xy_train_list.append((X_train, y_train))
                    Xy_validation_dict[country].append((X_validation, y_validation))

            # Concatenate training points among the countries and provinces.
            X_train = pd.concat([x[0] for x in Xy_train_list]).reset_index(drop = True) 
            X_train.to_csv(dir_data_hyper + "/train/X_train_split%d_h%d.csv" % (split_number, h+1), index_label = False)
            y_train = pd.concat([x[1] for x in Xy_train_list]).reset_index(drop = True)
            y_train.to_csv(dir_data_hyper + "/train/y_train_split%d_h%d.csv" % (split_number, h+1), index_label = False)
            # Concatenate training points among the provinces.
            for country in COUNTRIES_TO_CONSIDER:
                Xy_validation_dict[country] = (pd.concat([x[0] for x in Xy_validation_dict[country]]).reset_index(drop = True), pd.concat([x[1] for x in Xy_validation_dict[country]]).reset_index(drop = True))
                Xy_validation_dict[country][0].to_csv(dir_data_hyper + "/validation/%s/X_validation_split%d_h%d.csv" % (country, split_number, h+1), index_label = False)
                Xy_validation_dict[country][1].to_csv(dir_data_hyper + "/validation/%s/y_validation_split%d_h%d.csv" % (country, split_number, h+1), index_label = False)
                
    print("Complete!")


# ## Hyperopt

# Define the function that, given a space configuration of hyperparameters, computes a custom loss to minimize by the hyperparameter tuning.   
def hyperparameters(space, split_number, h, training, validation): 
    # The float parameters that are integer are converted to be int type.
    space = {k: int(v) if v.is_integer() else v for k,v in space.items()}
    # Split the space paramters based on two dictionaries: space1 and space2.
    # Model parameters.
    space_model = {k: v for k,v in space.items() if k in space1.keys()}
    # Features parameters.
    space_features = {k: v for k,v in space.items() if k in space2.keys()}
    
    # Select features.
    # Decide the indicators to keep based on values (0 or 1).
    space_features = {k: v for k,v in space_features.items() if v == 1}
    # Select the corresponding lags.
    space_features = {feature: lags_dict[feature] for feature in space_features.keys()}
    # Flatten list.
    features = list(itertools.chain(*list(space_features.values())))

    Xy_train = (training[0].copy(), training[1].copy())
    Xy_validation = validation.copy()

    # Get training and validation data.
    # Training.
    X_train = Xy_train[0]
    X_train = pd.concat([X_train.loc[:, feature] for feature in features], axis = 1).sort_index(axis = 1)
    y_train = Xy_train[1]

    # Validation.
    for country in COUNTRIES_TO_CONSIDER:
        X_validation = Xy_validation[country][0]
        X_validation = pd.concat([X_validation.loc[:, feature] for feature in features], axis = 1).sort_index(axis = 1)
        y_validation = Xy_validation[country][1]
        Xy_validation[country] = (X_validation, y_validation)

    # Concatenate training points among the countries.
    X_validation = pd.concat([Xy_validation[country][0] for country in COUNTRIES_TO_CONSIDER]).reset_index(drop = True)
    y_validation = pd.concat([Xy_validation[country][1] for country in COUNTRIES_TO_CONSIDER]).reset_index(drop = True)

    # Training.
    model = xgb.XGBRegressor(**space_model, objective = "reg:squarederror", tree_method = "hist", n_jobs = 3)
    if len(trials.trials) == 0:
        model.fit(X_train, y_train) 
        n_estimators = model.n_estimators
    else:
        model.fit(X_train, y_train, early_stopping_rounds = 50, eval_set = [(X_validation, y_validation)], 
                  eval_metric = "rmse", verbose = False) 
        # Retrieve the number of estimators stopped from early stopping.
        n_estimators = len(model.evals_result()["validation_0"]["rmse"])

    # Evaluation training set.
    y_hats_train = model.predict(X_train)
    # Loss.
    train_loss = mse(y_train.values.flatten(), y_hats_train)
    # r2 train.
    r2_train = model.score(X_train, y_train)
    # Evaluation validation set.
    r2_validation_list = list()
    validation_loss_list = list()
    for i,country in enumerate(COUNTRIES_TO_CONSIDER): 
        y_hats_validation = model.predict(Xy_validation[country][0])
        # Loss.
        validation_loss = mse(Xy_validation[country][1].values.flatten(), y_hats_validation)
        # r2 validation.
        r2_validation = model.score(Xy_validation[country][0], Xy_validation[country][1])
        r2_validation_list.append(r2_validation)
        validation_loss_list.append(validation_loss)

    validation_loss = np.mean(validation_loss_list)
    # Average of the r2 validation among the countries.
    r2_validation = np.mean(r2_validation_list)
    # r2 difference.
    r2_difference = np.abs(r2_train - r2_validation)
    
    # Define the loss to minimize during the hyperparameter tuning. I create some penalty if the r2 scores are negatives and if the r2 train is too low.
    if r2_train < 0:
        penalty = np.inf
    elif r2_validation < 0:
        penalty = np.inf
    else:
        penalty = (1-r2_train)

    loss_to_minimize = (r2_difference*0.7 + penalty*0.3)/1

    # Save results.
    results = space.copy()
    results["n_estimators"] = n_estimators
    results["attempt"] = len(trials.trials)
    results["h"] = h
    results["split"] = split_number
    results["train_loss"] = train_loss
    results["val_loss"] = validation_loss
    results["r2_train"] = r2_train
    results["r2_val"] = r2_validation
    results["shape_train"] = [X_train.shape]
    results["shape_val"] = [X_validation.shape]
    results["r2_difference"] = r2_difference
    results["loss_to_minimize"] = loss_to_minimize
    results = pd.DataFrame(results, index = [0], dtype = object)
    filename = dir + "/hyperparameter_tuning.csv"
    results.to_csv(filename, index = False, header = (not os.path.exists(filename)), mode = "a")

    return {"loss": loss_to_minimize, "status": STATUS_OK}


# Define the loop of the hyperparameter tuning.
# Define the number of configurations to try for each prediction horizon of each split.
TRIALS_STEPS = 300

for split_number in SPLITS_TO_CONSIDER:
    for h in range(TEST_SIZE):
        print("Split %d --- Prediction horizon %d" % (split_number, h+1))
        # Try to load an already saved trials object if exist, and increase the max.
        try:  
            trials = pickle.load(open(dir + "/hyperopt/hyp_trials_split_%d_h_%d.p" % (split_number, h+1), "rb"))
            max_trials = len(trials.trials) + TRIALS_STEPS

            print("Found saved Trials! Loading...")
            print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, TRIALS_STEPS))

            # Training.
            X_train = pd.read_csv(dir_data_hyper + "/train/X_train_split%d_h%d.csv" % (split_number, h+1), header = [0, 1], index_col = 0)
            y_train = pd.read_csv(dir_data_hyper + "/train/y_train_split%d_h%d.csv" % (split_number, h+1), header = [0, 1], index_col = 0)
            # Validation.
            Xy_validation_dict = {country: None for country in COUNTRIES_TO_CONSIDER}
            for country in COUNTRIES_TO_CONSIDER:
                X_validation = pd.read_csv(dir_data_hyper + "/validation/%s/X_validation_split%d_h%d.csv" % (country, split_number, h+1), header = [0, 1], index_col = 0) 
                y_validation = pd.read_csv(dir_data_hyper + "/validation/%s/y_validation_split%d_h%d.csv" % (country, split_number, h+1), header = [0, 1], index_col = 0) 
                Xy_validation_dict[country] = (X_validation, y_validation)
            
            best = fmin(fn = lambda x: hyperparameters(x, split_number, h+1, (X_train, y_train), Xy_validation_dict),
                        space = space,
                        algo = tpe.suggest,
                        max_evals = max_trials,
                        trials = trials)

            # Save the trials into a file.
            pickle.dump(trials, open(dir + "/hyperopt/hyp_trials_split_%d_h_%d.p" % (split_number, h+1), "wb"))
        # Create a new trials object and start searching.
        except:  
            # Use the default parameters of the model as starting point.
            starting_point_space1 = {"gamma": 0.,
                                     "n_estimators": 100.,
                                     "reg_alpha": 0.,
                                     "reg_lambda": 1.,
                                     "max_depth": 3., 
                                     "min_child_weight": 1., 
                                     "learning_rate": 0.1, 
                                     "subsample": 1.,
                                     "colsample_bytree": 1.}
            # Use only the independent variable as starting point.
            starting_point_space2 = {"3 Months Anomaly Rainfalls (%)": 0., 
                                     "1 Month Anomaly Rainfalls (%)": 0., 
                                     "Rainfalls (mm)": 0.,
                                     "Price cereals and tubers": 0.,
                                     "Fatalities": 0.,
                                     "NDVI Anomaly": 0.,
                                     "rCSI": 0.,
                                     "Ramadan": 1.,
                                     "FCG": 1.,
                                     "Population": 1.,
                                     "Code": 1., 
                                     "Lat": 1., 
                                     "Lon": 1., 
                                     "Day": 1.,
                                     "Month": 1.,
                                     "Year": 1.}
            
            # Merge the two dictionary to perform the hyperparameter tuning on both dictionaries.
            starting_point_space = dict(starting_point_space1, **starting_point_space2)

            if split_number > 1:
                # Load the file with the best results obtained of the previous split.
                results_hyp = pd.read_csv(dir + "/hyperparameter_tuning.csv")
                best_results_hyp = results_hyp.loc[results_hyp.groupby(["split", "h"]).apply(lambda x:  x["loss_to_minimize"].idxmin())].set_index(["split", "h"])
                best_parameters_previous = best_results_hyp.loc[(split_number-1, h+1)]

                starting_point_space_2 = dict(best_parameters_previous[starting_point_space.keys()])
                trials = generate_trials_to_calculate([starting_point_space, starting_point_space_2])
            else:
                trials = generate_trials_to_calculate([starting_point_space])

            max_trials = TRIALS_STEPS

            # Training.
            X_train = pd.read_csv(dir_data_hyper + "/train/X_train_split%d_h%d.csv" % (split_number, h+1), header = [0, 1], index_col = 0)
            y_train = pd.read_csv(dir_data_hyper + "/train/y_train_split%d_h%d.csv" % (split_number, h+1), header = [0, 1], index_col = 0)
            # Validation.
            Xy_validation_dict = {country: None for country in COUNTRIES_TO_CONSIDER}
            for country in COUNTRIES_TO_CONSIDER:
                X_validation = pd.read_csv(dir_data_hyper + "/validation/%s/X_validation_split%d_h%d.csv" % (country, split_number, h+1), header = [0, 1], index_col = 0) 
                y_validation = pd.read_csv(dir_data_hyper + "/validation/%s/y_validation_split%d_h%d.csv" % (country, split_number, h+1), header = [0, 1], index_col = 0) 
                Xy_validation_dict[country] = (X_validation, y_validation)

            best = fmin(fn = lambda x: hyperparameters(x, split_number, h+1, (X_train, y_train), Xy_validation_dict),
                        space = space,
                        algo = tpe.suggest,
                        max_evals = max_trials,
                        trials = trials)

            # Save the trials into a file.
            pickle.dump(trials, open(dir + "/hyperopt/hyp_trials_split_%d_h_%d.p" % (split_number, h+1), "wb"))

