import numpy as np
import pandas as pd
import argparse
import pickle
import shutil
import click
import os
from _gui import *
from _utils import *
from _default import *

# Add the python path to the folder containing some custom packages.
import sys
sys.path.insert(0, "../packages/")
from LagsCreator.LagsCreator import LagsCreator
from NestedCV.NestedCV import NestedCV

###############################
### USER-DEFINED PARAMETERS ###
###############################

parser_user = argparse.ArgumentParser(description = "This file allows to create the training (input and output) and test (input) points for the selected countries at provincial level in order to forecast the corresponding target time-series.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser_user.add_argument('--countries', type = str, default = ["Yemen"], nargs = "+", help = "Select the countries to consider for the current analysis.")
parser_user.add_argument('--start_date', type = str, default = "2018-01-01", help = "The start date - format YYYY-MM-DD. There are no important constraints on the choice of this date. The algorithm (see *LagsCreator*) automatically create the training and the test points starting from the availability of data.")
parser_user.add_argument('--end_date', type = str, default = "2020-11", help = "The end date - format YYYY-MM. The selected date represents the month that will refer to the last split (see *NestedCV*) on which forecasting will be tested.")
parser_user.add_argument('--folder_path_to_dataset', type = str, default = "./dataset", help = "The path to the folder where all the training and test points will be stored. If the folder doesn't exist, it will be created.")
parser_user.add_argument('--target', type = str, default = "FCG", help = "Define the name of the indicator we want to predict.")
parser_user.add_argument('--test_size', type = int, default = 30, choices = range(1, 31), help = "Define the number of days we want to learn to predict for the target variable. Our nested cross validation (see *NestedCV*) was meant to predict large time horizons (e.g. 20-30). The allowed values are [1-30].")
parser_user.add_argument('--number_of_splits', type = int, default = 10, help = "Define the number of total split we want to evaluate using our nested cross validation (see *NestedCV*).")
parser_user.add_argument('--features_time', type = str, default = ["Day", "Dayofweek", "Month", "Year"], nargs = "+", choices = ["Day", "Dayofweek", "Month", "Year", "Week", "Quarter", "Weekofyear", "Dayofyear"], help = "Define the time features we want to consider for the input points (see *LagsCreator*).")
parser_user.add_argument('--step_between_samples', type = int, default = 1, help = "Define the separation step during the creation of the training points (the step of the temporal sliding window (see *LagsCreator*)).")
parser_user.add_argument('--format', type = str, default = "feather", choices = ["csv", "feather", "xlsx"], help = "The file format to store training and test points.")
parser_user.add_argument('--gui_interface', action = "store_true", help = "If you want to select the lags for each indicator through a GUI interface otherwise the corresponding default value are taken (see *_default*).")

args = parser_user.parse_args()

#################
### WORKSPACE ###
#################

# Create the workspace folder for storing training and test points.
if os.path.exists(args.folder_path_to_dataset):
    if not click.confirm("If you continue you will overwrite the existing dataset (folder).", default = True):
        exit() 
    shutil.rmtree(args.folder_path_to_dataset) 
os.makedirs(args.folder_path_to_dataset)

# Save argparse arguments of the current session.
with open(args.folder_path_to_dataset + "/commandline_args.txt", "w") as f:
    f.write("\n".join(sys.argv[1:]))

# Save some parameters into pickle file.
with open(args.folder_path_to_dataset + "/global_variables", "wb") as f:
    pickle.dump([args.countries, args.target, args.test_size, args.number_of_splits, args.features_time, 
                 args.format, args.step_between_samples], f)

# Load the time-series dataset.
dfs = list()
indicators = list()
for country in args.countries:
    # Load the time-series data.
    df = pd.read_csv(f"../Dataset time-series/output_data/{country}/{country}.csv", header = [0, 1], index_col = 0)
    df.index = pd.to_datetime(df.index)
    df.index.freq = "D"
    # Add a level information regarding the county.
    df.columns = pd.MultiIndex.from_tuples(map(lambda x: (country, x[0], x[1]), df.columns), names = ["Country", "AdminStrata", "Indicator"])
    # Select the defined temporal range (end date -> end of the selected month).
    df = df.loc[args.start_date:pd.to_datetime(args.end_date) + pd.offsets.MonthEnd(1)]
    # Save indicator names at province level.
    for province in df.columns.get_level_values(1).unique():
        indicators.append(sorted(df[country][province].columns))
    # Append country.
    dfs.append(df)

# Concatenate data of the countries.
df = pd.concat(dfs, axis = 1)

# GUI interface (it allows to select only some indicators).
if args.gui_interface:
    select_indicators = df.columns.get_level_values(2).unique()
    
    interface = gui()
    select_indicators = interface.GUI4(list(select_indicators), args.target)
    select_indicators = [k for k,v in select_indicators.items() if v]
    
    indicators = [list(set(elem).intersection(set(select_indicators))) for elem in indicators]

    # Check if the indicators are present for all the countries and their provinces (necessary condition). 
    if not all_equal(indicators):
        raise ValueError("All the provinces of all the selected countries must contain the same indicators.")

    df = df.loc[:, df.columns.get_level_values(2).isin(select_indicators)]
else:
    # Check if the indicators are present for all the countries and their provinces (necessary condition). 
    if not all_equal(indicators):
        raise ValueError("All the provinces of all the selected countries must contain the same indicators.")
    
# Define the unique indicators of this dataset.
indicators = indicators[0]

df.to_csv(args.folder_path_to_dataset + "/dataset.csv")

# Define a default lags dictionary for each indicator present in the dataset.
defaultLags = {ind: defaultLags1[ind] if ind in defaultLags1.keys() else np.array([1]) for ind in indicators}

# GUI interface (it allows to modify default lags).
if args.gui_interface:
    interface = gui()
    lags_dict = interface.GUI1(defaultLags)
else:
    lags_dict = defaultLags.copy()

# Save the lags dictionary into a pickle file.
with open(args.folder_path_to_dataset + "/lags_dict", "wb") as fp:
    pickle.dump(lags_dict, fp)

# Create folder for containing training data.
os.makedirs(args.folder_path_to_dataset + "/train")
# Create folder for containing test data.
os.makedirs(args.folder_path_to_dataset + "/test")
for country in args.countries:
    provinces = df[country].columns.get_level_values(0).unique()
    for province in provinces:
        os.makedirs(args.folder_path_to_dataset + f"/train/{country}/{province}") 
        os.makedirs(args.folder_path_to_dataset + f"/test/{country}/{province}") 
        
############
### MAIN ###
############
        
# Create the nested cross validation.
cv = NestedCV(args.number_of_splits, args.test_size)
# Total nested cross validation.
SPLITS = cv.get_splits(df)
for split_number, (train, test) in SPLITS.items():
    print(f"Split {split_number}: range of days to predict (test) between {test.index[0].date()} - {test.index[-1].date()}")

# Create training and test points.
print("Creation training and test points...")
for split_number, (train, test) in SPLITS.items():
    print(f"Split {split_number}. Please wait.")
    # Define the first multi-sites (countries).
    countries = train.columns.get_level_values(0).unique()
    for country in countries:
        train_country = train[country]
        # Define the second multi-sites (provinces).
        provinces = train_country.columns.get_level_values(0).unique()
        for province in provinces:
            creator = LagsCreator(train_country[province], lags_dictionary = lags_dict, target = args.target)
            for h in range(args.test_size):
                # Training samples.
                X_train, y_train, X_test, _ = creator.to_supervised(h = h+1, step = args.step_between_samples, single_step = True, 
                                                                    return_dataframe = True, feature_time = args.features_time, 
                                                                    dtype = float)
                
                # Flat mult-index column.
                X_train.columns = ["|".join(col).strip() for col in X_train.columns.values]
                y_train.columns = ["|".join(col).strip() for col in y_train.columns.values]
                X_test.columns = ["|".join(col).strip() for col in X_test.columns.values]

                # Save train input and output.
                save(X_train, args.folder_path_to_dataset + f"/train/{country}/{province}/X_train_split_{split_number}_h_{h+1}",
                     args.format)
                save(y_train, args.folder_path_to_dataset + f"/train/{country}/{province}/y_train_split_{split_number}_h_{h+1}", 
                           args.format)
                # Save test input.
                save(X_test, args.folder_path_to_dataset + f"/test/{country}/{province}/X_test_split_{split_number}_h_{h+1}", 
                           args.format)

print("Complete!")
