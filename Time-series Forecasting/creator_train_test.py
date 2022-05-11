import warnings
warnings.filterwarnings("ignore")

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

parser_user = argparse.ArgumentParser(description = "This file allows to create the training (input and output) and test (input) points for the selected countries at provincial level in order to forecast the corresponding target time-series. The points are automatically collected starting from the availability of data through the algorithm *LagsCreator*.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

# Example usage: python creator_train_test.py --countries "Yemen" --end_date "2022-02" --folder_path_to_dataset "./Yemen/dataset" --number_of_splits 5 --gui_interface 

parser_user.add_argument('--countries', type = str, default = ["Yemen"], nargs = "+", help = "Select the countries to consider for the current analysis. N.B. If more than one country is selected, the forecasting analysis of the countries is fused together: 1) the test splitting is the same for all the countries; 2) the training phase puts into an unique 'pot' the data of all the selected countries; 3) if the validation phase is performed, the validation loss of each country are first considered independetly and then averaged together.")
parser_user.add_argument('--end_date', type = str, default = "2020-11", help = "The end date - format YYYY-MM. The selected date represents the month that will refer to the last split (see *NestedCV* algorithm) on which forecasting will be tested.")
parser_user.add_argument('--folder_path_to_dataset', type = str, default = "./dataset", help = "The path to the folder where all the training and test points will be stored. If the folder doesn't exist, it will be created.")
parser_user.add_argument('--target', type = str, default = "FCG", help = "Define the name of the indicator we want to predict.")
parser_user.add_argument('--test_size', type = int, default = 30, choices = range(1, 31), help = "Define the number of days we want to learn to predict for the target variable. The current nested cross validation (see *NestedCV* algorithm) is meant to predict large time horizons (e.g. 20-30). The allowed values are [1-30].")
parser_user.add_argument('--number_of_splits', type = int, default = 10, help = "Define the number of total split we want to evaluate using the nested cross validation technique (see *NestedCV* algorithm).")
parser_user.add_argument('--features_time', type = str, default = ["Day", "Dayofweek", "Month", "Year"], nargs = "+", choices = ["Day", "Dayofweek", "Month", "Year", "Week", "Quarter", "Weekofyear", "Dayofyear"], help = "Define the time features we want to consider for the input points (see *LagsCreator* algorithm).")
parser_user.add_argument('--step_between_samples', type = int, default = 1, help = "Define the separation step during the creation of the training points (the step of the temporal sliding window (see *LagsCreator* algorithm)).")
parser_user.add_argument('--format', type = str, default = "feather", choices = ["csv", "feather", "xlsx"], help = "The file format to store training and test points.")
parser_user.add_argument('--gui_interface', action = "store_true", help = "If you want to select the lags for each indicator through a GUI interface otherwise the corresponding default value are taken (see *_default*).")

args = parser_user.parse_args()

#################
### WORKSPACE ###
#################

# Create the workspace folder for storing training and test points.
if os.path.exists(args.folder_path_to_dataset):
    if not click.confirm(f"The folder '{args.folder_path_to_dataset}' already exists. If you continue you will overwrite the existing folder. Continue?", default = True):
        exit() 
    shutil.rmtree(args.folder_path_to_dataset) 
os.makedirs(args.folder_path_to_dataset)

# Save some parameters into a pickle file.
with open(f"{args.folder_path_to_dataset}/global_variables", "wb") as f:
    pickle.dump([args.countries, args.target, args.test_size, args.number_of_splits, args.features_time, args.format], f)
    
# Save argparse arguments of the current session into a txt file.
with open(f"{args.folder_path_to_dataset}/commandline_args.txt", "w") as f:
    print(args.__dict__, file = f)
    
####################
### DATA LOADING ###
####################

# Load the time-series dataset.
dfs = []
indicators = []
for country in args.countries:
    # Load the time-series data.
    df = pd.read_csv(f"../Dataset time-series/output_data/{country}/{country}.csv", header = [0, 1], index_col = 0)
    df.index = pd.to_datetime(df.index)
    df.index.freq = "D"
    # Add a level information regarding the country.
    df.columns = pd.MultiIndex.from_tuples(map(lambda x: (country, x[0], x[1]), df.columns), 
                                           names = ["Country", "AdminStrata", "Indicator"])
    # Select the defined temporal range (availability data -> end of the selected month).
    date = pd.to_datetime(args.end_date) + pd.offsets.MonthEnd(1)
    if date.month == 2:
        date = date + pd.offsets.Day(3)
    df = df.loc[:date]
    # Save indicator names at provincial level.
    for province in df.columns.get_level_values("AdminStrata").unique():
        indicators.append(sorted(df[country][province].columns))
    # Append country.
    dfs.append(df)

# Concatenate data of the countries.
df = pd.concat(dfs, axis = 1)

#####################
### GUI INTERFACE ###
#####################

# GUI NUMBER 1: INDICATORS
# GUI interface in order to select desired indicators.
if args.gui_interface:
    # Create user interface.
    interface = gui()
    # Selection of the indicators.
    selected_indicators = interface.GUI_indicators_1(list(df.columns.get_level_values("Indicator").unique()), args.target)
    selected_indicators = [k for k,v in selected_indicators.items() if v]
    
    # Check intersection between the selected indicators and the indicators present for each province.
    indicators = [list(set(indicator).intersection(set(selected_indicators))) for indicator in indicators]

    # Check if the indicators are present for all the countries and their provinces (necessary condition). 
    if not all_equal(indicators):
        raise ValueError("All the provinces of all the selected countries must contain the same indicators.")

    # Keep only the selected indicators.
    df = df.loc[:, df.columns.get_level_values("Indicator").isin(selected_indicators)]
else:
    # Check if the indicators are present for all the countries and their provinces (necessary condition). 
    if not all_equal(indicators):
        raise ValueError("All the provinces of all the selected countries must contain the same indicators.")
    
# Define the unique indicators of this dataset.
indicators = indicators[0]

# GUI NUMBER 2: LAGS
# Define a default lags dictionary for each indicator present in the dataset.
defaultLags = {indicator: defaultLags_1[indicator] if indicator in defaultLags_1.keys() else np.array([1]) for indicator in indicators}

# GUI interface in order to select the desired lags.
if args.gui_interface:
    interface = gui()
    lags_dict = interface.GUI_lags_1(defaultLags)
else:
    lags_dict = defaultLags.copy()

# Save the lags dictionary into a pickle file.
with open(f"{args.folder_path_to_dataset}/lags_dict", "wb") as fp:
    pickle.dump(lags_dict, fp)    
    
###################################
### TRAINING AND TEST WORKSPACE ###
###################################

# Save dataset.
df.to_csv(f"{args.folder_path_to_dataset}/dataset.csv")

# Create folder for containing training data.
os.makedirs(f"{args.folder_path_to_dataset}/train")
# Create folder for containing test data.
os.makedirs(f"{args.folder_path_to_dataset}/test")
for country in args.countries:
    provinces = df[country].columns.get_level_values("AdminStrata").unique()
    for province in provinces:
        os.makedirs(f"{args.folder_path_to_dataset}/train/{country}/{province}") 
        os.makedirs(f"{args.folder_path_to_dataset}/test/{country}/{province}") 
        
############################
### DATA POINTS CREATION ###
############################
        
# Create the nested cross validation.
cv = NestedCV(args.number_of_splits, args.test_size)
# Nested cross validation.
SPLITS = cv.get_splits(df)
for split_number, (train, test) in SPLITS.items():
    print(f"Split {split_number}: range of days to predict (test) between {test.index[0].date()} - {test.index[-1].date()}")

# Create training and test points.
print("Creation of the training and test points...")
for split_number, (train, test) in SPLITS.items():
    print(f"Split {split_number}. Please wait.")
    # Define the first multi-sites (countries).
    countries = train.columns.get_level_values("Country").unique()
    for country in countries:
        train_country = train[country]
        # Define the second multi-sites (provinces).
        provinces = train_country.columns.get_level_values("AdminStrata").unique()
        for province in provinces:
            creator = LagsCreator(train_country[province], lags_dictionary = lags_dict, target = args.target)
            for h in range(args.test_size):
                # Training samples.
                X_train, y_train, X_test, _ = creator.to_supervised(h = h+1, step = args.step_between_samples, single_step = True, 
                                                                    return_dataframe = True, feature_time = args.features_time, 
                                                                    dtype = float)
                
                # Flat multi-index column.
                X_train.columns = ["|".join(col).strip() for col in X_train.columns.values]
                y_train.columns = ["|".join(col).strip() for col in y_train.columns.values]
                X_test.columns = ["|".join(col).strip() for col in X_test.columns.values]

                # Save train input and output.
                save(X_train, f"{args.folder_path_to_dataset}/train/{country}/{province}/X_train_split_{split_number}_h_{h+1}",
                     args.format)
                save(y_train, f"{args.folder_path_to_dataset}/train/{country}/{province}/y_train_split_{split_number}_h_{h+1}", 
                     args.format)
                # Save test input.
                save(X_test, f"{args.folder_path_to_dataset}/test/{country}/{province}/X_test_split_{split_number}_h_{h+1}", 
                     args.format)

print("Complete!")
