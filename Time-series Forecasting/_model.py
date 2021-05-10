import numpy as np
import pandas as pd
import itertools
import xgboost as xgb
from _utils import *

def model(train, test, lags_dict, out, target, split_number, hyper = None, format = None, dir_data = None, 
          importance_type = "weight", n_jobs = 1):
    """
    This function allows to predict 'out' steps ahead in the future of the 'target' variable of each site in the
    'train' group. The predictions of 'out' steps in the future start from the last date of the 'train' group 
    provided.
    
    """
    # Use the best parameters obtained through a previous hyperparameter tuning.
    if hyper is not None:
        best_result, parameter_names_model, parameter_names_feature = hyper
        # Define the best parameters for the current split obtained by the hyperparameter tuning.
        parameter_names = parameter_names_model + parameter_names_feature
        best_parameters = best_result.loc[split_number][parameter_names].astype(float)
        # Model parameters.
        best_parameter_model = best_parameters[parameter_names_model]
        # Feature parameters.
        best_parameter_feature = best_parameters[parameter_names_feature]        

    ####################
    ### DATA LOADING ###
    ####################
    
    print("Loading data...")

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
                X_train = load(dir_data + f"/train/{country}/{province}/X_train_split_{split_number}_h_{h+1}", format) 
                y_train = load(dir_data + f"/train/{country}/{province}/y_train_split_{split_number}_h_{h+1}", format) 
                # Test samples.
                X_test = load(dir_data + f"/test/{country}/{province}/X_test_split_{split_number}_h_{h+1}", format) 

                # Get the features to keep for the current prediction horizon according to the hyperparameter tuning.
                if hyper is not None:
                    # Select features.
                    # Decide the indicators to keep based on values (0 or 1).
                    space_features = [k for k,v in dict(best_parameter_feature.loc[h+1]).items() if v == 1]
                    # Select all the corresponding lags.
                    space_features = {feature: take_lags(feature, lags = lags_dict[feature], h = h+1) for feature in space_features}
                    # Flat dictionary values.
                    features = list(itertools.chain(*list(space_features.values())))
                    # Keep features.
                    X_train = X_train[features].sort_index(axis = 1)
                    X_test = X_test[features].sort_index(axis = 1)

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
    columns = test.xs(target, axis = 1, level = 2, drop_level = False).rename(columns = {target: "Forecast"}).columns
    predictions = pd.DataFrame(index = test.index, columns = columns)

    # Training model.
    models = {h+1: None for h in range(out)}
    r2_train = {h+1: None for h in range(out)}
    for h in range(out):
        # Train the model for the current prediction horizon.
        X_train, y_train = training_points["X"][h+1], training_points["y"][h+1]
        
        # Get the best model parameters for the current prediction horizon if exist the information about.
        if hyper is not None:
            # Select best model parameters.
            best_parameter_model_h = dict(best_parameter_model.loc[h+1])
            # Convert to int type the float numbers that are integers (avoid issues).
            best_parameter_model_h = {k: int(v) if v.is_integer() else v for k,v in best_parameter_model_h.items()}

        # Model.
        if hyper is not None:
            model = xgb.XGBRegressor(**best_parameter_model_h, objective = "reg:squarederror", n_jobs = n_jobs)
        else:
            model = xgb.XGBRegressor(n_estimators = 100, objective = "reg:squarederror", n_jobs = n_jobs)
        # Train model.
        model.fit(X_train, y_train)
        
        # Feature importance.
        feature_importance = model.get_booster().get_score(importance_type = importance_type)
        
        # Save models.
        models[h+1] = (model, X_train.columns, feature_importance)
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
                # Store the predicted values into the dataframe.
                predictions[(country, province, "Forecast")].loc[predictions.index[h]] = y_hats
 
    # Define the shape of the training and test points.    
    shape_training_points = training_points.copy()
    for h in range(out):
        shape_training_points["X"][h+1] = shape_training_points["X"][h+1].shape
        shape_training_points["y"][h+1] = shape_training_points["y"][h+1].shape
 
    print("Complete!")

    return predictions, models, r2_train, shape_training_points
