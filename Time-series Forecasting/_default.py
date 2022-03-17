import numpy as np
from hyperopt import hp
from hyperopt.pyll.base import scope 

# DEFAULT: 'creator_train_test.py'.
# Define default lags dictionary for the indicators.
defaultLags_1 = {"1 Month Anomaly Rainfalls (%)": np.array([1,2,3]), 
                 "3 Months Anomaly Rainfalls (%)": np.array([1,2,3]), 
                 "Rainfalls (mm)": np.array([1,2,3]), 
                 "NDVI": np.array([1,2,3]), 
                 "NDVI Anomaly": np.array([1,2,3]), 
                 "Exchange rate": np.array([1,2,3]), 
                 "Price cereals and tubers": np.array([1,2,3]), 
                 "Fatalities": np.array([1,2,3,4,5,6,7]), 
                 "FCG": np.arange(1, 15), 
                 "rCSI": np.array([1,2,3,4,5,6,7]), 
                 "Ramadan": np.array([1]), 
                 "Code": np.array([1]), 
                 "Population": np.array([1]), 
                 "Lat": np.array([1]), 
                 "Lon": np.array([1])}

# DEFAULT: 'hyperparameter_tuning.py'.
# Define default lags for some indicators.
defaultLags_2 = {"1 Month Anomaly Rainfalls (%)": np.array([1]), 
                 "Rainfalls (mm)": np.array([1]), 
                 "Price cereals and tubers": np.array([1]), 
                 "NDVI Anomaly": np.array([1]),
                 "Fatalities": np.array([1,2,3]),
                 "FCG": np.array([1,2,3,4,5,6,7,8,9,10,11,12]), 
                 "rCSI":np.array([1,2,3]), 
                 "Ramadan": np.array([1])}

# Define default time features.
defaultTimes = ["Day", "Month", "Year"]

# Define the parameters of the XGBoost model to which perform the hyperparameter tuning.
defaultSpace_model = {"gamma": hp.quniform("gamma", 0.0, 1., 0.005),
                      "n_estimators": 1000,
                      "reg_alpha": hp.quniform("reg_alpha", 0.0, 1., 0.005),
                      "reg_lambda": hp.quniform("reg_lambda", 0.0, 1., 0.005),
                      "max_depth": scope.int(hp.quniform("max_depth", 3, 6, 1)), 
                      "min_child_weight": hp.quniform("min_child_weight", 1., 10., 0.1), 
                      "learning_rate": hp.quniform("learning_rate", 0.001, 0.3, 0.001), 
                      "subsample": hp.quniform("subsample", 0.6, 1.0, 0.05),
                      "colsample_bytree": hp.quniform("colsample_bytree", 0.6, 1.0, 0.05)}

# Define the parameters regarding the indicators to keep during the hyperparameter tuning (indicator selection during hyperparameter tuning).
defaultSpace_indicators = {"1 Month Anomaly Rainfalls (%)": hp.choice("1 Month Anomaly Rainfalls (%)", [True, False]), 
                           "Rainfalls (mm)": hp.choice("Rainfalls (mm)", [True, False]),
                           "Price cereals and tubers": hp.choice("Price cereals and tubers", [True, False]),
                           "Fatalities": hp.choice("Fatalities", [True, False]),
                           "NDVI Anomaly": hp.choice("NDVI Anomaly", [True, False]),
                           "rCSI": hp.choice("rCSI", [True, False]), 
                           "Ramadan": True,
                           "FCG": True,
                           "Day": True,
                           "Month": True, 
                           "Year": True}

# Define some default parameters of the model for starting configuration of the hyperparameter tuning search.
defaultStarting_point_space_model = {"gamma": 0.,
                                     "n_estimators": 100,
                                     "reg_alpha": 0.,
                                     "reg_lambda": 1.,
                                     "max_depth": 3, 
                                     "min_child_weight": 1., 
                                     "learning_rate": 0.1, 
                                     "subsample": 1.,
                                     "colsample_bytree": 1.}
