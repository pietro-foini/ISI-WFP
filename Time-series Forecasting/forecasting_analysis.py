import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import pandas as pd
import numpy as np
import os
import argparse
import shutil
import pickle
from _utils import *

# Set default sizes for figures.
plt.style.use("default") # style matplotlib
plt.rc("axes", labelsize = 12) # fontsize of the x and y labels
plt.rc("axes", titlesize = 15) # fontsize of the axes title
plt.rc("xtick", labelsize = 12) # fontsize of the tick labels
plt.rc("ytick", labelsize = 12) # fontsize of the tick labels
plt.rc("legend", fontsize = 12) # legend fontsize

###############################
### USER-DEFINED PARAMETERS ###
###############################

parser_user = argparse.ArgumentParser(description = "This file allows to analyze forecasting results obtained from a previous analysis.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)

# Example usage: python forecasting_analysis.py "./Yemen/out_hyper_1/out_test" --folder_path_to_hyperparameter_tuning "./Yemen/out_hyper_1" --split_features_importance 10

parser_user.add_argument('folder_path_to_output_test', type = str, help = "The path to the folder containing the results on the test splits.")
parser_user.add_argument('--folder_path_to_hyperparameter_tuning', type = str, help = "The path to the folder containing the results obtained from the related hyperparameter tuning (optional).")
parser_user.add_argument('--split_features_importance', type = int, default = 10, help = "The split to use in order to compute the feature importance.")

args = parser_user.parse_args()

#################
### WORKSPACE ###
#################

# Define the folder where to store the results arising from the current analysis.
dir_output = args.folder_path_to_output_test + "/images"
if os.path.exists(dir_output):
    shutil.rmtree(dir_output) 
os.makedirs(dir_output)

################
### ANALYSIS ###
################

## Hyper-parameters tuning results.

if args.folder_path_to_hyperparameter_tuning is not None:
    # Load all the configurations tested during hyperpameter tuning.
    hyper_params = pd.read_csv(args.folder_path_to_hyperparameter_tuning + "/hyperparameter_tuning.csv")

    # For each split and prediction horizon, we get the best configuration according to the minimum of the 'loss_to_minimize'.
    best_results = hyper_params.loc[hyper_params.groupby(["split", "h"])["loss_to_minimize"].idxmin()].set_index(["split", "h"])
    best_results.to_csv(args.folder_path_to_hyperparameter_tuning + "/best_results_hyper.csv")

    # Isolate the shapes of the training and validation data for each split and prediction horizon.
    shapes = best_results[["shape_train", "shape_val"]].applymap(eval)    
    shapes["train_points"], shapes["features"] = zip(*shapes["shape_train"])
    shapes["val_points"], _ = zip(*shapes["shape_val"])
    shapes.drop(["shape_train", "shape_val"], axis = 1, inplace = True)    
    shapes = shapes.unstack("h").stack(0)

    # Get the r2 results corresponding to the best configurations.
    best_r2 = best_results[["r2_train", "r2_val"]].unstack("split").reorder_levels([1, 0], axis = 1).sort_index(axis = 1, level = [0, 1])

    # Create boxplot r2 over all the splits.
    fig, ax = plt.subplots(figsize = (10, 3))    
    plot_r2_box_plot(best_r2, ax, label1 = "r2_train", label2 = "r2_val", color1 = "#355269", color2 = "#5eb91e")
    # Save figure.
    fig.savefig(dir_output + "/r2_hyper.png" , bbox_inches = "tight", dpi = 300)

    SPLITS = best_r2.columns.get_level_values("split").unique()
    
    # Create 'boxplot' for each split.
    for split in SPLITS:
        fig, ax = plt.subplots(figsize = (10, 3))    
        plot_r2_box_plot(best_r2[[split]], ax, label1 = "r2_train", label2 = "r2_val", title = f"Split {split}",
                         table = shapes.loc[split], color1 = "#355269", color2 = "#5eb91e")
        # Save figure.
        fig.savefig(dir_output + f"/r2_hyper_split_{split}.png" , bbox_inches = "tight", dpi = 300)


## Forecasting results.

# Load forecasting results for each split.
xls = pd.ExcelFile(args.folder_path_to_output_test + "/forecast.xlsx")

forecast_splits = dict()
for i, split in enumerate(xls.sheet_names[1:]):
    forecast_split = pd.read_excel(xls, split, index_col = 0, header = [0, 1, 2])
    # Reset the index.
    forecast_split.index = np.arange(1, len(forecast_split) + 1)
    forecast_split.index.names = ["Prediction horizon"]
    # Save the predictions.
    forecast_splits[split] = forecast_split
    
forecast_splits = pd.concat(forecast_splits, axis = 1)

# Load the training shapes for each split and prediction horizon.
training_shape = pd.read_csv(args.folder_path_to_output_test + "/training_shapes.csv", header = [0, 1], index_col = 0)

# Load the losses for each split and province (model and naive).
loss_sites = pd.read_csv(args.folder_path_to_output_test + "/loss_sites.csv", index_col = [0, 1], header = [0, 1])

# Load the losses for each split and prediction horizon (model and naive).
loss_h = pd.read_csv(args.folder_path_to_output_test + "/loss_h.csv", index_col = [0, 1], header = [0, 1])

# Load the r2 for each split and prediction horizon (model and naive).
r2_results = pd.read_csv(args.folder_path_to_output_test + "/r2_results.csv", index_col = 0, header = [0, 1, 2])

## Total prediction loss for the sites among the various splits

COUNTRIES = loss_sites.index.get_level_values(0).unique()
MODELS = loss_sites.columns.get_level_values(1).unique() # model or naive

# Define the number of figures on x axis.
cols = 2
# Define the number of figures on y axis.
rows = len(COUNTRIES)
# Define the subplot figure.
fig, axs = plt.subplots(rows, cols, sharey = True, figsize = (5*cols, 3*rows), squeeze = False)
fig.subplots_adjust(wspace = 0.1, hspace = 0.8)
for j, country in enumerate(COUNTRIES):
    for i, m in enumerate(MODELS):         
        value = loss_sites.loc[country].xs(m, axis = 1, level = 1, drop_level = False)
        # Create box-plot.
        value.transpose().reset_index(drop = True).boxplot(rot = 90, ax = axs[j,i])
        # Set attributes box-plot.
        axs[j,i].set_title(f"{country} -- {m}")
        axs[j,i].set_xlabel("Province")
        axs[j,i].set_ylabel(m)
        axs[j,i].grid(b = None)
        
fig.savefig(dir_output + "/loss_provinces.png" , bbox_inches = "tight", dpi = 300)

## Loss of each split as function of the prediction horizon.

colors = ["dodgerblue", "deeppink"]

COUNTRIES = loss_h.index.get_level_values(0).unique()
SPLITS = loss_h.columns.get_level_values(0).unique()
MODELS = loss_h.columns.get_level_values(1).unique()

# Define the number of figures on x axis.
cols = len(COUNTRIES)
# Define the number of figures on y axis.
rows = len(SPLITS)
# Define the subplot figure.
fig, axs = plt.subplots(rows, cols, figsize = (10*cols, 5*rows), squeeze = False)
fig.subplots_adjust(wspace = 0.05, hspace = 0.5)
for j, country in enumerate(COUNTRIES):
    for i, split in enumerate(SPLITS):      
        value = loss_h.loc[(country, split)]
        for k, m in enumerate(MODELS):  
            # Plot.
            value[m].plot(style = ".-", label = m, ax = axs[i,j], c = colors[k])
            # Set attributes of the plot.
            axs[i,j].set_title(f"{country} \n {split}")
            axs[i,j].legend(loc = "best")
            axs[i,j].set_xlabel("Prediction horizon")
            axs[i,j].set_ylabel("mse")

## Total loss over the splits as function of the prediction horizon.

# Median and quantiles.

COUNTRIES = loss_h.index.get_level_values(0).unique()
SPLITS = loss_h.columns.get_level_values(0).unique()
MODELS = loss_h.columns.get_level_values(1).unique()

for j, country in enumerate(COUNTRIES):
    fig, axs = plt.subplots(figsize = (10, 5))    
    for i, m in enumerate(MODELS):  
        value = loss_h.loc[country].xs(m, axis = 1, level = 1, drop_level = False)
        value_statistic = value.agg([lambda x: x.quantile(0.25), np.median, lambda x: x.quantile(0.75)], axis = 1)
        value_statistic.columns = ["lower_quantile", "median", "upper_quantile"]

        # Plot.
        value_statistic["median"].plot(style = ".-", label = m, ax = axs, c = colors[i])
        axs.fill_between(x = value_statistic["median"].index, y1 = value_statistic["lower_quantile"], 
                         y2 = value_statistic["upper_quantile"], color = colors[i], alpha = 0.3)
        # Set attributes of the plot.
        axs.legend(loc = "upper left")
        axs.set_xlabel("Prediction horizon")
        axs.set_ylabel("mse")
        axs.set_xticks(np.arange(1, len(value_statistic)+1))

        fig.savefig(dir_output + f"/Loss_{country}.png", bbox_inches = "tight", dpi = 300)
        
# Mean and standard deviation.

COUNTRIES = loss_h.index.get_level_values(0).unique()
SPLITS = loss_h.columns.get_level_values(0).unique()
MODELS = loss_h.columns.get_level_values(1).unique()

for j, country in enumerate(COUNTRIES):
    fig, axs = plt.subplots(figsize = (10, 5))    
    for i, m in enumerate(MODELS):  
        value = loss_h.loc[country].xs(m, axis = 1, level = 1, drop_level = False)
        value_statistic = value.agg([np.mean, np.std], axis = 1)
        value_statistic.columns = ["mean", "std"]

        # Plot.
        if i == 0:
            trans1 = Affine2D().translate(-0.2, 0.0) + axs.transData
            axs.errorbar(value_statistic.index, value_statistic["mean"], yerr = value_statistic["std"], marker = "o", 
                         linestyle = "none", transform = trans1, c = colors[i], label = m)
        else:
            trans2 = Affine2D().translate(+0.2, 0.0) + axs.transData
            axs.errorbar(value_statistic.index, value_statistic["mean"], yerr = value_statistic["std"], marker = "o", 
                         linestyle = "none", transform = trans2, c = colors[i], label = m)
        
        # Set attributes of the plot.
        axs.legend(loc = "upper left")
        axs.set_xlabel("Prediction horizon")
        axs.set_ylabel("mse")
        axs.set_xticks(np.arange(1, len(value_statistic)+1))

        fig.savefig(dir_output + f"/Loss_{country}_mean.png", bbox_inches = "tight", dpi = 300)

## r2 scores

COUNTRIES = r2_results.columns.get_level_values(0).unique()
SPLITS = r2_results.columns.get_level_values(1).unique()

for country in COUNTRIES:
    fig, ax = plt.subplots(figsize = (10, 3))    
    plot_r2_box_plot(r2_results[country], ax, "r2_model", "r2_naive")
    fig.savefig(dir_output + f"/r2_{country}.png" , bbox_inches = "tight", dpi = 300)
    for split in SPLITS:
        fig, ax = plt.subplots(figsize = (10, 3))    
        plot_r2_box_plot(r2_results[country][[split]], ax, "r2_model", "r2_naive", title = split,
                         table = training_shape[split].transpose());

## Actual vs Forecast

mStyles = ["s","+","x","h","v","^","o","H","*","D"]

COUNTRIES = forecast_splits.columns.get_level_values(1).unique()
SPLITS = forecast_splits.columns.get_level_values(0).unique()

for country in COUNTRIES:
    PROVINCES = forecast_splits.xs(country, axis = 1, level = 1).columns.get_level_values(1).unique()
    colors = {province: plt.get_cmap("tab20")(i) for i,province in enumerate(PROVINCES)}
    
    fig, axs = plt.subplots(figsize = (8, 8))
    # Add bisector.
    axs.plot(axs.get_xlim(), axs.get_ylim(), color = "black", linestyle = "--")

    for i,split in enumerate(SPLITS):
        predictions = forecast_splits[(split, country)]
        predictions = predictions.drop(["Naive"], axis = 1, level = 1)
        predictions = predictions/100
   
        def plot_scatter(group):
            gr = group[group.name]
            gr.plot.scatter(x = "FCG", y = "Forecast", s = 8, marker = mStyles[i], color = colors[group.name], 
                            ax = axs, label = group.name)

        predictions.groupby(axis = 1, level = 0).apply(plot_scatter)

        # Legend.
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        leg = plt.legend(by_label.values(), by_label.keys(), loc = "upper left", prop = {"size": 8})
        axs.add_artist(leg)
        h = [plt.plot([],[], color = "black", marker = mStyles[i], ls = "")[0] for i,cplit in enumerate(SPLITS)]
        axs.legend(handles = h, labels = [split for split in SPLITS], loc = "lower left")
        axs.set_xlim(0, 1)
        axs.set_ylim(0, 1)

    axs.set_xlabel("Actual", fontsize = 15)
    axs.set_ylabel("Forecast", fontsize = 15)

    fig.savefig(dir_output + f"/scatter_{country}.png", bbox_inches = "tight", dpi = 300)
    
## Features importance.

prediction_horizons = [1, 7, 14, 21, 28]

fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (15, 15))
fig.subplots_adjust(wspace = 0.9)

axs = axs.ravel()
for i, h in enumerate(prediction_horizons):
    with open(args.folder_path_to_output_test + f"/features_importance/features_split_{args.split_features_importance}_h_{h}", "rb") as fp:
        f_imp = pickle.load(fp)
    
    # Create histogram feature importance.
    keys = list(f_imp.keys())
    values = list(f_imp.values())
    data = pd.DataFrame(data = values, index = keys, columns = ["score"]).sort_values(by = "score", ascending = True)    
    data.plot(kind = "barh", ax = axs[i])
    axs[i].set_title(f"Prediction horizon: {h}")
    
# Remove extra plot.
axs[-1].set_axis_off()
    
fig.savefig(args.folder_path_to_output_test + "/features_importance/features_importance.png", bbox_inches = "tight", dpi = 300)

