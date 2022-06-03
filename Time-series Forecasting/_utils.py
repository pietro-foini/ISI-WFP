import pandas as pd
import numpy as np
import itertools 
import matplotlib.pyplot as plt
import seaborn as sns

def save(df, name, format):
    """Function that allows saving dataframe using different format."""
    if format == "csv":
        df.to_csv(name + ".csv")
    elif format == "feather":
        df.to_feather(name + ".feather")
    elif format == "xlsx":
        df.to_excel(name + ".xlsx")
        
def load(name, format):
    """Function that allows loading dataframe using different format."""
    if format == "csv":
        return pd.read_csv(name + ".csv")
    elif format == "feather":
        return pd.read_feather(name + ".feather")
    elif format == "xlsx":
        return pd.read_excel(name + ".xlsx")
        
def all_equal(iterable):
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

def take_lags(x, lags = None, h = None):
    if lags is not None:
        lags = [f"{x}|x(t)" if i == 1 else f"{x}|x(t-{(i-1)})" for i in lags]
    else:
        lags = [f"{x}|x(t+{h})"]
    return lags

def r2(y_true, y_pred):
    """r2 metric according to xgboost implementation."""
    return (1 - (((y_true - y_pred) ** 2).sum())/(((y_true - y_true.mean()) ** 2).sum()))

def plot_r2_box_plot(data, ax, label1, label2, title = None, table = None, color1 = sns.color_palette("tab10")[0], 
                     color2 = sns.color_palette("tab10")[1], ylabel="R$^2$"):
    # Define x-ticks.
    x_ticks = np.arange(1, len(data) + 1)
    medianprops = dict(linestyle = "-", linewidth = 2.5)
    # Boxplot.
    bp1 = ax.boxplot(data.xs(label1, axis = 1, level = 1).T, positions = x_ticks*2.0 - 0.4, sym = "", widths = 0.6, 
                     medianprops = medianprops)
    bp2 = ax.boxplot(data.xs(label2, axis = 1, level = 1).T, positions = x_ticks*2.0 + 0.4, sym = "", widths = 0.6, 
                     medianprops = medianprops, manage_ticks = False)

    # Draw temporary lines for legend.
    ax.plot([], c = color1, label = label1)
    ax.plot([], c = color2, label = label2)
    
    # Set attributes of the plot.
    ax.set_title(title)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(ylabel)
    #ax.set_ylim([0, 1])
    ax.tick_params(labeltop = False, labelright = True)
    ax.set_xticks(ax.get_xticks() + 0.5)
    ax.set_xticklabels(x_ticks)  
    ax.set_xlim([0, ax.get_xticks()[-1] + 1.8])
    ax.legend(loc = "best")
    
    # Insert information table.
    if table is not None:
        ax.table(cellText = table.values, rowLabels = table.index, colLabels = table.columns,
                 bbox = [0.0, -0.55, 1, .28], loc = "bottom")

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], color = color)
        plt.setp(bp["whiskers"], color = color)
        plt.setp(bp["caps"], color = color)
        plt.setp(bp["medians"], color = color)

    # Set the colors of the boxplots.
    set_box_color(bp1, color1) 
    set_box_color(bp2, color2)

    # Add boxplots to the axis.
    ax = bp1
    ax = bp2

    return ax
