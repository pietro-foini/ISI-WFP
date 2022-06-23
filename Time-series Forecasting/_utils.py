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

def adjust_boxplot(ax, colors=["#355269", "#5eb91e"], linewidth=1.5):
    """Adjust boxplot layout with group of couple of boxes."""
    for i, artist in enumerate(ax.artists):
        if len(colors) > 1:
            if i % 2 == 0:
                col = colors[0]
            else:
                col = colors[1]   
        else:
            col = colors[0]
        # This sets the color for the main box.
        artist.set_edgecolor(col)
        artist.set_facecolor("None")
        # Each box has 6 associated Line2D objects (to make the whiskers, etc.).
        # Loop over them here, and use the same colour as above.
        for j in range(i*5,i*5+5):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)

    # Also fix the legend.
    if len(colors) > 1:
        for legpatch in ax.get_legend().get_patches(): 
            legpatch.set_linewidth(linewidth)
            legpatch.set_edgecolor(legpatch.get_facecolor())
            legpatch.set_facecolor("None")
    
    return ax

def plot_prediction(df, split, country, province, ax):
    df = df[split][country][province]
    # Add the time-series to the figure.
    df.columns.name = None
    last_date = df["Forecast"].last_valid_index()
    df = df.loc[:last_date]
    for column in df.columns: 
        if column == "FCG":
            df[column].plot(ax = ax, label = "_", style = ":", c = "black", alpha = 0.5)
        elif column == "Naive":
            df[column].plot(ax = ax, label = "naive", style = "-", c = sns.color_palette("tab10")[0], legend = False)
        else:
            df[column].plot(ax = ax, label = "model", style = "-", c = sns.color_palette("tab10")[1], legend = False)
            
    # Set legend.
    ax.legend(title = df.columns.name, loc = "best")
    # Set axis names.
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.autoscale()
