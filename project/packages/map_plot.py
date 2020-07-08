import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Python module.
#
#
# Pietro Foini
#
# Year: 2020
    
def draw_adminstratas(country, adminstratas, folder_to_shapefiles, figsize = (15, 10), annotation = False, 
                      annotation_selected = False):
    """
    This module allows to plot the political boundaries of the selected country. More precisely, it allows to view the provinces selected, 
    thanks to the 'adminstratas' parameter from the others.
    
    Parameters
    ----------
    country: a string parameter corresponding to the name of the country.
    adminstratas: a list of strings of provinces which are highlighted when viewing the map.
    folder_to_shapefiles: the path to reach the folder where the shapefiles are stored.
    figsize: the size of the map figure.
    annotation: a boolean parameter to set if you want to visualize the names of all the provinces.
    annotation_selected: a boolean parameter to set if you want to visualize only the names of the provinces selected.
    
    """
    # Load the dataframe of the country.
    gdf = gpd.read_file(folder_to_shapefiles + "/" + country + ".shp")
    
    # Draw only the adminstratas defined by the user.
    gdf["draw"] = gdf.apply(lambda x: 1 if x.admin in adminstratas else np.nan, axis = 1)
    # Create figure.
    fig, ax = plt.subplots(figsize = figsize)
    gdf.plot(column = "draw", ax = fig.gca(), cmap = "bwr", edgecolor = "black", legend = False, alpha = 0.6,
             missing_kwds = {"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "Missing values"})
    if annotation:
        if annotation_selected:
            gdf = gdf[gdf.admin.isin(adminstratas)]
        for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf.admin):
            ax.annotate(label, xy = (x, y), xytext = (3, 3), textcoords = "offset points", color = "black")
    ax.set_title(country)
    plt.axis("off")
    plt.show()
    
def choropleth(country, quantiles, folder_to_shapefiles, figsize = (15, 10), annotation = False, 
               annotation_selected = False):
    """
    This module allows to plot the choropleth map providing a quantiles for the provinces of the selected country.
    
    Parameters
    ----------
    country: a string parameter corresponding to the name of the country.
    quantiles: a pandas serie object with index the provinces names and as values the corresponding quantiles.
    folder_to_shapefiles: the path to reach the folder where the shapefiles are stored.
    figsize: the size of the map figure.
    annotation: a boolean parameter to set if you want to visualize the names of all the provinces.
    annotation_selected: a boolean parameter to set if you want to visualize only the names of the provinces selected.
    
    """
    # Load the dataframe of the country.
    gdf = gpd.read_file(folder_to_shapefiles + "/" + country + ".shp")
    
    # Define the selected adminstratas.
    adminstratas = quantiles.index
    
    # Draw only the adminstratas defined by the user.
    gdf["draw"] = gdf.apply(lambda x: quantiles.loc[x.admin] if x.admin in adminstratas else np.nan, axis = 1)
    # Create figure.
    f, ax = plt.subplots(1, figsize = figsize)
    ax = gdf.plot(column = "draw", ax = ax, edgecolor = "black", legend = True, alpha = 1, scheme = "quantiles", 
                  missing_kwds = {"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "Missing values"}, 
                  legend_kwds = dict(loc = "upper left", bbox_to_anchor = (1, 1)))
    if annotation:
        if annotation_selected:
            gdf = gdf[gdf.admin.isin(adminstratas)]
        for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf.admin):
            ax.annotate(label, xy = (x, y), xytext = (4, 4), textcoords = "offset points", color = "chocolate")

    ax.set_title(country)
    plt.axis("off")
    plt.show()
    