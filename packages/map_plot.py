import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
plt.style.use("default")

# Python module.
#
#
# Pietro Foini
#
# Year: 2020
    
def draw_adminstratas(country, adminstratas, folder_to_shapefiles, figsize = (15, 10), cmap = "bwr", annotation = False, 
                      annotation_selected = False, path_to_save = None, dpi = 100):
    """
    This module allows to plot the political boundaries of the selected country using shapefiles. More precisely, it allows to view the
    provinces (adminstratas) provided by the users.
    
    Parameters
    ----------
    country: a string parameter corresponding to the name of the country; e.g. "Yemen".
    adminstratas: a list of strings of provincial names which are highlighted when viewing the map; e.g. ["Abyan", "Aden", "Taizz"].
    folder_to_shapefiles: the path to reach the folder where the right shapefile is stored (with desired political granularity).
    figsize: the size of the figure.
    cmap: the cmap to use for drawing map.
    annotation: a boolean parameter to set if you want to visualize the names of all the provinces of the country.
    annotation_selected: a boolean parameter to set if you want to visualize only the names of the provinces provided.
    
    """
    # Load the dataframe of the country.
    gdf = gpd.read_file(folder_to_shapefiles + "/" + country + ".shp")
    
    # Draw only the adminstratas defined by the user.
    gdf["draw"] = gdf.apply(lambda x: 1 if x.admin in adminstratas else np.nan, axis = 1)
    # Create figure.
    fig, ax = plt.subplots(figsize = figsize)
    gdf.plot(column = "draw", ax = fig.gca(), cmap = cmap, edgecolor = "black", legend = False, alpha = 0.6,
             missing_kwds = {"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "Missing values"})
    if annotation:
        if annotation_selected:
            gdf = gdf[gdf.admin.isin(adminstratas)]
        for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf.admin):
            ax.annotate(label, xy = (x, y), xytext = (3, 3), textcoords = "offset points", color = "black")
    ax.set_title(country)
    plt.axis("off")
    if path_to_save is not None:
        fig.savefig(path_to_save, dpi = dpi, bbox_inches = "tight")
    plt.show()
    
def choropleth(country, quantiles, folder_to_shapefiles, figsize = (15, 10), annotation = False, cmap = "bwr",
               annotation_selected = False, path_to_save = None, dpi = 100):
    """
    This module allows to plot the choropleth map providing a the provincial names of the selected country and their corresponding 
    values with which the map will be colored.
    
    Parameters
    ----------
    country: a string parameter corresponding to the name of the country; e.g. "Yemen".
    quantiles: a pandas serie object with as index the provincial names and as values the corresponding values.
    folder_to_shapefiles: the path to reach the folder where the shapefiles are stored.
    figsize: the size of the figure.
    annotation: a boolean parameter to set if you want to visualize the names of all the provinces of the country.
    cmap: the cmap to use for drawing map.
    annotation_selected: a boolean parameter to set if you want to visualize only the names of the provinces provided.
    
    """
    # Load the dataframe of the country.
    gdf = gpd.read_file(folder_to_shapefiles + "/" + country + ".shp")
    
    # Define the provided adminstratas.
    adminstratas = quantiles.index
    
    # Draw only the adminstratas defined by the user.
    gdf["draw"] = gdf.apply(lambda x: quantiles.loc[x.admin] if x.admin in adminstratas else np.nan, axis = 1)
    # Create figure.
    f, ax = plt.subplots(1, figsize = figsize)
    ax = gdf.plot(column = "draw", ax = ax, cmap = cmap, edgecolor = "black", legend = True, alpha = 1, scheme = "quantiles", 
                  missing_kwds = {"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "Missing values"}, 
                  legend_kwds = dict(loc = "upper left", bbox_to_anchor = (1, 1)))
    if annotation:
        if annotation_selected:
            gdf = gdf[gdf.admin.isin(adminstratas)]
        for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf.admin):
            ax.annotate(label, xy = (x, y), xytext = (4, 4), textcoords = "offset points", color = "chocolate")

    ax.set_title(country)
    plt.axis("off")
    if path_to_save is not None:
        fig.savefig(path_to_save, dpi = dpi, bbox_inches = "tight")
    plt.show()
    