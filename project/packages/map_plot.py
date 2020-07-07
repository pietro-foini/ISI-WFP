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
    This module allows to plot the political boundaries of the selected country. More precisely, it allows to distinguish 
    all the provinces to those that are pass in the 'adminstratas' parameter.
    
    Parameters
    ----------
    country: a string corresponding to the name of the country.
    adminstratas: a list of provinces strings which are highlighted when viewing the map.
    folder_to_shapefiles: the path to reach the folder where the shapefiles are stored.
    figsize: the size of the map figure.
    annotation: a boolean parameter to set if you want to visualize the names of the adminstratas.
    annotation_selected: a boolean parameter to set if you want to visualize only the names of the adminstratas selected.
    
    """
    # Load the dataframe of the country.
    gdf = gpd.read_file(folder_to_shapefiles + "/" + country + "/" + country + ".shp")
    
    # Consider or not some provinces/sub-provinces of the Nigeria country.
    if country == "Nigeria" and any(admin in ["Adamawa Central", "Adamawa North", "Adamawa South", "Borno Central", "Borno North", "Borno South", "Yobe East", "Yobe North", "Yobe South"] for admin in adminstratas):
        gdf = gdf[~gdf.admin.isin(["Adamawa", "Borno", "Yobe"])]
    elif country == "Nigeria" and not any(admin in ["Adamawa Central", "Adamawa North", "Adamawa South", "Borno Central", "Borno North", "Borno South", "Yobe East", "Yobe North", "Yobe South"] for admin in adminstratas):
        gdf = gdf[~gdf.admin.isin(["Adamawa Central", "Adamawa North", "Adamawa South", "Borno Central", "Borno North", "Borno South", "Yobe East", "Yobe North", "Yobe South"])]       
    
    # Select only the adminstratas defined by the user.
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
    