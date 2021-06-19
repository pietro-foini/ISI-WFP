from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings("ignore", "The GeoSeries you are attempting", UserWarning)
plt.style.use("default")

# Python module.
#
#
# Pietro Foini
#
# Year: 2020
    
def draw_adminstratas(country, adminstratas, folder_to_shapefiles, adminstratas_to_highlight = None, figsize = (10, 7), 
                      annotation = False, annotation_selected = False, path_to_save = None, dpi = 100):
    """
    This module allows to plot the political boundaries of the selected country highlighting the administrative regions 
    provided by the users.
    
    Parameters
    ----------
    country: a string parameter corresponding to the name of the country.
    adminstratas: a list of the administrative region names that will be colored. If some regions are missing, they will be 
       highlighted with a specific pattern.
    folder_to_shapefiles: the path to reach the folder where the shapefiles of the country are stored.
    adminstratas_to_highlight: a sublist of the administrative region names (they must be contained in the 'adminstratas' parameter)
       that will be colored when viewing the map using another color.    
    figsize: the size of the map.
    annotation: a boolean parameter to set if you want to visualize all the administartive region names of the country.
    annotation_selected: a boolean parameter to set if you want to visualize only the adminstartive region names provided into the
       'adminstratas_to_highlight' parameter.
    path_to_save: the path where to save the map. If 'None', tha map is not saved.
    dpi: resolution of the saved figure.
    
    """
    # Check parameters.
    if adminstratas_to_highlight is not None:
        if not set(adminstratas_to_highlight).issubset(set(adminstratas)):
            raise ValueError("The parameter 'adminstratas_to_highlight' must be a subset of the parameter 'adminstratas'.")
    
    # Define cmap.
    cmap = LinearSegmentedColormap.from_list("cmap", [sns.color_palette("pastel")[1], sns.color_palette("tab10")[1]], N = 2)
    
    # Load the dataframe of the country.
    gdf = gpd.read_file(folder_to_shapefiles)

    # Draw only the adminstratas defined by the user.
    def assign_color(x):
        if x["region"] in adminstratas:
            if adminstratas_to_highlight is not None:
                if x["region"] in adminstratas_to_highlight:
                    return 1
                else:
                    return 0
            else:
                return 1
        else:
            return np.nan

    gdf["draw"] = gdf.apply(assign_color, axis = 1)

    # Create figure.
    fig, ax = plt.subplots(figsize = figsize)
    gdf.plot(column = "draw", ax = fig.gca(), cmap = cmap, edgecolor = "grey", legend = False, alpha = 1.,
             missing_kwds = {"color": "#b7ada7", "edgecolor": "grey", "hatch": "///", "label": "Missing values"})
    if annotation:
        if annotation_selected:
            gdf = gdf[gdf["region"].isin(adminstratas)]
        for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf["region"]):
            ax.text(x, y, label, ha = "center", size = 10, path_effects = [pe.withStroke(linewidth = 2, foreground = "w")])
    plt.axis("off")
    plt.show()
    # Save the figure.
    if path_to_save is not None:
        fig.savefig(path_to_save, dpi = dpi, bbox_inches = "tight")
    
def choropleth(country, quantiles, folder_to_shapefiles, figsize = (15, 10), annotation = False, cmap = "bwr",
               annotation_selected = False, path_to_save = None, dpi = 100):
    """
    This module allows to plot the choropleth map providing a the provincial names of the selected country and their corresponding 
    values with which the map will be colored.
    
    Parameters
    ----------
    country: a string parameter corresponding to the name of the country.
    quantiles: a pandas serie object with as index the provincial names and as values the corresponding values.
    folder_to_shapefiles: the path to reach the folder where the shapefiles are stored.
    figsize: the size of the figure.
    annotation: a boolean parameter to set if you want to visualize the names of all the provinces of the country.
    cmap: the cmap to use for drawing map.
    annotation_selected: a boolean parameter to set if you want to visualize only the names of the provinces provided.
    path_to_save: the path where to save the map. If 'None', tha map is not saved.
    dpi: resolution of the saved figure.
    
    """
    # Load the dataframe of the country.
    gdf = gpd.read_file(folder_to_shapefiles)
    
    # Define the provided adminstratas.
    adminstratas = quantiles.index
    
    # Draw only the adminstratas defined by the user.
    gdf["draw"] = gdf.apply(lambda x: quantiles.loc[x["region"]] if x["region"] in adminstratas else np.nan, axis = 1)
    # Create figure.
    f, ax = plt.subplots(figsize = figsize)
    ax = gdf.plot(column = "draw", ax = ax, cmap = cmap, edgecolor = "black", legend = True, alpha = 1, scheme = "quantiles", 
                  missing_kwds = {"color": "#b7ada7", "edgecolor": "grey", "hatch": "///", "label": "Missing values"}, 
                  legend_kwds = dict(loc = "upper left", bbox_to_anchor = (1, 1)))
    if annotation:
        if annotation_selected:
            gdf = gdf[gdf["region"].isin(adminstratas)]
        for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf["region"]):
            ax.text(x, y, label, ha = "center", size = 10, path_effects = [pe.withStroke(linewidth = 2, foreground = "w")])

    ax.set_title(country)
    plt.axis("off")
    plt.show()
    # Save the figure.
    if path_to_save is not None:
        fig.savefig(path_to_save, dpi = dpi, bbox_inches = "tight")
    
    