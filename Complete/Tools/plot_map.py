import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
    
def draw_admin(country, adminstratas, figsize = (15, 10), annotation = False, annotation_selected = False):
    gdf = gpd.read_file("./shape_files/" + country + "/" + country + ".shp")
    
    if country == "World":
        gdf.rename(columns = {"country": "admin"}, inplace = True)
    
    if country == "Nigeria" and any(admin in ["Adamawa Central", "Adamawa North", "Adamawa South", "Borno Central", "Borno North", "Borno South", "Yobe East", "Yobe North", "Yobe South"] for admin in adminstratas):
        gdf = gdf[~gdf.admin.isin(["Adamawa", "Borno", "Yobe"])]
    elif country == "Nigeria" and not any(admin in ["Adamawa Central", "Adamawa North", "Adamawa South", "Borno Central", "Borno North", "Borno South", "Yobe East", "Yobe North", "Yobe South"] for admin in adminstratas):
        gdf = gdf[~gdf.admin.isin(["Adamawa Central", "Adamawa North", "Adamawa South", "Borno Central", "Borno North", "Borno South", "Yobe East", "Yobe North", "Yobe South"])]       
    
    gdf["draw"] = gdf.apply(lambda x: 1 if x.admin in adminstratas else np.nan, axis = 1)
    f, ax = plt.subplots(1, figsize = figsize)
    ax = gdf.plot(column = "draw", ax = ax, cmap = "bwr", edgecolor = "black", legend = False, alpha = 0.6,
                  missing_kwds = {"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "Missing values"})
    if annotation:
        if annotation_selected:
            gdf = gdf[gdf.admin.isin(adminstratas)]
        for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf.admin):
            ax.annotate(label, xy = (x, y), xytext = (4, 4), textcoords = "offset points", color = "black")
    plt.title(country)
    limit = plt.axis("off")
    
def choropleth(country, adminstratas, quantiles, figsize = (15, 10), annotation = False, annotation_selected = False):
    gdf = gpd.read_file("./shape_files/" + country + "/" + country + ".shp")
    
    if country == "World":
        gdf.rename(columns = {"country": "admin"}, inplace = True)
    
    if country == "Nigeria" and any(admin in ["Adamawa Central", "Adamawa North", "Adamawa South", "Borno Central", "Borno North", "Borno South", "Yobe East", "Yobe North", "Yobe South"] for admin in adminstratas):
        gdf = gdf[~gdf.admin.isin(["Adamawa", "Borno", "Yobe"])]
    elif country == "Nigeria" and not any(admin in ["Adamawa Central", "Adamawa North", "Adamawa South", "Borno Central", "Borno North", "Borno South", "Yobe East", "Yobe North", "Yobe South"] for admin in adminstratas):
        gdf = gdf[~gdf.admin.isin(["Adamawa Central", "Adamawa North", "Adamawa South", "Borno Central", "Borno North", "Borno South", "Yobe East", "Yobe North", "Yobe South"])]       
    
    gdf["draw"] = gdf.apply(lambda x: quantiles.loc[x.admin] if x.admin in adminstratas else np.nan, axis = 1)
    f, ax = plt.subplots(1, figsize = figsize)
    ax = gdf.plot(column = "draw", ax = ax, edgecolor = "black", legend = True, alpha = 1, scheme = "quantiles", 
                  missing_kwds = {"color": "lightgrey", "edgecolor": "red", "hatch": "///", "label": "Missing values"})
    if annotation:
        if annotation_selected:
            gdf = gdf[gdf.admin.isin(adminstratas)]
        for x, y, label in zip(gdf.centroid.x, gdf.centroid.y, gdf.admin):
            ax.annotate(label, xy = (x, y), xytext = (4, 4), textcoords = "offset points", color = "chocolate")
    plt.title(country)
    limit = plt.axis("off")
    
