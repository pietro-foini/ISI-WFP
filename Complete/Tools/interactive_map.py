import geopandas as gpd
from ipyleaflet import Map, GeoData, basemaps, Popup
from ipywidgets import HTML, Output
from IPython.display import clear_output
    
def interactive_map(country, func_on_click, *args):
    
    attributes = {"Yemen": {"zoom": 6, "center": (15.8, 48.0)}, "Syria": {"zoom": 6, "center": (34.8, 38.0)}, 
                  "World": {"zoom": 3, "center": (20.8, 26.0)}, "Burkina Faso": {"zoom": 6, "center": (12.4, -1)}, 
                  "Nigeria": {"zoom": 6, "center": (9.3, 8)}}

    # Open the shapefile of the adminstrata of the Yemen country.
    gdf = gpd.read_file("./shape_files/" + country + "/" + country + ".shp")
    
    if country == "World" and args:
        states = args[0]
        gdf = gdf[gdf.admin.isin(states)]

    # Create default interactive map.
    map = Map(center = attributes[country]["center"], zoom = attributes[country]["zoom"], basemap = basemaps.CartoDB.Positron, 
              min_zoom = attributes[country]["zoom"], dragging = True)
    out = Output()
    
    # Country adminstrata interaction.
    geo_data = GeoData(geo_dataframe = gdf, name = "shapefile",
                       style = {"color": "black", "fillColor": "#EE983C", "opacity": 2, "weight": 1., "dashArray": "1", "fillOpacity": 0.8},
                       hover_style = {"fillColor": "#EE983C", "fillOpacity": .2})
    
    def click(properties = None, **args):
        with out:
            if properties is None:
                return
            clear_output()

            try:
                layer = next(x for x in map.layers if x.name == "Selected")
                map.remove_layer(layer)
            except:
                pass

            try:
                adminstrata = properties["admin"]

                geo_data_selected = GeoData(geo_dataframe = gdf[gdf.admin == adminstrata], name = "Selected", 
                                style = {"color": "#000D45", "fillColor": "rgba(0,0,0,0)", "opacity": 2, "weight": 3., "dashArray": "1", "fillOpacity": 0.8})
                map.add_layer(geo_data_selected)

                func_on_click(adminstrata)
            except:
                print("No time-series for this adminstrata!")

    def handle_interaction(**kwargs):
        if kwargs.get("type") == "mousemove":
            y, x = kwargs.get("coordinates")
            point = gpd.points_from_xy(x = [x], y = [y])[0]
            intersection = gdf.loc[gdf.intersects(point)]    

            try:
                layer = next(x for x in map.layers if x.name == "Infobox")
                prev_adminstrata = layer.class_name
            except:
                prev_adminstrata = None

            if not intersection.empty:
                adminstrata = intersection.admin.values[0]

            if not intersection.empty and prev_adminstrata == None: 
                message = HTML(value = "<h3><b>{}</b></h3><h4>Country: ".format(adminstrata) + country + "</h4>")
                x = intersection.centroid.x.values[0]
                y = intersection.centroid.y.values[0]
                popup = Popup(location = (y, x), child = message, close_button = False, name = "Infobox", class_name = adminstrata)
                map.add_layer(popup)
            elif (intersection.empty and prev_adminstrata != None) or (not intersection.empty and prev_adminstrata != None and prev_adminstrata != adminstrata):
                map.remove_layer(layer)

    map.on_interaction(handle_interaction)
    map.add_layer(geo_data)
    geo_data.on_click(click)
    
    return map, out