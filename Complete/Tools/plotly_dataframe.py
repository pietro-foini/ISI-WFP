import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from ipywidgets import interact, widgets, fixed
from IPython.display import display

# Plot dataframe with multi-columns index using plotly.
def plot(df, title = None, yaxis = None, style = "lines", first_last_valid_index_group = False):
    
    def subplot_level4(name1, name2, name3, graph, df):
        
        if name1 != None and name2 != None and name3 != None and name2 in df[name1].columns.get_level_values(0).unique() and name3 in df[name1][name2].columns.get_level_values(0).unique():
            group = df[name1][name2][name3]

            if first_last_valid_index_group:
                # Adjust time-series group.
                first_idx = group.first_valid_index()
                last_idx = group.last_valid_index()
                group = group.loc[first_idx:last_idx]

            # Create figure.
            if graph == "Time-series":
                # Set default trace colors with colorway.
                colorway = sns.color_palette("hls", 8).as_hex()
                layout = go.Layout(colorway = colorway)

                fig = go.Figure(layout = layout)

                for column in group.columns:
                    fig.add_trace(go.Scatter(x = group.index, y = group[column], name = column, mode = style, 
                                             showlegend = True, line = dict(width = 1.5)))

                # Edit the layout of the y-axis.
                fig.update_layout(yaxis_title = dict(text = yaxis, font = dict(size = 10)))
                # Edit the layout of the title.
                if title:
                    title_name = title + " - " + name1 + " - " + name2 + " - " + name3
                else:
                    title_name = name1 + " - " + name2 + " - " + name3
                fig.update_layout(title = dict(text = title_name, y = 0.9, x = 0.5))
                # Add range slider.
                fig.update_layout(xaxis = dict(title = "Datetime", rangeselector = dict(), rangeslider = dict(visible = True), 
                                               type = "date"))
                fig.show()
            if graph == "Missing values":  
                # Visualization of the missing values of the current time-series.
                miss = group.notnull().astype("int")
                fig = go.Figure(data = go.Heatmap(z = miss, zmin = 0, zmax = 1, x = miss.columns, y = miss.index, xgap = 3, 
                                                  colorscale = [[0, "white"], [1, "black"]], showscale = False, 
                                                  hoverinfo = "x+y"))
                fig.show()
        else:
            pass
    
    def subplot_level3(name1, name2, graph, df):

        if name1 != None and name2 != None and name2 in df[name1].columns.get_level_values(0).unique():
            group = df[name1][name2]

            if first_last_valid_index_group:
                # Adjust time-series group.
                first_idx = group.first_valid_index()
                last_idx = group.last_valid_index()
                group = group.loc[first_idx:last_idx]

            # Create figure.
            if graph == "Time-series":
                # Set default trace colors with colorway.
                colorway = sns.color_palette("hls", 8).as_hex()
                layout = go.Layout(colorway = colorway)

                fig = go.Figure(layout = layout)

                for column in group.columns:
                    fig.add_trace(go.Scatter(x = group.index, y = group[column], name = column, mode = style, 
                                             showlegend = True, line = dict(width = 1.5)))

                # Edit the layout of the y-axis.
                fig.update_layout(yaxis_title = dict(text = yaxis, font = dict(size = 10)))
                # Edit the layout of the title.
                if title:
                    title_name = title + " - " + name1 + " - " + name2
                else:
                    title_name = name1 + " - " + name2
                fig.update_layout(title = dict(text = title_name, y = 0.9, x = 0.5))
                # Add range slider.
                fig.update_layout(xaxis = dict(title = "Datetime", rangeselector = dict(), rangeslider = dict(visible = True), 
                                               type = "date"))
                fig.show()
            if graph == "Missing values":  
                # Visualization of the missing values of the current time-series.
                miss = group.notnull().astype("int")
                fig = go.Figure(data = go.Heatmap(z = miss, zmin = 0, zmax = 1, x = miss.columns, y = miss.index, xgap = 3, 
                                                  colorscale = [[0, "white"], [1, "black"]], showscale = False, 
                                                  hoverinfo = "x+y"))
                fig.show()
        else:
            pass
    
    def subplot_level2(name, graph, df):
        group = df[name]
        
        if first_last_valid_index_group:
            # Adjust time-series group.
            first_idx = group.first_valid_index()
            last_idx = group.last_valid_index()
            group = group.loc[first_idx:last_idx]
            
        # Create figure.
        if graph == "Time-series":
            # Set default trace colors with colorway.
            colorway = sns.color_palette("hls", 8).as_hex()
            layout = go.Layout(colorway = colorway)

            fig = go.Figure(layout = layout)

            for column in group.columns:
                fig.add_trace(go.Scatter(x = group.index, y = group[column], name = column, mode = style, 
                                         showlegend = True, line = dict(width = 1.5)))

            # Edit the layout of the y-axis.
            fig.update_layout(yaxis_title = dict(text = yaxis, font = dict(size = 10)))
            # Edit the layout of the title.
            if title:
                title_name = title + " - " + name
            else:
                title_name = name
            fig.update_layout(title = dict(text = title_name, y = 0.9, x = 0.5))
            # Add range slider.
            fig.update_layout(xaxis = dict(title = "Datetime", rangeselector = dict(), rangeslider = dict(visible = True), 
                                           type = "date"))

            fig.show()
        if graph == "Missing values":    
            # Visualization of the missing values of the current time-series.
            miss = group.notnull().astype("int")
            fig = go.Figure(data = go.Heatmap(z = miss, zmin = 0, zmax = 1, x = miss.columns, y = miss.index, xgap = 3, 
                                              colorscale = [[0, "white"], [1, "black"]], showscale = False, 
                                              hoverinfo = "x+y"))
            fig.show()
            
    def subplot_level1(graph, df):
        # Create figure.
        if graph == "Time-series":
            # Set default trace colors with colorway.
            colorway = sns.color_palette("hls", 8).as_hex()
            layout = go.Layout(colorway = colorway)

            fig = go.Figure(layout = layout)

            for column in df.columns:
                fig.add_trace(go.Scatter(x = df.index, y = df[column], name = column, mode = style, 
                                         showlegend = True, line = dict(width = 1.5)))

            # Edit the layout of the y-axis.
            fig.update_layout(yaxis_title = dict(text = yaxis, font = dict(size = 10)))
            # Edit the layout of the title.
            fig.update_layout(title = dict(text = title, y = 0.9, x = 0.5))
            # Add range slider.
            fig.update_layout(xaxis = dict(title = "Datetime", rangeselector = dict(), rangeslider = dict(visible = True), 
                                           type = "date"))
            fig.show()
        if graph == "Missing values":  
            # Visualization of the missing values of the current time-series.
            miss = df.notnull().astype("int")
            fig = go.Figure(data = go.Heatmap(z = miss, zmin = 0, zmax = 1, x = miss.columns, y = miss.index, xgap = 3, 
                                              colorscale = [[0, "white"], [1, "black"]], showscale = False, hoverinfo = "x+y"))
            fig.show()
    
    # Check if dataframe not contains multi-columns.
    if not isinstance(df.columns, pd.core.index.MultiIndex):
        # Create figure.
        w1 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
        p = interact(subplot_level1, graph = w1, df = fixed(df))
    else:
        # Check if dataframe has 2 levels of multi-columns.
        if len(df.columns.levels) == 2:
            if len(df.columns.get_level_values(1).unique()) == 1:
                df = df.droplevel(level = 1, axis = 1)
                # Create figure.
                w1 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                p = interact(subplot_level1, graph = w1, df = fixed(df))
            elif len(df.columns.get_level_values(0).unique()) == 1 and len(df.columns.get_level_values(1).unique()) != 1:
                w1 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                p = interact(subplot_level2, name = fixed(df.columns.get_level_values(0).unique().values[0]), graph = w1, df = fixed(df))
            elif len(df.columns.get_level_values(0).unique()) != 1 and len(df.columns.get_level_values(1).unique()) != 1: 
                # Create figure.
                w1 = widgets.ToggleButtons(options = df.columns.get_level_values(0).unique(), 
                                           description = df.columns.get_level_values(0).name, 
                                           disabled = False)
                w2 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                p = interact(subplot_level2, name = w1, graph = w2, df = fixed(df))
        # Check if dataframe has 3 levels of multi-columns.
        elif len(df.columns.levels) == 3:
            if len(df.columns.get_level_values(2).unique()) == 1:
                df = df.droplevel(level = 2, axis = 1)
                # Create figure.
                w1 = widgets.ToggleButtons(options = df.columns.get_level_values(0).unique(), 
                                           description = df.columns.get_level_values(0).name, 
                                           disabled = False)
                w2 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                p = interact(subplot_level2, name = w1, graph = w2, df = fixed(df))
            else:
                # Create figure.
                w1 = widgets.Dropdown(options = df.columns.get_level_values(0).unique(), description = df.columns.get_level_values(0).name, 
                                      disabled = False, value = None)
                w2 = widgets.Dropdown(description = df.columns.get_level_values(1).name, disabled = False)

                # Define a function that updates the content of w2 based on what we select for w1.
                def update(*args):
                    if w1.value:
                        w2.options = df[w1.value].columns.get_level_values(0).unique()
                w1.observe(update)
                  
                w3 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                hbox = widgets.HBox([w1, w2])
                out = widgets.interactive_output(subplot_level3, {"name1": w1, "name2": w2, "graph": w3, "df": fixed(df)})
                display(hbox, w3, out)
        # Check if dataframe has 4 levels of multi-columns.
        elif len(df.columns.levels) == 4:
                # Create figure.
                w1 = widgets.Dropdown(options = df.columns.get_level_values(0).unique(), description = df.columns.get_level_values(0).name, 
                                      disabled = False, value = None)
                w2 = widgets.Dropdown(description = df.columns.get_level_values(1).name, disabled = False)
                w3 = widgets.Dropdown(description = df.columns.get_level_values(2).name, disabled = False)

                # Define a function that updates the content of w2 based on what we select for w1.
                def update(*args):
                    if w1.value:
                        w2.options = df[w1.value].columns.get_level_values(0).unique()
                        w3.options = df[w1.value][w2.value].columns.get_level_values(0).unique()
                w1.observe(update)
                  
                w4 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                hbox = widgets.HBox([w1, w2, w3])
                out = widgets.interactive_output(subplot_level4, {"name1": w1, "name2": w2, "name3": w3, "graph": w4, "df": fixed(df)})
                display(hbox, w4, out)
                
def plot_comparison(df1, df2, title = None, yaxis = None, first_last_valid_index_group = False):
    
    def sub_plot(country):
        subdf1 = df1[country]
        subdf2 = df2[country]
        
        if first_last_valid_index_group:
            # Adjust time-series group.
            first_idx = subdf1.first_valid_index()
            last_idx = subdf1.last_valid_index()
            subdf1 = subdf1.loc[first_idx:last_idx]
            # Adjust time-series group.
            first_idx = subdf2.first_valid_index()
            last_idx = subdf2.last_valid_index()
            subdf2 = subdf2.loc[first_idx:last_idx]

        # Create figure.
        fig = go.Figure()

        colorway = sns.color_palette("hls", 8).as_hex()

        subdf1.columns = subdf1.columns.droplevel(1)
        subdf2.columns = subdf2.columns.droplevel(1)

        for i, col in enumerate(subdf1.columns):
            fig.add_trace(go.Scatter(x = subdf1.index, y = subdf1[col], name = col, line = dict(width = 1.5, 
                                                                                                color = colorway[i % len(colorway)]), 
                                     legendgroup = col))
            fig.add_trace(go.Scatter(x = subdf2.index, y = subdf2[col], name = col, line = dict(width = 1., 
                                                                                                color = colorway[i % len(colorway)]), 
                                     showlegend = False, legendgroup = col))

        # Edit the layout.
        if title:
            title_name = title + " - " + country
        else:
            title_name = country
        fig.update_layout(title = dict(text = title_name, y = 0.9, x = 0.5), 
                          yaxis_title = dict(text = yaxis, font = dict(size = 10)))
        # Add range slider.
        fig.update_layout(xaxis = dict(title = "Datetime", rangeselector = dict(), rangeslider = dict(visible = True), 
                                       type = "date"))

        fig.show()

    w = widgets.ToggleButtons(options = list(df1.columns.levels[0]), description = df1.columns.get_level_values(0).name, disabled = False)
    p = interact(sub_plot, country = w)
    
def plot_hist(df, title = None, yaxis = None):
    
    def sub_plot_hist(name1, name2, df):
        if name1 != None and name2 != None and name2 in df[name1].columns.get_level_values(0).unique():  
            df = df.droplevel(level = 2, axis = 1)
            group = df[name1][name2]
            group = group.dropna()

            fig = go.Figure()
            fig.add_trace(go.Histogram(histfunc = "avg", y = group, x = group.index, nbinsx = len(group) - 1, 
                                       marker_color = "#330C73", opacity = 0.75))
            fig.update_layout(go.Layout(bargap = 0.1))
            # Edit the layout of the y-axis.
            fig.update_layout(yaxis_title = dict(text = yaxis, font = dict(size = 10)))
            # Edit the layout of the title.
            if title:
                title_name = title + " - " + name1 + " - " + name2 
            else:
                title_name = name1 + " - " + name2 
            fig.update_layout(title = dict(text = title_name, y = 0.9, x = 0.5))
            # Add range slider.
            fig.update_layout(xaxis = dict(title = "Datetime", rangeselector = dict(), rangeslider = dict(visible = True), 
                                           type = "date"))
            fig.show()
        else:
            pass

    # Create figure.
    w1 = widgets.Dropdown(options = df.columns.get_level_values(0).unique(), description = df.columns.get_level_values(0).name, 
                          disabled = False, value = None)
    w2 = widgets.Dropdown(description = df.columns.get_level_values(1).name, disabled = False)

    # Define a function that updates the content of w2 based on what we select for w1.
    def update(*args):
        if w1.value:
            w2.options = df[w1.value].columns.get_level_values(0).unique()
    w1.observe(update)

    hbox = widgets.HBox([w1, w2])
    out = widgets.interactive_output(sub_plot_hist, {"name1": w1, "name2": w2, "df": fixed(df)})
    display(hbox, out)