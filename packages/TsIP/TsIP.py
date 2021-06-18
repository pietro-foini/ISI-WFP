import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, widgets, fixed
from IPython.display import display
plt.style.use("default")

# Python module.
#
#
# Pietro Foini
#
# Year: 2020

class TsIP:
    """TsIP (Time-series Interactive Plot).
    
    TsIP is a python library developed to interactively visualize multiple time-series quickly and easily. 
    The implementation of this tool responds to the need to visualize time-series stored into a pandas dataframe with
    hierarchical multi-index on axis 1, taking advantage of dynamic user interaction.
    The time-series are plotted using either the matplotlib library or the plotly library (user will). For a more 
    involving navigation within the time-series visualization, it is advice to use the plotly library rather than the
    matplotlib library. 
    
    """
    
    def __init__(self, df, df2 = None):
        """
        ***Initialization function***
 
        Initialization of the TsIP class.
        
        Parameters
        ----------
        df: a pandas dataframe with hierarchical multi-index on axis 1 where the time-series are stored. The dataframe must 
           have as index a single pandas datetime column with an appropriate frequency set. The dataframe can have 
           from a single level up to 4 levels on axis 1. Moreover, all the levels of the dataframe must have set a name.
        df2: a pandas dataframe completely equal to df except for the values inside it in order to compare the time-series.
           
       """
        # Define the dataframe.
        self.df = df
        self.df2 = df2
        
    def plot_df_level_1(self, graph, df, title_name = None, df2 = None):
        """
        ***Sub-plot function***
        
        This sub-function allows to plot the time-series stored into a pandas dataframe with a single level on axis 1.
        
        Parameters
        ----------
        graph: a string parameter to set between 'Time-series' and 'Missing values'. If 'Time-series' is set, the time-series
           stored into the dataframe are visualized. If 'Missing values' is set, the heatmap regarding the nan values of all
           the time-series is visualized.
           Being this function built to depend on other functions, the parameter 'graph' is to consider interactive, i.e. the 
           user can switch from one value to another thanks to a widget button.
        df: the pandas dataframe with a single level on axis 1.
        title_name: the title to set for the figures generated.
        df2: the second pandas dataframe with a single level on axis 1.
        
        """
        # Set default trace colors with colorway of the time-series.
        colorway = sns.color_palette("tab10", 8).as_hex()
        #colorway = sns.color_palette("hls", 8).as_hex()

        # Visualization using Matplotlib library.
        if self.matplotlib:
            # Plot of the time-series.
            if graph == "Time-series":
                # Create figure.
                fig, ax = plt.subplots(figsize = (15, 5))
                # Set colorway of the time-series.
                ax.set_prop_cycle("color", colorway)
                
                # Define the style for matplotlib library.
                if self.style == "lines+markers":
                    style = ".-"
                elif self.style == "lines":
                    style = "-"  
                elif self.style == "mix":
                    style = "mix" 

                if self.comparison:
                    # Add the time-series to the figure.
                    for i, column in enumerate(df.columns):
                        df[column].plot(ax = fig.gca(), label = column, style = style, color = "red")
                    for i, column in enumerate(df2.columns):   
                        df2[column].plot(ax = fig.gca(), label = column, style = style, color = "blue")
                else:
                    # Add the time-series to the figure.
                    for column in df.columns:  
                        if style != "mix":
                            df[column].plot(ax = fig.gca(), label = column, style = style)
                        else:
                            if df[column].loc[df[column].first_valid_index():df[column].last_valid_index()].isna().sum() > 0:
                                df[column].dropna(how = "all").plot(ax = fig.gca(), label = column, style = ".-")
                            else:
                                df[column].plot(ax = fig.gca(), label = column, style = "-")

                # Set legend.
                ax.legend(title = df.columns.name, loc = "center left", bbox_to_anchor = (1.0, 0.5))
                # Set axis names.
                ax.set_ylabel(self.yaxis, fontsize = 15)
                ax.set_xlabel("Datetime", fontsize = 15)
                # Set title.
                ax.set_title(title_name, fontsize = 18)
                ax.tick_params(labelsize = 15)
                ax.autoscale()
                plt.show()
            # Plot of the missing values.
            if graph == "Missing values":  
                # Create figure.
                fig, ax = plt.subplots(figsize = (15, 5))
                # Visualization of the missing values of the current time-series.
                miss = df.notnull().astype("int")
                # Create heatmap.
                sns.heatmap(miss, ax = fig.gca(), vmin = 0, vmax = 1, cmap = sns.color_palette(["white", "black"]), cbar = False)
                # Add an x gap between the columns.
                for i in range(miss.shape[1] + 1):
                    ax.axvline(i, color = "white", lw = 3)
                plt.show()
        # Visualization using Plotly library.
        else:
            # Plot of the time-series.
            if graph == "Time-series":
                # Create figure.
                fig = go.Figure(layout = go.Layout(colorway = colorway))

                if self.comparison:
                     # Add the time-series to the figure.
                    for i, column in enumerate(df.columns):
                        fig.add_trace(go.Scatter(x = df.index, y = df[column], name = column, line = dict(width = 1.5, color = "red")))
                    for i, column in enumerate(df2.columns):   
                        fig.add_trace(go.Scatter(x = df2.index, y = df2[column], name = column, line = dict(width = 1., color = "blue")))
                else:
                    # Add the time-series to the figure.
                    for column in df.columns:
                        if self.style != "mix":
                            fig.add_trace(go.Scatter(x = df.index, y = df[column], name = column, mode = self.style, 
                                                     showlegend = True, line = dict(width = 1.5)))
                        else:
                            if df[column].loc[df[column].first_valid_index():df[column].last_valid_index()].isna().sum() > 0:
                                fig.add_trace(go.Scatter(x = df[column].dropna(how = "all").index, y = df[column].dropna(how = "all"), 
                                                         name = column, mode = "lines+markers", showlegend = True, 
                                                         line = dict(width = 1.5)))
                            else:
                                fig.add_trace(go.Scatter(x = df.index, y = df[column], name = column, mode = "lines", 
                                                         showlegend = True, line = dict(width = 1.5)))

                # Edit the layout of the y-axis.
                fig.update_layout(yaxis_title = dict(text = self.yaxis, font = dict(size = 10)))
                # Edit the layout of the title.
                fig.update_layout(title = dict(text = title_name, y = 0.9, x = 0.5))
                # Add range slider on x axis.
                fig.update_layout(xaxis = dict(title = "Datetime", rangeselector = dict(), rangeslider = dict(visible = True), 
                                               type = "date"))
                fig.show()
            if graph == "Missing values":  
                # Visualization of the missing values of the current time-series.
                miss = df.notnull().astype("int")
                # Create figure.
                fig = go.Figure(data = go.Heatmap(z = miss, zmin = 0, zmax = 1, x = miss.columns, y = miss.index, xgap = 3, 
                                                  colorscale = [[0, "white"], [1, "black"]], showscale = False, hoverinfo = "x+y"))
                fig.show()
                
    def plot_df_level_2(self, name, graph, df, df2 = None):
        """
        ***Sub-plot function***
        
        This sub-function allows to plot the time-series stored into a pandas dataframe with 2 level on axis 1. More precisely,
        the functionality of this visualization is dependent on dynamic user interaction.
        
        Parameters
        ----------
        name: the name of the sub-dataframe on level 0 to plot. This choice is dynamic thanks to an interactive button in
           the main function.
        graph: a string parameter to set between 'Time-series' and 'Missing values'. If 'Time-series' is set, the time-series
           stored into the dataframe are visualized. If 'Missing values' is set, the heatmap regarding the nan values of all
           the time-series is visualized.
           Being this function built to depend on other functions, the parameter 'graph' is to consider interactive, i.e. the 
           user can switch from one value to another thanks to a widget button.
        df: the pandas dataframe with 2 levels on axis 1.
        df2: the second pandas dataframe with a single level on axis 1.
        
        """      
        # Select the subdataframe thanks to an interactive button.
        group = df[name]

        # Select sub-dataframe between its first and last valid values.
        if self.first_last_valid_index_group:
            # Adjust time-series group.
            first_idx = group.first_valid_index()
            last_idx = group.last_valid_index()
            group = group.loc[first_idx:last_idx]
            
        if self.comparison:
            group2 = df2[name]
            if self.first_last_valid_index_group:
                # Adjust time-series group.
                first_idx = group2.first_valid_index()
                last_idx = group2.last_valid_index()
                group2 = group2.loc[first_idx:last_idx]
        else:
            group2 = None
        
        # Define title.
        if self.title:
            title_name = self.title + " - " + str(name)
        else:
            title_name = str(name)
        
        # Visualization of the time-series.
        self.plot_df_level_1(graph, group, title_name, group2)
        
                
    def plot_df_level_3(self, name1, name2, graph, df, df2 = None):
        """
        ***Sub-plot function***
        
        This sub-function allows to plot the time-series stored into a pandas dataframe with 3 level on axis 1. More precisely, 
        the functionality of this visualization is dependent on dynamic user interaction.
        
        Parameters
        ----------
        name1: the name of the sub-dataframe on level 0 to plot. This choice is dynamic thanks to an interactive button in 
           the main function.
        name2: the name of the sub-dataframe on level 1 to plot. This choice is dynamic thanks to an interactive button in
           the main function.
        graph: a string parameter to set between 'Time-series' and 'Missing values'. If 'Time-series' is set, the time-series
           stored into the dataframe are visualized. If 'Missing values' is set, the heatmap regarding the nan values of all
           the time-series is visualized.
           Being this function built to depend on other functions, the parameter 'graph' is to consider interactive, i.e. the 
           user can switch from one value to another thanks to a widget button.
        df: the pandas dataframe with 3 levels on axis 1.
        df2: the second pandas dataframe with a single level on axis 1.
        
        """
        # Select the subdataframe.
        if name1 != None and name2 != None and name2 in df[name1].columns.get_level_values(0).unique():
            group = df[name1][name2]

            # Select subdataframe between its first and last valid values.
            if self.first_last_valid_index_group:
                # Adjust time-series group.
                first_idx = group.first_valid_index()
                last_idx = group.last_valid_index()
                group = group.loc[first_idx:last_idx]
                
            if self.comparison:
                group2 = df2[name1][name2]
                if self.first_last_valid_index_group:
                    # Adjust time-series group.
                    first_idx = group2.first_valid_index()
                    last_idx = group2.last_valid_index()
                    group2 = group2.loc[first_idx:last_idx]
            else:
                group2 = None
            
            # Define title.
            if self.title:
                title_name = self.title + " - " + str(name1) + " - " + str(name2)
            else:
                title_name = str(name1) + " - " + str(name2)

            # Visualization of the time-series.
            self.plot_df_level_1(graph, group, title_name, group2)
            
        else:
            pass
        
    def plot_df_level_4(self, name1, name2, name3, graph, df, df2 = None):
        """
        ***Sub-plot function***
        
        This sub-function allows to plot the time-series stored into a pandas dataframe with 4 level on axis 1. More precisely,
        the functionality of this visualization is dependent on dynamic user interaction.
        
        Parameters
        ----------
        name1: the name of the sub-dataframe on level 0 to plot. This choice is dynamic thanks to an interactive button in 
           the main function.
        name2: the name of the sub-dataframe on level 1 to plot. This choice is dynamic thanks to an interactive button in 
           the main function.
        name3: the name of the sub-dataframe on level 2 to plot. This choice is dynamic thanks to an interactive button in 
           the main function.
        graph: a string parameter to set between 'Time-series' and 'Missing values'. If 'Time-series' is set, the time-series
           stored into the dataframe are visualized. If 'Missing values' is set, the heatmap regarding the nan values of all
           the time-series is visualized.
           Being this function built to depend on other functions, the parameter 'graph' is to consider interactive, i.e. the 
           user can switch from one value to another thanks to a widget button.
        df: the pandas dataframe with 4 levels on axis 1.
        df2: the second pandas dataframe with a single level on axis 1.
        
        """
        # Select the subdataframe.
        if name1 != None and name2 != None and name3 != None and name2 in df[name1].columns.get_level_values(0).unique() and name3 in df[name1][name2].columns.get_level_values(0).unique():
            group = df[name1][name2][name3]
            
            if self.first_last_valid_index_group:
                # Adjust time-series group.
                first_idx = group.first_valid_index()
                last_idx = group.last_valid_index()
                group = group.loc[first_idx:last_idx]
                
            if self.comparison:
                group2 = df2[name1][name2][name3]
                if self.first_last_valid_index_group:
                    # Adjust time-series group.
                    first_idx = group2.first_valid_index()
                    last_idx = group2.last_valid_index()
                    group2 = group2.loc[first_idx:last_idx]
            else:
                group2 = None
            
            # Define title.
            if self.title:
                title_name = self.title + " - " + str(name1) + " - " + str(name2) + " - " + str(name3)
            else:
                title_name = str(name1) + " - " + str(name2) + " - " + str(name3)
                
            # Visualization of the time-series.
            self.plot_df_level_1(graph, group, title_name, group2)
            
        else:
            pass
                
    def interactive_plot_df(self, title = None, yaxis = None, style = "lines", first_last_valid_index_group = False, 
                            matplotlib = False, comparison = False, normalization = False):
        """
        ***Main function***
        
        This main function allows to interactively plot the time-series stored into a multi-index columns dataframe. It 
        has the potential to manage dataframes that can have from a single level up to 4 levels on axis 1.
        
        Parameters
        ----------
        title: the title to add to the figures. 
        yaxis: a string value to add on y axis.
        style: the style of the plots. It can be 'lines', 'lines+markers' or 'mix'.
        first_last_valid_index_group: if you want to plt the time-series groups using as reference datetime only the values
           between their first and last valid indeces and not using the entire datetime object of the dataframe.
        matplotlib: if you want to use matplotlib (True) or plotly (False) library to visualize the time-series.
        comparison: if you want to compare the time-series of two equal hierarchical dataframes (df and df2).
           
       """
        # Define the parameters as attributes of the class.
        self.title = title
        self.yaxis = yaxis
        self.style = style
        self.first_last_valid_index_group = first_last_valid_index_group
        self.matplotlib = matplotlib 
        self.comparison = comparison 
        
        if normalization:
            self.df = self.df.apply(lambda x: x/x.max())

        ### CHECK MULTI-INDEX COLUMNS ###
        
        # 1 LEVEL, AXIS 1.
        if not isinstance(self.df.columns, pd.MultiIndex):
            # Create interactive figure.
            w = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
            p = interact(self.plot_df_level_1, graph = w, df = fixed(self.df), title_name = fixed(self.title), df2 = fixed(self.df2))
        else:
            # 2 LEVELS, AXIS 1.
            if len(self.df.columns.levels) == 2:
                # Check if on level 1 there is only an unique feature (in this case the level 1 is deleted and the dataframe returns to have a single level on axis 1).
                if len(self.df.columns.get_level_values(1).unique()) == 1:
                    df = self.df.droplevel(level = 1, axis = 1)
                    if self.comparison:
                        df2 = self.df2.droplevel(level = 1, axis = 1)
                    else:
                        df2 = None
                    # Create figure.
                    w = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                    p = interact(self.plot_df_level_1, graph = w, df = fixed(df), title_name = fixed(self.title), df2 = fixed(df2))
                # Check if the dataframe has an unique feature on level 0 and multiple features on level 1.
                elif len(self.df.columns.get_level_values(0).unique()) == 1 and len(self.df.columns.get_level_values(1).unique()) != 1:
                    w = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                    p = interact(self.plot_df_level_2, name = fixed(self.df.columns.get_level_values(0).unique().values[0]), graph = w, df = fixed(self.df), df2 = fixed(self.df2))
                # Check if dataframe has multiple features both on level 0 and level 1.    
                elif len(self.df.columns.get_level_values(0).unique()) != 1 and len(self.df.columns.get_level_values(1).unique()) != 1: 
                    # Create figure.
                    w1 = widgets.ToggleButtons(options = self.df.columns.get_level_values(0).unique(), 
                                               description = self.df.columns.get_level_values(0).name, 
                                               disabled = False)
                    w2 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                    p = interact(self.plot_df_level_2, name = w1, graph = w2, df = fixed(self.df), df2 = fixed(self.df2))       
            # 3 LEVELS, AXIS 1.
            elif len(self.df.columns.levels) == 3:
                # Check if on level 2 there is only an unique feature (in this case the level 2 is deleted and the dataframe returns to have two levels).
                if len(self.df.columns.get_level_values(2).unique()) == 1:
                    df = self.df.droplevel(level = 2, axis = 1)
                    if self.comparison:
                        df2 = self.df2.droplevel(level = 2, axis = 1)
                    else:
                        df2 = None
                    # Create figure.
                    w1 = widgets.ToggleButtons(options = df.columns.get_level_values(0).unique(), 
                                               description = df.columns.get_level_values(0).name, 
                                               disabled = False)
                    w2 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                    p = interact(self.plot_df_level_2, name = w1, graph = w2, df = fixed(df), df2 = fixed(df2))
                else:
                    # Create figure.
                    w1 = widgets.Dropdown(options = self.df.columns.get_level_values(0).unique(), description = self.df.columns.get_level_values(0).name, 
                                          disabled = False, value = None)
                    w2 = widgets.Dropdown(description = self.df.columns.get_level_values(1).name, disabled = False)

                    # Define a function that updates the content of w2 based on what we select for w1.
                    def update(*args):
                        if w1.value:
                            w2.options = self.df[w1.value].columns.get_level_values(0).unique()
                    w1.observe(update)

                    w3 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                    hbox = widgets.HBox([w1, w2])
                    out = widgets.interactive_output(self.plot_df_level_3, {"name1": w1, "name2": w2, "graph": w3, "df": fixed(self.df), "df2": fixed(self.df2)})
                    display(hbox, w3, out)       
            # 4 LEVELS, AXIS 1.
            elif len(self.df.columns.levels) == 4:
                # Create figure.
                w1 = widgets.Dropdown(options = self.df.columns.get_level_values(0).unique(), description = self.df.columns.get_level_values(0).name, 
                                      disabled = False, value = None)
                w2 = widgets.Dropdown(description = self.df.columns.get_level_values(1).name, disabled = False)
                w3 = widgets.Dropdown(description = self.df.columns.get_level_values(2).name, disabled = False)

                # Define a function that updates the content of w2 based on what we select for w1, and w3 based on w1 nad w2.
                def update(*args):
                    if w1.value:
                        w2.options = self.df[w1.value].columns.get_level_values(0).unique()
                        w3.options = self.df[w1.value][w2.value].columns.get_level_values(0).unique()
                w1.observe(update)
                
                def update2(*args):
                    if w2.value:
                        w3.options = self.df[w1.value][w2.value].columns.get_level_values(0).unique()
                
                w2.observe(update2)

                w4 = widgets.RadioButtons(options = ["Time-series", "Missing values"], description = "Select:", disabled = False)
                hbox = widgets.HBox([w1, w2, w3])
                out = widgets.interactive_output(self.plot_df_level_4, {"name1": w1, "name2": w2, "name3": w3, "graph": w4, "df": fixed(self.df), "df2": fixed(self.df2)})
                display(hbox, w4, out)    
   
    def interactive_plot_predictions(self, quantiles = True, title = None, matplotlib = False, yaxis = None, style = "lines", 
                                     first_last_valid_index_group = False, save_path = None):
        """
        ***Main function***

        This main function allows to interactively plot the predictions statistic of time-series stored into a multi-index 
        columns dataframe with two levels. For each group of the level 0, the time-series into the level 1 must be:
        ['forecast', 'original', 'lower_quantile', 'upper_quantile'].

        Parameters
        ----------
        quantiles: if you want to plot the predicted quantiles.
        title: the title to add to the figures. 
        yaxis: a string value to add on y axis.
        style: the style of the plots. It can be 'lines' or 'lines+markers'.
        matplotlib: if you want to use matplotlib (True) or plotly (False) library to visualize the time-series.
        first_last_valid_index_group: if you want to plt the time-series groups using as reference datetime only the values
           between their first and last valid indeces and not using the entire datetime object of the dataframe.
        save_path: the path where to save the figures displayed. Ifo 'None' the figures are not saved.

       """
        # Define the parameters as attributes of the class.
        self.quantiles = quantiles
        self.title = title
        self.yaxis = yaxis
        self.style = style
        self.matplotlib = matplotlib 
        self.first_last_valid_index_group = first_last_valid_index_group
        self.save_path = save_path

        def plot_statistic_prediction(name, df):
            # Select the subdataframe thanks to an interactive button.
            group = df[name]

            if self.first_last_valid_index_group:
                # Adjust time-series group.
                first_idx = group.first_valid_index()
                last_idx = group.last_valid_index()
                group = group.loc[first_idx:last_idx]

            if self.matplotlib:
                fig, ax = plt.subplots(figsize = (10, 4))

                # Define the style for matplotlib library.
                if self.style == "lines+markers":
                    style = ".-"
                else:
                    style = "-" 

                # Plot entire original serie.
                group["original"].plot(ax = fig.gca(), color = "#0087ff", label = "Actual", style = style)
                # Plot predicted serie.
                group["forecast"].plot(ax = fig.gca(), color = "red", label = "Forecast", style = style)
                # Plot quantiles
                if self.quantiles:
                    ax.fill_between(x = group["forecast"].index, y1 = group["lower_quantile"], y2 = group["upper_quantile"], 
                                    color = "red", alpha = 0.5)
                # Set legend.
                ax.legend(loc = "best", prop = {"size": 15})
                # Set axis names.
                ax.set_ylabel(self.yaxis, fontsize = 15)
                ax.set_xlabel("Datetime", fontsize = 15)
                # Set title.
                ax.set_title(self.title, fontsize = 15)
                ax.tick_params(labelsize = 12)
                #ax.autoscale()
                
                if self.save_path is not None:
                    fig.savefig(save_path, bbox_inches = "tight", dpi = 300)
                
                plt.show()
            else:
                # Create figure.
                fig = go.Figure()
                # Plot quantiles.
                if self.quantiles:
                    fig.add_trace(go.Scatter(x = group["lower_quantile"].index, y = group["lower_quantile"], 
                                             name = "lower_quantile", fill = None, mode = "lines", 
                                             line = dict(width = .5, color = "#B6B6B6")))
                    fig.add_trace(go.Scatter(x = group["upper_quantile"].index, y = group["upper_quantile"], 
                                             name = "upper_quantile", fill = "tonexty", mode = "lines", 
                                             line = dict(width = .5, color = "#B6B6B6")))
                # Plot original serie.
                fig.add_trace(go.Scatter(x = group["original"].index, y = group["original"], mode = self.style, 
                                         name = self.title, legendgroup = self.title, 
                                         line = dict(width = 1.5, color = "#1281FF")))
                # Plot predicted serie.
                fig.add_trace(go.Scatter(x = group["forecast"].index, y = group["forecast"], 
                                         name = "prediction", mode = self.style, line = dict(width = 1.5, color = "#FF8F17")))

                # Edit the layout of the y-axis.
                fig.update_layout(yaxis_title = dict(text = self.yaxis, font = dict(size = 10)))
                # Edit the layout of the title.
                fig.update_layout(title = dict(text = self.title, y = 0.9, x = 0.5))
                # Add range slider on x axis.
                fig.update_layout(xaxis = dict(title = "Datetime", rangeselector = dict(), rangeslider = dict(visible = True), 
                                               type = "date"))
                fig.show()

        # 2 LEVELS, AXIS 1.
        # Create figure.
        w = widgets.ToggleButtons(options = self.df.columns.get_level_values(0).unique(), 
                                  description = self.df.columns.get_level_values(0).name, 
                                  disabled = False)
        p = interact(plot_statistic_prediction, name = w, df = fixed(self.df))  
               