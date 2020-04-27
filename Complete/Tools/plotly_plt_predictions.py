import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_predictions(serie_original, serie_predicted, serie_quantile_25 = None, serie_quantile_75 = None, 
                     title = None, plotly = False, save = False, dir_to_save = None):
    if plotly:
        # Create figure.
        fig = go.Figure()
        # Plot quantiles.
        if type(serie_quantile_25) != type(None):
            fig.add_trace(go.Scatter(x = serie_quantile_25.index, y = serie_quantile_25, name = "quantile 25", fill = None, mode = "lines", line = dict(width = .5, color = "#B6B6B6")))
        if type(serie_quantile_75) != type(None):
            fig.add_trace(go.Scatter(x = serie_quantile_75.index, y = serie_quantile_75, name = "quantile 75", fill = "tonexty", mode = "lines", line = dict(width = .5, color = "#B6B6B6")))
        # Plot original serie.
        fig.add_trace(go.Scatter(x = serie_original.index, y = serie_original, mode = "lines", name = title, legendgroup = title, 
                                 line = dict(width = 1.5, color = "#1281FF")))
        # Plot predicted serie.
        fig.add_trace(go.Scatter(x = serie_predicted.index, y = serie_predicted, name = "prediction", mode = "lines", line = dict(width = 1.5, color = "#FF8F17")))

        # Edit the layout.
        fig.update_layout(title = dict(text = title, y = 0.9, x = 0.5), 
                          yaxis_title = dict(text = "% of people with poor and borderline FCS", font = dict(size = 10)))

        # Add range slider.
        fig.update_layout(xaxis = dict(title = "Datetime", rangeselector = dict(), rangeslider = dict(visible = True), 
                                       type = "date"))
        
        if save:
            fig.write_image(dir_to_save + title + ".png", width = 900, height = 550, scale = 2)
        
        return fig        
    else:
        fig = plt.figure(figsize = (20, 5))
        # Plot entire original serie.
        serie_original.plot(ax = fig.gca(), color = "#1281FF", label = "original")
        # Plot predicted serie.
        serie_predicted.plot(ax = fig.gca(), color = "#FF8F17", label = "predicted")
        # Plot quantiles
        if type(serie_quantile_25) != type(None) and type(serie_quantile_75) != type(None):
            plt.fill_between(x = serie_predicted.index, y1 = serie_quantile_25, y2 = serie_quantile_75, color = "#B6B6B6", alpha = 0.5)
        plt.title(title)
        plt.ylabel("% of people with poor and borderline FCS")
        plt.xlabel("Datetime")
        plt.legend(loc = "center left", bbox_to_anchor = (1.0, 0.5))
        plt.autoscale()
        
        if save:
            plt.savefig(dir_to_save + title + ".png")
            
        plt.close()
            
        return fig