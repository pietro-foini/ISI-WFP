# TsIP

**TsIP** (**T**ime-**s**eries **I**nteractive **P**lot) is a python library developed to interactively visualize multiple time-series quickly and easily. 
The implementation of this tool addresses the need to visualize time-series stored into a pandas dataframe with hierarchical multi-index on axis 1, taking advantage of dynamic user interaction.
The time-series are plotted using either the *matplotlib* library or the *plotly* library (user will). For a more involving navigation within the time-series visualization, it is advice to use the plotly library rather than the matplotlib library.  

## Examples

1. Supposing to have a simple dataframe `df` with a single level on axis 1:

<p align="center">
<img src="./images_readme/level1.png" width="800">
</p>

Using the TsIP library, we can easily visualize all the time-series with a simple command:

<p align="center">
<img src="./images_readme/level1.gif" width="800">
</p>

N.B. If `matplotlib = False`, the time-series are visualized using the plotly library which as you can see is much more engaging. Otherwise, a similar interactive plot is shown using the matplotlib library.

2. Supposing to have a dataframe `df` with two levels on axis 1:

<p align="center">
<img src="./images_readme/level2.png" width="800">
</p>

We now visualize the time-series using the matplotlib library as example:

<p align="center">
<img src="./images_readme/level2.gif" width="800">
</p>

N.B. The TsIP module works fine if all the levels of the dataframe have set a name:

<p align="center">
<img src="./images_readme/level2_level_names.png" width="800">
</p>

3. Supposing to have a dataframe `df` with three levels on axis 1:

<p align="center">
<img src="./images_readme/level3.png" width="800">
</p>

Returning to visualize the time-series using the plotly library:

<p align="center">
<img src="./images_readme/level3.gif" width="800">
</p>

The TsIP module has the potential to manage dataframes that can have up to 4 levels on axis 1. 

Furthermore, it is also possible to compare the time-series that belong to two equal hierarchical dataframes.

<p align="center">
<img src="./images_readme/comparison.png" width="800">
</p>

The package also contains a further function to plot the predictions results obtained by a time-series forecasting algorithm.

