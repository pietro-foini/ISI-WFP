# Rainfall-Vegetation

In this research, we integrate information regarding precipitation and vegetation. In fact, WFP provides a platform allowing users to assess the performance of the current and past rainfall seasons, the timing and intensity of drier or wetter than average conditions and their impact on vegetation status at the sub-national level for most countries\footnote{The primary data sources are CHIRPS gridded rainfall dataset produced by the Climate Hazards Group at the University of California, Santa Barbara and the MODIS NDVI CMG data made available by NOAA-NASA.}. Summarizing, the available information are the rainfall amount, 1-month rainfall anomaly, 3-month rainfall anomaly, NDVI index, and NDVI index anomaly. The corresponding data refers to the dekad of the months: 1-10 days, 11-20 days, and 21 up to the end of the month.

For more detail see the following site: https://dataviz.vam.wfp.org/seasonal_explorer/rainfall_vegetation/visualizations#

## Folder structure

- *wfp_data*: this folder contains the rainfall and vegetation raw data provided by WFP;
- *Rainfall.ipynb*: this jupyter notebook analyzes the rainfall raw data and consequently creates related time-series;
- *Vegetation.ipynb*: this jupyter notebook analyzes the vegetation raw data and consequently creates related time-series;
- *output_timeseries*: this folder contains all the rainfall and vegetation time-series created by the Rainfall.ipynb and the Vegetation.ipynb notebooks.


