# Forecasting food insecurity

Repository for paper: ***

Authors: Pietro Foini, Michele Tizzoni, Daniela Paolotti, Elisa Omodei

## Overview

In 2019 alone, more than 135 million people across 55 countries faced crisis levels of acute food insecurity or worse, requiring urgent humanitarian and livelihood assistance. Prior to any intervention, the UN World Food Programme (WFP) undertakes an analysis of the food security situation in an area or country to determine the most appropriate operational response. This analysis includes projections, based on expert opinion, on how the situation will evolve, but no mathematical model is currently available to provide robust quantitative estimates. To fill this need, in this work we develop a machine learning model to forecast the evolution of the prevalence of people with insufficient food consumption in different geographical areas. As this socio-economic condition is affected by the evolution of contextual variables, e.g. market prices, conflicts, precipitation or vegetation, etc., we include the associated time-series in order to investigate their causal links with food insecurity. This analysis is performed using a measure that detects the amount of direct transfer of information between pairs of time-series, the Symbolic Transfer Entropy (STE).
Next, we focus on obtaining 30-days predictions for the food insecurity time-series employing the eXtreme Gradient Boosting (XGBoost) machine learning algorithm. 

<p align="center">
  <img src="./Hunger Map.png" width="700">
</p>

<p align="center">Hunger Map (live: https://hungermap.wfp.org/)</p>

## Folders structure

We have divided the main analyzes of the project into several folders. The *packages* folder contains some custom python packages used by other folders of the project. For good navigation within the project, we recommend that you first examine folders *Data Sources* and then *Dataset time-series*. The other folders (except the folder *packages*) are strictly dependent on the results of these two folders. In particular the recommended order for examining the remaining analyzes is as follows: *Correlation*, *Permutation Entropy*, *Symbolic Transfer Entropy* and *Time-series Forecasting*.

## Install the Environment

We provide a yml file containing the necessary packages for the current project. Once you have [conda](https://docs.anaconda.com/anaconda/install/) installed, you can create an environment as follows:
```
conda env create --file ISI_WFP.yml 
```

## Dependencies

The script has been tested running Python 3.6 (Anaconda/miniconda), with the following packages installed (along with their dependencies):

- `pandas==1.1.2`
- `hyperopt==0.2.3`
- `seaborn==0.11.1`
- `geopandas==0.7.0`
- `scipy==1.3.1`
- `ipywidgets==7.5.1`
- `dataframe_image==0.1.1`
- `xgboost==0.90`
- `numpy==1.15.4`
- `matplotlib==3.1.2`
- `plotly==4.2.1`
- `Click==7.0`
- `xlrd==1.2.0`
- `descartes==1.1.0`
- `scikit-learn==0.21.3`
- `mapclassify==2.1.1`
- `patsy==0.5.1`
- `statsmodels==0.11.1`
- `pyinform==0.2.0`
- `imageio==2.4.1`
- `pyarrow==1.0.0`
- `openpyxl==2.6.2`

## License

MIT

## Notes

Github performs a static render of the notebooks and it doesn't include the embedded HTML/JavaScript that makes up a plotly/widgets graph. One option to better expore the proect from remote is to paste the link to the GitHub notebook into http://nbviewer.jupyter.org/.

## Contact Us

Please open an issue or contact pietro.foini1@gmail.com with any questions.
