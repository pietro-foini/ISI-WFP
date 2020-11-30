# ISI-WFP

## Overview

In 2018 alone, more than 113 million people across 53 countries faced Crisis levels of acute food insecurity or worse, requiring urgent humanitarian and livelihoods assistance. Prior to any intervention, WFP undertakes an analysis of the food security situation in an area or country to determine the most appropriate operational response. This analysis includes projections, based on expert opinion, on how the situation will evolve, but no mathematical model is currently available to provide robust quantitative estimates. To fill this need, WFP and ISI Foundation are joining forces to develop a machine learning model to forecast the evolution of food security-related indicators. The overall objective of the model is to provide timely predictions to better plan and prepare for emerging food security crises.

The first part of the work will focus on the analysis of two food security-related indicators, that are being collected on a continuous basis by WFP for countries where it operates: the Food Consumption Score (FCS) and the Reduced Coping Strategy Index (rCSI). The goal is to build a model able to predict future values based on previously observed values. Additional indicators might be included in a later stage.
The second part of the work will consist of integrating additional data sources into the model, such as: macroeconomic indicators, market prices, precipitation and vegetation, conflicts, information from news and social media, and displacement data. Enriching the machine learning model with this contextual information will allow to improve the effectiveness and accuracy of the forecast.

## Structure

We have divided the main analyzes of the project into several folders. The *packages* folder contains some custom packages used by other folders of the project. For good navigation within the project, we recommend that you first examine folders *Data Sources* and then *Dataset time-series*. The other folders (except the folder *packages*) are strictly dependent on the results of these two folders. In particular the recommended order for examining the other folders is as follows: *Correlation*, *Symbolic Transfer Entropy* and *Time-series Forecasting*.
