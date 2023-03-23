# On the forecastability of food insecurity 

Repository for paper: [*On the forecastability of food insecurity*](https://www.nature.com/articles/s41598-023-29700-y)

Authors: Pietro Foini, Michele Tizzoni, Giulia Martini, Daniela Paolotti, Elisa Omodei


## Overview

Food insecurity, defined as the lack of physical or economic access to safe, nutritious and sufficient food, remains one of the main challenges included in the 2030 Agenda for Sustainable Development. Near real-time data on the food insecurity situation collected by international organizations such as the World Food Programme can be crucial to monitor and forecast time trends of insufficient food consumption levels in countries at risk. Here, using food consumption observations in combination with secondary data on conflict, extreme weather events and economic shocks (e.g. rise in goods' market price), we build a forecasting model based on gradient boosted regression trees to create predictions on the evolution of insufficient food consumption trends up to 30 days in to the future in 6 countries (Burkina Faso, Cameroon, Mali, Nigeria, Syria and Yemen).


## Directory structure

The analyzes of the project are arranged into several folders. The *packages* folder contains some custom python modules. For good navigation within the project, we recommend to first examine folders *Data Sources* and then *Dataset time-series*. The other folders (except the folder *packages*) are strictly dependent on the results of these two folders. In particular the recommended order for examining the remaining analyzes is as follows: *Correlation*, *Permutation Entropy*, and *Time-series Forecasting*.


## Install the Environment

We provide a .yml file containing the necessary packages for the current project. Once you have [conda](https://docs.anaconda.com/anaconda/install/) installed, you can create an environment as follows:
```
conda env create --file environment.yml 

```

## License

MIT


## How to Cite

```
@article{foini2023forecastability,
  title={On the forecastability of food insecurity},
  author={Foini, Pietro and Tizzoni, Michele and Martini, Giulia and Paolotti, Daniela and Omodei, Elisa},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={2793},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```


## Contact Us

Please open an issue or contact pietro.foini1@gmail.com with any questions.
