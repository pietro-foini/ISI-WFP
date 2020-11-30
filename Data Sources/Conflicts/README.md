# Conflicts

A precarious political situation can lead to general discontent among the people by fostering unrest that can reflect on food security. Clionadh Raleigh et al. [1] describe as ACLED collects realtime and historical data on political violence and protest events in nearly 100 countries. The data are collected each week after that individual researchers have scrutinized information from available reports. This conflict dataset is disaggregated by date (when the event happened); type of violence (what happened); actors (who is involved); and location (where the event happened). An additional useful information, when available, regards the number of fatalities which occur during the events. ACLED does not independently verify details of fatalities, and includes this information as an estimate only, reflecting the content of media reports. ACLED currently codes six types of events, both violent and non-violent. These include:

- *Battles*: violent interactions between two organised armed groups;
- *Explosions/Remote violence*: one-sided violence events in which the tool for engaging in conflict creates asymmetry by taking away the ability of the target to respond;
- *Violence against civilians*: violent events where an organised armed group deliberately inflicts violence upon unarmed non-combatants;
- *Protests*: a public demonstration against a political entity, government institution, policy or group in which the participants are not violent;
- *Riots*: violent events where demonstrators or mobs engage in disruptive acts or disorganised acts of violence against property or people;
- *Strategic development*: accounts for often non-violent activity by conflict and other agents within the context of the war/dispute. Recruitment, looting and arrests are included.

In addition, twenty-five sub-event types are introduced to further disaggregate instances of violence within the wider event type categories.

For more details about the data: https://www.acleddata.com/resources/general-guides/ and https://www.acleddata.com/data/.

[1]. “Raleigh, Clionadh, Andrew Linke, Håvard Hegre and Joakim Karlsen. (2010). “Introducing ACLED-Armed Conflict Location and Event Data.” Journal of Peace Research 47(5) 651-660.”

## Folder structure

- *ACLED_data*: this folder contains the conflicts raw data provided by ACLED;
- *Conflicts.ipynb*: this jupyter notebook analyzes the conflicts raw data and consequently builds related time-series;
- *output_timeseries*: this folder contains the conflicts time-series created by the Conflict.ipynb notebook.
