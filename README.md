# River Forecasting
Project for forecasting river flows based on rainfall

## Trained models

### Franklin at Fincham
Level gauge:
- franklin_at_fincham
- source: water_data_online

Rainfall:
- franklin_at_fincham
- source: water_data_online
- lat: 42.24 
- lon: 145.77

### Collingwood below alma
Level gauge:
- collingwood_below_alma
- source: water_data_online

Rainfall:
- franklin_at_fincham
- source: water_data_online
- lat: 42.24 
- lon: 145.77


## To do

### Utility

- separate requirements.txt out into train vs use
- remove hardcoding of test/train split value (low)
- better test train split for time series data (low)
- implement parrallel processing in train models (high)


### Data sources

- WaterOnline http://www.bom.gov.au/waterdata/?fbclid=IwAR13h11ahx7RzoIxJyVHd-3Gho5_aNep6TT-vqk6Arp8CFyCzDijZ5hTPIE
  - explore what other sections and gauge combinations could be good
- WikiRiver
  - Get updated _data
- BOM provided barrington river data_ (low)
- work out how to clean the 'bielsdown_at_dorrigo' rain gauge info from water data online (seems in this specific case the quality code needs to be 10, not the case with some other gauges)
- norwegian weather api https://developer.yr.no/featured-products/forecast/ or https://api.met.no/weatherapi/locationforecast/2.0/documentation

### Modelling ideas

- feature selection (HIGH)
- EBM models (MED)
- Monotonically constrained models
- Quantile regression for random forests
- Quantile regression for xgboost when it becomes available (or try this? https://towardsdatascience.com/confidence-intervals-for-xgboost-cac2955a8fde)
- variational inference (less keen on this)
- impulse response neural networks

- multi rain inputs
- upstream river gauge inputs



### Integrations
- WikiRiver


### Software design

- Make package installable
  - MANIFEST.in
- config.yaml (MED)
- TESTS (HIGH)



## Done

### Software design
- make package installable
  - Pyproject.toml
  - Versioning
- API?

### Utility

- forecast input validation!
- made streamlit app
- made package installable and version tracked
- allow better selection of models to be trained
- added compression to saving of models
- have optional retrain or not per model per timestep
- during training save the final decent contiguous section as validation data
- improve steamlit app 


### Datasources

- functionality to use data from water data online
- downloaded and tried the Franklin at fincham data
- historical rain forecast info from the norwegian met https://thredds.met.no/thredds/metno.html - they only store historical forecast for scandinavia


### Modelling

- Quantile regression
- hyperparmeter search (HIGH)
- compare current franklin results with longer dataset (HIGH), diminishing returns

### BUGS
- second level tqdm is not displaying progress bar properly
