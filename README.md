# River Forecasting
Project for forecasting river flows based on rainfall

## To do

### Utility

- improve steamlit app 
- remove hardcoding of test/train split value
- during training save the final decent contiguous section as validation data
- better test train split for time series data
- have optional retrain or not per model per timestep

### Data sources

- WaterOnline http://www.bom.gov.au/waterdata/?fbclid=IwAR13h11ahx7RzoIxJyVHd-3Gho5_aNep6TT-vqk6Arp8CFyCzDijZ5hTPIE
  - explore what other sections and gauge combinations could be good
- WikiRiver
  - Get updated _data
- BOM provided barrington river data_

### Modelling ideas

- Monotonically constrained models
- Quantile regression for random forests
- Quantile regression for xgboost when it becomes available (or try this? https://towardsdatascience.com/confidence-intervals-for-xgboost-cac2955a8fde)
- variational inference (less keen on this)
- impulse response neural networks
- hyperparmeter search (HIGH)
- feature selection (HIGH)
- multi rain inputs
- upstream river gauge inputs
- compare current franklin results with longer dataset (HIGH)
- EBM models (MED)

### Integrations

- Openweather API (HIGH)
- WikiRiver


### Software design

- Make package installable
  - MANIFEST.in
- API?
- config.yaml (MED)
- TESTS (HIGH)



## Done

### Software design
- make package installable
  - Pyproject.toml
  - Versioning

### Utility

- made streamlit app
- made package installable and version tracked
- allow better selection of models to be trained
- added compression to saving of models

### Datasources

- functionality to use data from water data online
- downloaded and tried the Franklin at fincham data

### Modelling

- Quantile regression
