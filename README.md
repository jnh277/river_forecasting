# River Forecasting
Project for forecasting river flows based on rainfall

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

### Modelling

- Quantile regression
- hyperparmeter search (HIGH)
- compare current franklin results with longer dataset (HIGH), diminishing returns

### BUGS
- second level tqdm is not displaying progress bar properly
