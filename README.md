# River Forecasting
Project for forecasting river flows based on rainfall


## To do

### Utility

- improve steamlit app 
- remove hardcoding of test/train split value
- during training save the final decent contiguous section as validation data

### Data sources

- WaterOnline http://www.bom.gov.au/waterdata/?fbclid=IwAR13h11ahx7RzoIxJyVHd-3Gho5_aNep6TT-vqk6Arp8CFyCzDijZ5hTPIE
  - explore what other sections and gauge combinations could be good
- WikiRiver
  - Get updated _data
- BOM provided barrington river data_

### Modelling ideas

- Monotonically constrained models
- Quantile regression
- variational inference
- impulse response neural networks
- hyperparmeter search
- feature selection
- multi rain inputs
- upstream river gauge inputs
- compare current franklin results with longer dataset

### Integrations

- Openweather API
- WikiRiver


### Software design

- Make package installable
  - Pyproject.toml
  - MANIFEST.in
  - Versioning
- API?



## Done

### Utility

- made streamlit app
- made package installable and version tracked

### Datasources

- functionality to use data from water data online
- downloaded and tried the Franklin at fincham data
