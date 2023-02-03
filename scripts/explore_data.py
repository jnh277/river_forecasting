import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import joblib
import os
import numpy as np

from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
from river_forecasting.data import load_section, split_contiguous, RainImpulseResponse

SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"

shoalhaven_df = load_section(SECTION_NAME)


dfs = split_contiguous(shoalhaven_df)


ccf = sm.tsa.stattools.ccf(dfs[1]["level"], dfs[1]["rain"])
ccf_2 = sm.tsa.stattools.ccf(10**(dfs[1]["level"]), dfs[1]["rain"])

plt.plot(ccf[:24*7])
plt.xlabel("lag (hours)")
plt.ylabel("cross correlation factor")
plt.title("cross correlation between rain and river level")
plt.show()

# add level difference information
for df in dfs:
    df["level_diff"] = df["level"].diff()

# add rain impulse
rainImpulseResponse = RainImpulseResponse()
rainImpulseResponse.fit(dfs)

joblib.dump(rainImpulseResponse, os.path.join("../models", SECTION_NAME, "rainImpulseResponse.pkl"))
rainImpulseResponse = joblib.load(os.path.join("../models", SECTION_NAME, "rainImpulseResponse.pkl"))

transformed_data = rainImpulseResponse.transform(dfs)

# merge data
data = pd.concat(transformed_data, axis=0)
data.dropna(inplace=True)

# apply time series stuff
forecast_horizon = 5
all_cols = list(data.columns)
lag_transformer = LagFeatures(variables=['level']+[n for n in all_cols if 'rain_impulse' in n],
                              freq=[f'{-forecast_horizon}h'])

win_f = WindowFeatures(
    window=["3h", "10h", "24h", "48h"], functions=["mean"], variables=["level", "rain", "level_diff"]
)
data_ts_trans = lag_transformer.fit_transform(data).dropna()
data_ts_trans = win_f.fit_transform(data_ts_trans)
data_ts_trans.dropna(inplace=True)

y = data_ts_trans[f"level_lag_-{forecast_horizon}h"]
X = data_ts_trans.drop(columns=[f"level_lag_-{forecast_horizon}h","frame"])

tt = pd.concat([y, X], axis=1)
# calculate the correlation matrix
corr = tt.corr()

# plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

pipe = Pipeline([
    ("min max scaling", MinMaxScaler()),
    # ("linear ridge regression", Ridge(alpha=0.275))
    # ("random forest", RandomForestRegressor(min_samples_leaf=2))
    ("KNN", KNeighborsRegressor(n_neighbors=3))
])

pipe.fit(X_train, y_train)

joblib.dump(pipe, os.path.join("../models", SECTION_NAME, "pipe.pkl"))
pipe = joblib.load(os.path.join("../models", SECTION_NAME, "pipe.pkl"))

y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

print("train score", pipe.score(X_train, y_train))
print("train mse", mean_squared_error(y_train, y_train_pred))
print("test score ", pipe.score(X_test, y_test))
print("test mse ", mean_squared_error(y_test, y_test_pred))

plt.plot(y_test.to_numpy())
plt.plot(y_test_pred)
plt.title(f"{forecast_horizon}h ahead prediction vs actual")
plt.show()

## test saving


