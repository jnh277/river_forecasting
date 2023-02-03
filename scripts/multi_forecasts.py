import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import seaborn as sns
import numpy as np
from tqdm import tqdm

from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
from river_forecasting.data import load_section, split_contiguous, RainImpulseResponse


forecast_horizon = 24
model_dicts = {'forecast_horizon':forecast_horizon}


SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"
shoalhaven_df = load_section(SECTION_NAME)
dfs = split_contiguous(shoalhaven_df)

# add level difference information
for df in dfs:
    df["level_diff"] = df["level"].diff()

# add rain impulse
rainImpulseResponse = RainImpulseResponse()
rainImpulseResponse.fit(dfs)
transformed_data = rainImpulseResponse.transform(dfs)

# merge data
data = pd.concat(transformed_data, axis=0)
data.dropna(inplace=True)

# apply time series stuff
for fh in tqdm(range(1,forecast_horizon+1), "training models"):

    all_cols = list(data.columns)
    lag_transformer = LagFeatures(variables=['level']+[n for n in all_cols if 'rain_impulse' in n],
                                  freq=[f'{-fh}h'])

    win_f = WindowFeatures(
        window=["3h", "10h", "24h", "48h"], functions=["mean"], variables=["level", "rain", "level_diff"]
    )
    data_ts_trans = lag_transformer.fit_transform(data).dropna()
    data_ts_trans = win_f.fit_transform(data_ts_trans)
    data_ts_trans.dropna(inplace=True)

    y = data_ts_trans[f"level_lag_-{fh}h"]
    X = data_ts_trans.drop(columns=[f"level_lag_-{fh}h","frame"])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    pipe = Pipeline([
        ("min max scaling", MinMaxScaler()),
        # ("linear ridge regression", Ridge(alpha=0.275))
        # ("random forest", RandomForestRegressor(min_samples_leaf=2))
        ("KNN", KNeighborsRegressor(n_neighbors=3))
    ])

    pipe.fit(X_train, y_train)

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    test_score = pipe.score(X_test, y_test)
    train_score = pipe.score(X_train, y_train)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)

    model_dict = {'X_train':X_train,
                  'y_train':y_train,
                  'X_test':X_test,
                  'y_test':y_test,
                  'train_score':train_score,
                  'test_score':test_score,
                  'test_mae':test_mae,
                  'train_mae':train_mae}
    model_dicts[f'{fh}h'] = model_dict

test_score = []
test_mae = []
for i in range(1, forecast_horizon):
    test_score.append(model_dicts[f'{i}h']['test_score'])
    test_mae.append(model_dicts[f'{i}h']["test_mae"])

plt.subplot(2,1,1)
plt.plot(np.arange(1,forecast_horizon), test_mae)
plt.ylabel("MAE")
plt.xlabel("forecast hour")

plt.subplot(2,1,2)
plt.plot(np.arange(1,forecast_horizon), test_score)
plt.ylabel("R Score")
plt.xlabel("forecast hour")

plt.tight_layout()
plt.show()


