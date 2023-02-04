import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns

from river_forecasting.data import load_training_data
from river_forecasting.processing import RainImpulseResponseFilter
from river_forecasting.features import TimeSeriesFeatures
from river_forecasting import model_manager
from river_forecasting.models import init_scikit_pipe, RegressionModelType

SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"


## This sectoin is loading training data
dfs = load_training_data(SECTION_NAME)
### end of loading training data


ccf = sm.tsa.stattools.ccf(dfs[1]["level"], dfs[1]["rain"])
ccf_2 = sm.tsa.stattools.ccf(10**(dfs[1]["level"]), dfs[1]["rain"])

plt.plot(ccf[:24*7])
plt.xlabel("lag (hours)")
plt.ylabel("cross correlation factor")
plt.title("cross correlation between rain and river level")
plt.show()

# Filters
rainFIR = RainImpulseResponseFilter()
rainFIR.fit_filter(dfs)

# model_manager.save_rain_fir(rainFIR=rainFIR, section_name=SECTION_NAME)
# rainFIR = model_manager.load_rain_fir(section_name=SECTION_NAME)
transformed_data = rainFIR.apply_filter(dfs)
### END Filters



# merge data
data = pd.concat(transformed_data, axis=0)
data.dropna(inplace=True)

# TIME SERIES FEATURES
forecast_step = 0
time_series_features = TimeSeriesFeatures(forecast_step=forecast_step)
model_manager.save_ts_feature(forecast_step=forecast_step,
                              time_series_features=time_series_features,
                              section_name=SECTION_NAME)
time_series_features = model_manager.load_ts_feature(forecast_step=forecast_step, section_name=SECTION_NAME)
X, y = time_series_features.fit_transform(data)


## END TIME SERIES

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

pipe = init_scikit_pipe(RegressionModelType.KNN)

pipe.fit(X_train, y_train)

# joblib.dump(pipe, os.path.join("../models", SECTION_NAME, "pipe.pkl"))
# pipe = joblib.load(os.path.join("../models", SECTION_NAME, "pipe.pkl"))

y_train_pred = pipe.predict(X_train)
y_test_pred = pipe.predict(X_test)

print("train score", pipe.score(X_train, y_train))
print("train mse", mean_squared_error(y_train, y_train_pred))
print("test score ", pipe.score(X_test, y_test))
print("test mse ", mean_squared_error(y_test, y_test_pred))

plt.plot(y_test.to_numpy())
plt.plot(y_test_pred)
plt.title(f"{forecast_step}h ahead prediction vs actual")
plt.show()

## test saving


