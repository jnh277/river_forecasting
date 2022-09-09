import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from river_forecasting.data import load_section, split_contiguous

SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"

shoalhaven_df = load_section(SECTION_NAME)

times = shoalhaven_df.index.values
td = (times[1:]-times[:-1])/60/60/1e9

dfs = split_contiguous(shoalhaven_df)

class Feature_square():
    """ adds squared value of a feature """

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.variables:
            X[var + "^2"] = X[var] ** 2
        return X

class Feature_Difference():
    """ adds the difference of a feature """

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.variables:
            X[var + "_diff"] = X[var].diff().fillna(method="bfill")

        return X

class ARX_Transform():
    """ adds the difference of a feature """

    def __init__(self, variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for var in self.variables:
            X[var + "_diff"] = X[var].diff().fillna(method="bfill")

        return X


# todo: need to not split randomly
# todo: need to include that we know future inputs or an alternte way of forecasting
# todo: blah blah blah

X_train, X_test, y_train, y_test = train_test_split(dfs[1][:-10],dfs[1]["level"][10:], test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(dfs[1]["rain"], dfs[1]["level"], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("level_difference", Feature_Difference(["level"])),
    ("squared transformer", Feature_square(["rain", "level"])),
    ("Scaler", MinMaxScaler()),
    ("linear regression", LinearRegression())
])

pipeline.fit(X_train, y_train)
y_test_pred = pipeline.predict(X_test)

plt.plot(y_test.values[:100])
plt.plot(y_test_pred[:100])
plt.show()



# plt.subplot(2,1,1)
# plt.plot(shoalhaven_df.index,shoalhaven_df["rain"])
# plt.ylabel("rainfall (mm) per hour")
# plt.xlabel("date and time (hour increments)")
#
# plt.subplot(2,1,2)
# plt.plot(shoalhaven_df.index, shoalhaven_df["level"])
# plt.ylabel("river level (m)")
# plt.xlabel("date and time (hour increments)")
#
# plt.tight_layout()
# plt.show()




