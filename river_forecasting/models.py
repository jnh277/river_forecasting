from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from enum import Enum

"""
    Module for machline learning pipes/models to perform training and prediction after
     time series feature processing
     
     scikit-learn pipes:
     - KNN
     - RandomForest
     

Important note,
rainfall recorded at time t will impact river level at time t


"""

class RegressionModelType(Enum):
    KNN = 0,
    RF = 1,

def init_scikit_pipe(regression_model: RegressionModelType, **kwargs):
    if regression_model==RegressionModelType.KNN:
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("KNN", KNeighborsRegressor(n_neighbors=3))
        ])
    elif regression_model==RegressionModelType.RF:
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("random forest", RandomForestRegressor(min_samples_leaf=2, max_samples=3))
        ])

