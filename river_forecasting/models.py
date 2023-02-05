from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from feature_engine.selection import DropFeatures
from enum import Enum
import xgboost as xg
import pandas as pd
import numpy as np

"""
    Module for machline learning pipes/models to perform training and prediction after
     time series feature processing
     
     scikit-learn pipes:
     - KNN
     - RandomForest
     

Important note,
rainfall recorded at time t will impact river level at time t


"""

def simple_std_eval(*,
                    X_test: pd.DataFrame,
                    y_test_pred: pd.DataFrame,
                    y_test: pd.DataFrame) -> (float, float):

    ind = (y_test_pred > X_test["level_window_3h_mean"].values-0.2) & (y_test_pred < X_test["level_window_3h_mean"].values+0.1)
    error = y_test_pred - y_test

    steady_std = np.std(error[ind])
    non_steady_std = np.std(error[~ind])

    return steady_std, non_steady_std



class RegressionModelType(Enum):
    KNN = 0,
    RF = 1,
    XGBOOST = 2,
    LINEAR = 3,
    RIDGE = 4,

def init_scikit_pipe(regression_model: RegressionModelType, **kwargs):
    if regression_model==RegressionModelType.KNN:
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("KNN", KNeighborsRegressor(n_neighbors=3))
        ])
    elif regression_model==RegressionModelType.RF:
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("random forest", RandomForestRegressor(min_samples_leaf=2))
        ])
    elif regression_model==RegressionModelType.XGBOOST:
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("xgboost", xg.XGBRegressor(objective ='reg:squarederror', n_estimators=100))
        ])
    elif regression_model==RegressionModelType.LINEAR:
        return Pipeline([
            ("Drop level", DropFeatures(features_to_drop=["level"])),
            ("min max scaling", MinMaxScaler()),
            ("Polynomial", PolynomialFeatures(degree=(1, 3), include_bias=False)),
            ("LinReg", LinearRegression())
        ])
    elif regression_model==RegressionModelType.RIDGE:
        return Pipeline([
            ("Drop level", DropFeatures(features_to_drop=["level", "level_window_3h_mean", "level_window_10h_mean", "level_window_24h_mean", "level_window_48h_mean"])),
            ("min max scaling", MinMaxScaler()),        # min max scalar converts to numpy array...
            ("Polynomial", PolynomialFeatures(degree=(1,3), include_bias=False)),
            ("Ridge", Ridge(alpha=1.0))
        ])
    else:
        raise Exception(ValueError)

