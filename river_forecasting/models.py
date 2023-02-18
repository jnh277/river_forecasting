from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from feature_engine.selection import DropFeatures
from enum import Enum
import xgboost as xg
from sklearn.ensemble import GradientBoostingRegressor
from typing import Optional

import pandas as pd
import numpy as np

# for xgboost quantile regressoin
from xgboost.sklearn import XGBRegressor
from functools import partial

"""
    Module for machline learning pipes/models to perform training and prediction after
     time series feature processing
     
     scikit-learn pipes:
     - KNN
     - RandomForest
     

Important note,
rainfall recorded at time t will impact river level at time t


"""

class LossType(Enum):
    SQUARED_ERROR = 0,
    QUANTILE = 1,

class RegressionModelType(Enum):
    KNN = 0,
    RF = 1,
    XGBOOST = 2,
    LINEAR = 3,
    RIDGE = 4,
    GRADBOOST = 5,
    QUANTILE_GRADBOOST = 6,
    # QUANTILE_XGBOOST = 7,


QUANTILE_MODELS = [RegressionModelType.QUANTILE_GRADBOOST]

def init_scikit_pipe(regression_model: RegressionModelType, quantile: Optional[float] = None, **kwargs):
    if regression_model == RegressionModelType.KNN:
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("KNN", KNeighborsRegressor(n_neighbors=3))
        ])
    elif regression_model == RegressionModelType.RF:
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("random forest", RandomForestRegressor(min_samples_leaf=2))
        ])
    elif regression_model == RegressionModelType.XGBOOST:
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("xgboost", xg.XGBRegressor(objective='reg:squarederror', n_estimators=100))
        ])
    elif regression_model == RegressionModelType.LINEAR:
        return Pipeline([
            ("Drop level", DropFeatures(features_to_drop=["level"])),
            ("min max scaling", MinMaxScaler()),
            ("Polynomial", PolynomialFeatures(degree=(1, 3), include_bias=False)),
            ("LinReg", LinearRegression())
        ])
    elif regression_model == RegressionModelType.RIDGE:
        return Pipeline([
            # ("Drop level", DropFeatures(
            #     features_to_drop=["level", "level_window_3h_mean", "level_window_10h_mean", "level_window_24h_mean",
            #                       "level_window_48h_mean"])),
            ("Drop level", DropFeatures(
                features_to_drop=["level", "level_window_3h_mean", "level_window_10h_mean", "level_window_24h_mean"])), # not droping 48 hour mean
            ("min max scaling", MinMaxScaler()),  # min max scalar converts to numpy array...
            ("Polynomial", PolynomialFeatures(degree=(1, 3), include_bias=False)),
            ("Ridge", Ridge(alpha=1.0))
        ])
    elif regression_model == RegressionModelType.GRADBOOST:
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("gradboost", GradientBoostingRegressor(n_estimators=100))
        ])
    if regression_model == RegressionModelType.QUANTILE_GRADBOOST:
        assert (quantile is not None) and (0.0 < quantile < 1.0), "quantile must be a float > 0 and < 1"
        return Pipeline([
            ("min max scaling", MinMaxScaler()),
            ("quantile gradboost", GradientBoostingRegressor(loss="quantile", n_estimators=100, alpha=quantile))
        ])
    # elif regression_model == RegressionModelType.QUANTILE_XGBOOST:
    #     assert (quantile is not None) and (0.0 < quantile < 1.0), "quantile must be a float > 0 and < 1"
    #     return Pipeline([
    #         ("min max scaling", MinMaxScaler()),
    #         ("quantile xgboost", xg.XGBRegressor(objective='reg:quantile-loss', n_estimators=100))])
    else:
        raise Exception(ValueError)


""" 
    Quantile loss xgboost
    see:
    - https://colab.research.google.com/github/benoitdescamps/benoit-descamps-blogs/blob/master/notebooks/quantile_xgb/xgboost_quantile_regression.ipynb#scrollTo=y7LIgbTUz2RR
    - https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b

"""


class XGBQuantile(XGBRegressor):
    def __init__(self, quant_alpha=0.95, quant_delta=1.0, quant_thres=1.0, quant_var=1.0, base_score=0.5,
                 booster='gbtree', colsample_bylevel=1,
                 colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1,
                 missing=1, n_estimators=100,
                 n_jobs=1, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1,
                 scale_pos_weight=1, subsample=1):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var

        super().__init__(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
                         colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate,
                         max_delta_step=max_delta_step,
                         max_depth=max_depth, min_child_weight=min_child_weight, missing=missing,
                         n_estimators=n_estimators,
                         n_jobs=n_jobs, objective=objective, random_state=random_state,
                         reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight,
                         subsample=subsample)

        self.test = None

    def fit(self, X, y):
        super().set_params(objective=partial(XGBQuantile.quantile_loss, alpha=self.quant_alpha, delta=self.quant_delta,
                                             threshold=self.quant_thres, var=self.quant_var))
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        y_pred = super().predict(X)
        score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
        score = 1. / score
        return score

    @staticmethod
    def quantile_loss(y_true, y_pred, alpha, delta, threshold, var):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta

        grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (
                2 * np.random.randint(2, size=len(y_true)) - 1.0) * var
        hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
        return grad, hess

    @staticmethod
    def original_quantile_loss(y_true, y_pred, alpha, delta):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - (
                (x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta
        return grad, hess

    @staticmethod
    def quantile_score(y_true, y_pred, alpha):
        score = XGBQuantile.quantile_cost(x=y_true - y_pred, alpha=alpha)
        score = np.sum(score)
        return score

    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha - 1.0) * x * (x < 0) + alpha * x * (x >= 0)

    @staticmethod
    def get_split_gain(gradient, hessian, l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i]) / (np.sum(hessian[:i]) + l) + np.sum(gradient[i:]) / (
                    np.sum(hessian[i:]) + l) - np.sum(gradient) / (np.sum(hessian) + l))

        return np.array(split_gain)
