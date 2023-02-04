"""
    Module for custom features
    - time series features (arx style transforms)

"""
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
import pandas as pd


class TimeSeriesFeatures():
    """
    TimeSeriesFeatures
    :param forecast_step: (int) 0 is predicting current hour based on rainfall that has occured this hour
    """

    def __init__(self, forecast_step: int = 0,
                 window_features: list[str] = ("level", "rain", "level_diff"),
                 windows: list[str] = ("3h", "10h", "24h", "48h")):
        self.forecast_step = forecast_step
        self.window_features = list(window_features)
        self.windows = list(windows)
        self.win_f = WindowFeatures(
            window=self.windows, functions=["mean"], variables=self.window_features
        )

    def fit_transform(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """ fit the features and then transform the training data
            neither of the two composite transforms learn anything other than which columns to apply t
            returns X and y for training
        """
        all_cols = list(data.columns)
        self.lag_features = [n for n in all_cols if 'rain_impulse' in n]
        self.lag_transformer_train = LagFeatures(variables=['level'] + self.lag_features,
                                                 freq=[f'{-self.forecast_step}h'])
        self.lag_transformer_pred = LagFeatures(variables=self.lag_features,
                                                freq=[f'{-self.forecast_step}h'])

        data_ts_trans = self.lag_transformer_train.fit_transform(data).dropna()
        # self.lag_transformer_pred.fit(data)
        data_ts_trans = self.win_f.fit_transform(data_ts_trans)
        data_ts_trans.dropna(inplace=True)

        if self.forecast_step: # it not 0
            y = data_ts_trans[f"level_lag_-{self.forecast_step}h"]
            X = data_ts_trans.drop(columns=[f"level_lag_-{self.forecast_step}h"])
        else:
            y = data_ts_trans[f"level_lag_{self.forecast_step}h"]
            X = data_ts_trans.drop(columns=[f"level_lag_{self.forecast_step}h"])

        return X, y

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data and returns X for prediction
        """
        data_ts_trans = self.lag_transformer_pred.fit_transform(data).dropna()
        X = self.win_f.fit_transform(data_ts_trans).dropna()

        return X
