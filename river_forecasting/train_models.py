"""
    module to train features and ml pipes for a river section

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from tqdm import tqdm
import numpy as np

from river_forecasting.data import load_training_data
from river_forecasting.processing import RainImpulseResponseFilter
from river_forecasting import model_manager
from river_forecasting.features import TimeSeriesFeatures
from river_forecasting.models import init_scikit_pipe, RegressionModelType, simple_std_eval


regression_model_types = [RegressionModelType.KNN,
                          RegressionModelType.RF,
                          RegressionModelType.XGBOOST,
                          # RegressionModelType.LINEAR,
                          RegressionModelType.RIDGE]

# def simple_std_model(y_pred, y, level):
#     inds = np.abs(y_pred - level) > 0.05
#     steady_std = np.std(y_pred[inds] - y[inds])
#     non_steady_std = np.std(y_pred[~inds] - y[~inds])
#
#     return steady_std, non_steady_std

def train_model(*, section_name: str, forecast_horizon: int=5):
    """
    Train ml models for a river section
    """
    # load the data and do some initial processing and checks
    dfs = load_training_data(section_name=section_name)

    # apply a couple of impulse response filters to the data
    rainFIR = RainImpulseResponseFilter()
    rainFIR.fit_filter(dfs)
    transformed_data = rainFIR.apply_filter(dfs)

    # save the fit filters
    model_manager.save_rain_fir(rainFIR=rainFIR, section_name=section_name)

    # merge data from contiguous segments
    data = pd.concat(transformed_data, axis=0)
    data.dropna(inplace=True)

    # build models for each forecast step up to forecast horizon

    model_info_dicts = []
    for forecast_step in tqdm(range(1, forecast_horizon+1), "Training models to forecast over horizon"):
        # time series features
        time_series_features = TimeSeriesFeatures(forecast_step=forecast_step)
        X, y = time_series_features.fit_transform(data)

        # save the timeseries feature
        model_manager.save_ts_feature(forecast_step=forecast_step,
                                      time_series_features=time_series_features,
                                      section_name=section_name)

        # split data into test and train
        X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42,
                                                            shuffle=False)


        for regression_model_type in regression_model_types:
            X_train = X_train_.copy()
            y_train = y_train_.copy()
            X_test = X_test_.copy()
            y_test = y_test_.copy()


            pipe = init_scikit_pipe(regression_model_type)
            pipe.fit(X_train, y_train)

            # save fit pipe
            model_manager.save_trained_pipe(pipe=pipe, section_name=section_name,
                                            forecast_step=forecast_step,
                                            regression_model_type=regression_model_type)

            # evaluate pipe
            y_train_pred = pipe.predict(X_train)
            y_test_pred = pipe.predict(X_test)

            steady_std, non_steady_std = simple_std_eval(X_test=X_test, y_test=y_test, y_test_pred=y_test_pred)

            model_info_dict = {
                "regression_model_type":regression_model_type.name,
                "forecast_step":forecast_step,
                "section_name":section_name,
                "train score":pipe.score(X_train, y_train),
                "train mse":mean_squared_error(y_train, y_train_pred),
                "train mae":mean_absolute_error(y_train, y_train_pred),
                "test score": pipe.score(X_test, y_test),
                "test mse": mean_squared_error(y_test, y_test_pred),
                "test mae": mean_absolute_error(y_test, y_test_pred),
                "steady std": steady_std,
                "non steady std": non_steady_std
            }
            model_info_dicts.append(model_info_dict)

    model_info = pd.DataFrame(model_info_dicts)
    model_info.to_csv(os.path.join("../models", section_name, "model_info.csv"))


if __name__=="__main__":
    SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"
    forecast_horizon=24
    train_model(section_name=SECTION_NAME, forecast_horizon=24)








