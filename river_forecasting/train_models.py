"""
    module to train features and ml pipes for a river section

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_pinball_loss
import os
from tqdm import tqdm
from typing import Optional
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from river_forecasting.data import load_training_data
from river_forecasting.processing import RainImpulseResponseFilter
from river_forecasting import model_manager
from river_forecasting.features import TimeSeriesFeatures
from river_forecasting.models import init_scikit_pipe, RegressionModelType, QUANTILE_MODELS, LossType
from river_forecasting.hp_opt import run_hpopt


DEFAULT_REGRESSION_TYPES = [RegressionModelType.KNN,
                            # RegressionModelType.RF,       # model is too large
                            RegressionModelType.XGBOOST,
                            # RegressionModelType.LINEAR,   # similar to ridge but always worse
                            RegressionModelType.RIDGE,
                            RegressionModelType.GRADBOOST]


# QUANTILES = (0.025, 0.175, 0.825, 0.975)
# QUANTILES = (0.175, 0.825)
# QUANTILES = (0.025, 0.975)
QUANTILES = (0.1, 0.9)  # 80%

def train_model(*, section_name: str,
                forecast_horizon: int = 5,
                source="wikiriver",
                regression_model_types: list[RegressionModelType] = tuple(DEFAULT_REGRESSION_TYPES),
                retrain: bool = False,
                tune: bool = False,
                ):
    """
    Train ml models for a river section
    """

    model_info_manager = model_manager.ModelInfoManager(section_name=section_name)

    if not isinstance(regression_model_types, list):
        regression_model_types = [regression_model_types]

    # load the data and do some initial processing and checks
    dfs = load_training_data(section_name=section_name, source=source)

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
    for forecast_step in tqdm(range(1, forecast_horizon + 1), desc="Iterating forecast horizon", position=0):
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

        model_types, loss_types, quantiles = parse_model_types(regression_model_types)


        for model_type, loss_type, quantile in (pbar := tqdm(zip(model_types, loss_types, quantiles), leave=False, position=1, total=len(model_types), desc="Iterating models")):
            if loss_type==LossType.QUANTILE:
                pbar.set_postfix({'Model type': model_type.name, "quantile": quantile})
                # pbar.set_description(desc=f"Training {model_type} for forecast step {forecast_step} and quantile {quantile}")
            else:
                pbar.set_postfix({'Model type': model_type.name})
                # pbar.set_description(desc=f"Training {model_type} for forecast step {forecast_step}")

            if (not retrain) and (model_info_manager.already_trained(model_type=model_type,
                                                                     forecast_step=forecast_step,
                                                                     quantile=quantile)):
                continue

            X_train = X_train_.copy()
            y_train = y_train_.copy()
            X_test = X_test_.copy()
            y_test = y_test_.copy()

            if tune:
                pipe = run_hpopt(model_type, X_train_, y_train_, X_test_, y_test_, quantile)
            else:
                pipe = init_scikit_pipe(model_type, quantile=quantile)
                pipe.fit(X_train, y_train)

            model_manager.save_trained_pipe(pipe=pipe, section_name=section_name, forecast_step=forecast_step,
                                            regression_model_type=model_type, quantile=quantile)

            # evaluate pipe
            y_train_pred = pipe.predict(X_train)
            y_test_pred = pipe.predict(X_test)

            train_mse = None
            test_mse = None
            train_mae = None
            test_mae = None
            test_pinball = None
            train_pinball = None
            if loss_type==LossType.SQUARED_ERROR:
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
            elif loss_type==LossType.QUANTILE:
                train_pinball = mean_pinball_loss(y_train, y_train_pred, alpha=quantile)
                test_pinball = mean_pinball_loss(y_test, y_test_pred, alpha=quantile)


            model_info_dict = {
                "regression_model_type": model_type.name,
                "forecast_step": forecast_step,
                "section_name": section_name,
                "loss type": loss_type.name,
                "quantile": quantile,
                "train score": pipe.score(X_train, y_train),
                "test score": pipe.score(X_test, y_test),
                "train mse": train_mse,
                "train mae": train_mae,
                "test mse": test_mse,
                "test mae": test_mae,
                "train pinball": train_pinball,
                "test pinball": test_pinball,
                "tuned": tune
            }

            model_info_dicts.append(model_info_dict)

    # model_info = model_manager.update_replace_model_info(model_info_dicts, regression_model_types, section_name)
    model_info_manager.update(model_info_dicts=model_info_dicts)
    model_info_manager.save()

    dfs[-1].to_csv(os.path.join("../models", SECTION_NAME, "val_data.csv"))
    # model_info.to_csv(os.path.join("../models", section_name, "model_info.csv"))


def parse_model_types(regression_model_types: RegressionModelType) -> (list, list, list):
    loss_types = []
    model_types = []
    quantiles = []
    for model_type in regression_model_types:
        if model_type in QUANTILE_MODELS:
            for quantile in QUANTILES:
                quantiles.append(quantile)
                model_types.append(model_type)
                loss_types.append(LossType.QUANTILE)
        else:
            loss_types.append(LossType.SQUARED_ERROR)
            model_types.append(model_type)
            quantiles.append(None)
    return model_types, loss_types, quantiles



if __name__ == "__main__":
    # SECTION_NAME = "franklin_at_fincham"
    # SECTION_NAME = "franklin_at_fincham_long"
    # forecast_horizon=96
    # train_model(section_name=SECTION_NAME, forecast_horizon=forecast_horizon, source="waterdataonline",
    #             regression_model_types=[RegressionModelType.GRADBOOST,
    #                                     RegressionModelType.XGBOOST,
    #                                     RegressionModelType.RIDGE,
    #                                     RegressionModelType.QUANTILE_GRADBOOST])

    SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"
    forecast_horizon = 24
    train_model(section_name=SECTION_NAME, forecast_horizon=forecast_horizon,
                regression_model_types=[
                                        RegressionModelType.XGBOOST,
                                        # RegressionModelType.KNN,
                                        # RegressionModelType.RF,
                                        # RegressionModelType.RIDGE,
                                        # RegressionModelType.GRADBOOST,
                                        RegressionModelType.QUANTILE_GRADBOOST
                                        ],
                tune=True,
                retrain=True)

    # model_manager.delete_models(SECTION_NAME)