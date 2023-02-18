"""
    Module for saving and loading trained transforms and pipes

"""
from river_forecasting.processing import RainImpulseResponseFilter
from river_forecasting.features import TimeSeriesFeatures
from river_forecasting.models import RegressionModelType, LossType
from sklearn.pipeline import Pipeline
import os.path as Path
import os
import joblib
import pandas as pd
import shutil
from pydantic import BaseModel
from typing import Optional

MODEL_DIRECTORY = "../models/"
PICKLE_PROTOCOL = 5


def save_trained_pipe(*, pipe: Pipeline,
                      section_name: str,
                      forecast_step: int,
                      regression_model_type: RegressionModelType,
                      quantile: Optional[float] = None,
                      extra_str: str = "") -> None:
    section_folder = os.path.join(MODEL_DIRECTORY, section_name)
    make_folder(folder=section_folder)
    step_folder = os.path.join(section_folder, f"step_{forecast_step}hour")
    make_folder(step_folder)
    q_str = f"_q{quantile}".replace(".","_") if (quantile is not None) else ""
    filepath = Path.join(step_folder, f"model_{regression_model_type.name}{q_str}{extra_str}.pkl")
    remove_old_pipe(filepath=filepath)
    joblib.dump(pipe, filename=filepath, protocol=PICKLE_PROTOCOL)

def load_trained_pipe(*, section_name: str,
                      forecast_step: int,
                      regression_model_type: RegressionModelType,
                      quantile: Optional[float]=None,
                      extra_str: str = "")->Pipeline:
    section_folder = os.path.join(MODEL_DIRECTORY, section_name)
    step_folder = os.path.join(section_folder, f"step_{forecast_step}hour")
    make_folder(step_folder)
    q_str = f"_q{quantile}".replace(".","_") if (quantile is not None) else ""
    filepath = Path.join(step_folder, f"model_{regression_model_type.name}{q_str}{extra_str}.pkl")
    return joblib.load(filename=filepath)

def save_rain_fir(*, rainFIR: RainImpulseResponseFilter, section_name: str):
    """
    Removes old saved rainImpulseResponseFilter for this section and saves new
    """
    folder = os.path.join(MODEL_DIRECTORY, section_name)
    make_folder(folder)

    filepath = Path.join(folder, "rain_fir.pkl")
    remove_old_pipe(filepath=filepath)

    joblib.dump(rainFIR, filepath, protocol=PICKLE_PROTOCOL)

def load_rain_fir(*, section_name: str) -> RainImpulseResponseFilter:
    """
        load saved rain_fir for the section
    """
    filepath = os.path.join(MODEL_DIRECTORY, section_name, "rain_fir.pkl")
    return joblib.load(filename=filepath)

def save_ts_feature(*, forecast_step: int, time_series_features: TimeSeriesFeatures, section_name) -> None:
    """
        Removes old saved time series transforms for this forecast step and section name and saves new
    """
    section_folder = os.path.join(MODEL_DIRECTORY, section_name)
    make_folder(folder=section_folder)
    step_folder = os.path.join(section_folder, f"step_{forecast_step}hour")
    make_folder(folder=step_folder)
    filepath = os.path.join(step_folder, "ts_feature.pkl")
    remove_old_pipe(filepath)
    joblib.dump(time_series_features, filepath, protocol=PICKLE_PROTOCOL)

def load_ts_feature(*, section_name:str, forecast_step: int) -> TimeSeriesFeatures:
    filepath = os.path.join(MODEL_DIRECTORY, section_name, f"step_{forecast_step}hour", "ts_feature.pkl")
    return joblib.load(filename=filepath)

def make_folder(folder: Path):
    """
    makes folder if it doesnt exist
    """
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass

def remove_old_pipe(filepath: Path) -> None:
    """
    removes existing saved pipe or saved transform
    """
    try:
        os.remove(filepath)
    except OSError:
        pass


def update_replace_model_info(model_info_dicts:list[dict],
                              regression_model_types:list[RegressionModelType],
                              section_name: str) -> pd.DataFrame:
    try:
        model_info = pd.read_csv(os.path.join(MODEL_DIRECTORY, section_name, "model_info.csv"), index_col="Unnamed: 0")
        inds = model_info["regression_model_type"].isin([t.name for t in regression_model_types])
        model_info = model_info[~inds].copy()
    except FileNotFoundError:
        model_info = pd.DataFrame()
    model_info = pd.concat([model_info, pd.DataFrame(model_info_dicts)], axis=0)
    return model_info

def delete_models(section_name: str):
    path = os.path.join(MODEL_DIRECTORY, section_name)
    shutil.rmtree(path)
    pass

# class ModelInfo(BaseModel):
#     regression_model_type: RegressionModelType
#     forecast_step: int
#     loss: LossType
#     section_name: str
#     train_score: float
#     test_score: float
#     train_mse: Optional[float]
#     test_mse: Optional[float]
#     test_mae: Optional[float]
#     train_mae: Optional[float]




# model_info_dict = {
#         "regression_model_type": regression_model_type.name,
#         "forecast_step": forecast_step,
#         "section_name": section_name,
#         "train score": pipe.score(X_train, y_train),
#         "train mse": mean_squared_error(y_train, y_train_pred),
#         "train mae": mean_absolute_error(y_train, y_train_pred),
#         "test score": pipe.score(X_test, y_test),
#         "test mse": mean_squared_error(y_test, y_test_pred),
#         "test mae": mean_absolute_error(y_test, y_test_pred),
#     }