"""
    Module for saving and loading trained transforms and pipes

"""
from river_forecasting.processing import RainImpulseResponseFilter
from river_forecasting.features import TimeSeriesFeatures
from river_forecasting.models import RegressionModelType
from sklearn.pipeline import Pipeline
import os.path as Path
import os
import joblib


def save_trained_pipe(*, pipe: Pipeline,
                      section_name: str,
                      forecast_step: int,
                      regression_model_type: RegressionModelType,
                      extra_str: str = "") -> None:
    section_folder = os.path.join("../models/", section_name)
    make_folder(folder=section_folder)
    step_folder = os.path.join(section_folder, f"step_{forecast_step}hour")
    make_folder(step_folder)
    filepath = Path.join(step_folder, f"model_{regression_model_type.name}{extra_str}.pkl")
    remove_old_pipe(filepath=filepath)
    joblib.dump(pipe, filename=filepath)

def load_trained_pipe(*, section_name: str,
                      forecast_step: int,
                      regression_model_type: RegressionModelType,
                      extra_str: str = "")->Pipeline:
    section_folder = os.path.join("../models/", section_name)
    step_folder = os.path.join(section_folder, f"step_{forecast_step}hour")
    make_folder(step_folder)
    filepath = Path.join(step_folder, f"model_{regression_model_type.name}{extra_str}.pkl")
    return joblib.load(filename=filepath)

def save_rain_fir(*, rainFIR: RainImpulseResponseFilter, section_name: str):
    """
    Removes old saved rainImpulseResponseFilter for this section and saves new
    """
    folder = os.path.join("../models", section_name)
    make_folder(folder)

    filepath = Path.join(folder, "rain_fir.pkl")
    remove_old_pipe(filepath=filepath)

    joblib.dump(rainFIR, filepath)

def load_rain_fir(*, section_name: str) -> RainImpulseResponseFilter:
    """
        load saved rain_fir for the section
    """
    filepath = os.path.join("../models", section_name, "rain_fir.pkl")
    return joblib.load(filename=filepath)

def save_ts_feature(*, forecast_step: int, time_series_features: TimeSeriesFeatures, section_name) -> None:
    """
        Removes old saved time series transforms for this forecast step and section name and saves new
    """
    section_folder = os.path.join("../models/", section_name)
    make_folder(folder=section_folder)
    step_folder = os.path.join(section_folder, f"step_{forecast_step}hour")
    make_folder(folder=step_folder)
    filepath = os.path.join(step_folder, "ts_feature.pkl")
    remove_old_pipe(filepath)
    joblib.dump(time_series_features, filepath)

def load_ts_feature(*, section_name:str, forecast_step: int) -> TimeSeriesFeatures:
    filepath = os.path.join("../models/", section_name, f"step_{forecast_step}hour", "ts_feature.pkl")
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
