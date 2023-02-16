"""
    Module to make predicitons using the forecast data

"""
import pandas as pd
import os
import numpy as np

from river_forecasting.model_manager import load_rain_fir, load_ts_feature, load_trained_pipe, RegressionModelType


# todo: add min historical length
class Predictor():
    def __init__(self, section_name: str,
                 min_history: int = 100):
        self.rainFIR = load_rain_fir(section_name=section_name)
        self.model_info_all = pd.read_csv(os.path.join("../models", section_name, "model_info.csv"),
                                          index_col="Unnamed: 0")
        self.section_name = section_name
        self.min_history = min_history

        # select the models to use
        self.model_info_all.sort_values(by=["test score"], inplace=True, ascending=False)
        self.model_info = self.model_info_all.groupby(by="forecast_step").head(1)

        self.max_forecast_horizon = self.model_info["forecast_step"].max()

        # load models
        self.ts_features = {}
        self.ml_pipes = {}
        for forecast_step in range(1, self.max_forecast_horizon + 1):
            self.ts_features[forecast_step] = load_ts_feature(section_name=section_name, forecast_step=forecast_step)

            step_model_info = self.model_info[self.model_info["forecast_step"] == forecast_step]
            regression_model_type = RegressionModelType[step_model_info["regression_model_type"].values[0]]
            self.ml_pipes[forecast_step] = load_trained_pipe(section_name=section_name,
                                                             forecast_step=forecast_step,
                                                             regression_model_type=regression_model_type
                                                             )

    def predict(self, *,
                level_history: pd.Series,
                rain_history: pd.Series,
                level_diff_history: pd.Series,
                future_rainfall: pd.Series) -> (list, list, list):

        assert len(level_history) >= self.min_history, f"must provide at least {self.min_history}h of past data"

        len_rainfall = len(future_rainfall)
        forecast_horizon = min(len_rainfall, self.max_forecast_horizon)

        # index = [list(past_df), list(future_rainfall.index]
        d = {"level": level_history,
             "rain": pd.concat([rain_history, future_rainfall]),
             "level_diff": level_diff_history}
        data = pd.DataFrame(d)

        data = self.rainFIR.apply_filter([data])[0]
        preds = []
        upper = []
        lower = []

        for forecast_step in range(1, forecast_horizon + 1):
            # only predict for the final point
            X = self.ts_features[forecast_step].transform(data)
            X_curr = X[self.ml_pipes[forecast_step].feature_names_in_].tail(1)
            pred = self.ml_pipes[forecast_step].predict(X_curr)
            preds.append(pred.item())

        return preds, upper, lower


def validate():
    pass


if __name__ == "__main__":
    from river_forecasting.data import load_training_data
    import matplotlib.pyplot as plt
    import time

    # SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"
    # data = load_training_data(SECTION_NAME)[1]
    SECTION_NAME = "franklin_at_fincham"
    data = load_training_data(section_name=SECTION_NAME, source="waterdataonline")[-1]

    data.to_csv(os.path.join("../models", SECTION_NAME, "val_data.csv"))
    test = pd.read_csv(os.path.join("../models", SECTION_NAME, "val_data.csv"))

    predictor = Predictor(section_name=SECTION_NAME)

    forecast_horizon = 96

    # current = len(data)-400
    current = 300
    df_past = data[:current].copy()
    df_future = data[current:current + forecast_horizon].copy()
    future_rainfall = df_future["rain"].copy()

    rain_history = df_past["rain"].to_list()
    level_history = df_past["level"].to_list()
    level_diff_history = df_past["level_diff"].to_list()
    rain_future = df_future["rain"].to_list()

    level_future = df_future["level"].copy()

    index = data[:current + forecast_horizon].index


    t1 = time.perf_counter()
    preds, upper, lower = predictor.predict(level_history=pd.Series(level_history, index=df_past.index),
                                            rain_history=pd.Series(rain_history, index=df_past.index),
                                            level_diff_history=pd.Series(level_diff_history, index=df_past.index),
                                            future_rainfall=pd.Series(rain_future, index=df_future.index))


    t2 = time.perf_counter()
    print("time to predict ",t2-t1)


    plt.plot(level_future, label="True")
    plt.plot(pd.Series(preds, index=level_future.index), label="predicted")
    plt.plot(pd.Series(lower, index=level_future.index), label="lower", linestyle='--', color='g')
    plt.plot(pd.Series(upper, index=level_future.index), label="upper", linestyle='--', color='r')
    # plt.fill_between(x=preds.index, y1=pd.Series(lower, index=level_future.index),y2=pd.Series(upper, index=level_future.index), alphe=0.2)
    plt.title(f"predictions for a {forecast_horizon} hour horizon")
    plt.xlabel("time in future")
    plt.ylabel("river level (m)")
    plt.legend()
    plt.show()


    # x = pd.Series(level_history, index=df_past.index).to_json()
    # pd.read_json(x, typ='series', orient='records')
