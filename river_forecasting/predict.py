"""
    Module to make predicitons using the forecast data

"""
import pandas as pd
import os

from river_forecasting.model_manager import load_rain_fir, load_ts_feature, load_trained_pipe, RegressionModelType


class Predictor():
    def __init__(self, section_name: str):
        self.rainFIR = load_rain_fir(section_name=section_name)
        self.model_info_all = pd.read_csv(os.path.join("../models", section_name, "model_info.csv"),
                                          index_col="Unnamed: 0")
        self.section_name = section_name

        # select the models to use
        self.model_info_all.sort_values(by=["test score"], inplace=True, ascending=False)
        self.model_info = self.model_info_all.groupby(by="forecast_step").head(1)

        self.max_forecast_horizon = self.model_info["forecast_step"].max()

        # load models
        self.ts_features = {}
        self.ml_pipes = {}
        for forecast_step in range(self.max_forecast_horizon+1):
            self.ts_features[forecast_step] = load_ts_feature(section_name=section_name, forecast_step=forecast_step)

            step_model_info = self.model_info[self.model_info["forecast_step"]==forecast_step]
            regression_model_type = RegressionModelType[step_model_info["regression_model_type"].values[0]]
            self.ml_pipes[forecast_step] = load_trained_pipe(section_name=section_name,
                                                   forecast_step=forecast_step,
                                                   regression_model_type=regression_model_type
                                                   )

    def predict(self, *,
                level_history: pd.Series,
                rain_history: pd.Series,
                level_diff_history: pd.Series,
                future_rainfall: pd.Series) -> dict:

        len_rainfall = len(future_rainfall)
        forecast_horizon = min(len_rainfall, self.max_forecast_horizon)

        # index = [list(past_df), list(future_rainfall.index]
        d = {"level": level_history,
             "rain": pd.concat([rain_history, future_rainfall]),
             "level_diff": level_diff_history}
        data = pd.DataFrame(d)

        data = self.rainFIR.apply_filter([data])[0]
        pred = {}

        for forecast_step in range(forecast_horizon):
            # only predict for the final point
            X = self.ts_features[forecast_step].transform(data)[-1]
            pred[forecast_step] = self.ml_pipes[forecast_step].predict(X)

        return pred




def validate():
    pass

if __name__=="__main__":
    from river_forecasting.data import load_training_data

    SECTION_NAME = "shoalhaven-river-oallen-ford-to-tallowa-dam"

    predictor = Predictor(section_name=SECTION_NAME)

    data = load_training_data(SECTION_NAME)[1]

    current = 300
    df_past = data[:current].copy()
    df_future = data[current:current+25].copy()
    future_rainfall = df_future["rain"].copy()

    rain_history = df_past["rain"].to_list()
    level_history = df_past["level"].to_list()
    level_diff_history = df_past["level_diff"].to_list()
    rain_future = df_future["rain"].to_list()

    index = data[:current+25].index

    # d = {"level":pd.Series(level_history, index=df_past.index),
    #      "rain":pd.Series(rain_history + rain_future, index=index),
    #      "level_diff":pd.Series(level_diff_history,index=df_past.index)}
    # df = pd.DataFrame(d)


    preds = predictor.predict(level_history=pd.Series(level_history, index=df_past.index),
                              rain_history=pd.Series(rain_history, index=df_past.index),
                              level_diff_history=pd.Series(level_diff_history,index=df_past.index),
                              future_rainfall=pd.Series(rain_future, index=df_future.index))

