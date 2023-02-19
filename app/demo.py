import streamlit as st
import pandas as pd
import os.path as Path
import altair as alt
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from river_forecasting.data import load_training_data
from river_forecasting.predict import Predictor
from river_forecasting.models import RegressionModelType, QUANTILE_MODELS

st.set_page_config(layout="wide")

TEST_TRAIN_SPLIT = 0.2
DATA_FOLDER = "../data"
MODEL_FOLDER = "../models"
SECTION_NAME = "franklin_at_fincham"
forecast_horizon = 96

st.title(f'{SECTION_NAME} river forecasting demo')


@st.cache_data
def load_data_wrapper() -> pd.DataFrame:
    data = pd.read_csv(Path.join(MODEL_FOLDER, SECTION_NAME, "val_data.csv"),
                       parse_dates=["datetime"],
                       index_col="datetime")
    return data


@st.cache_data
def load_predictor_wrapper() -> Predictor:
    predictor = Predictor(section_name=SECTION_NAME,
                          mean_predictor_type=RegressionModelType.XGBOOST,
                          quantile_predictor_type=RegressionModelType.QUANTILE_GRADBOOST
                          )
    return predictor


@st.cache_data
def load_model_info() -> pd.DataFrame:
    model_info = pd.read_csv(Path.join(MODEL_FOLDER, SECTION_NAME, "model_info.csv"))
    return model_info


data = load_data_wrapper()
# predictor = load_predictor_wrapper()
model_info = load_model_info()
predictor = load_predictor_wrapper()

# inds = model_info["test score"].idxmax()
# model_info = model_info[model_info["regression_model_type"]=="XGBOOST"].copy()
inds = model_info["regression_model_type"].isin([t.name for t in QUANTILE_MODELS])

test_score_plot = alt.Chart(model_info[~inds], width=600, height=300).mark_line().encode(
    x='forecast_step',
    y=alt.Y('test score', scale=alt.Scale(domain=[max(0, model_info[~inds]["test score"].min()-0.1), 1])),
    color="regression_model_type"
)

mae_plot = alt.Chart(model_info[~inds], width=600, height=300).mark_line().encode(
    x='forecast_step',
    y='test mae',
    color="regression_model_type"
)

model_performance_plot = test_score_plot | mae_plot

st.markdown("## Trained model validation scores")
st.altair_chart(model_performance_plot, use_container_width=True)

## select forecast point


forecast_point = st.slider(
    "select forecast point",
    min_value=data.index[0].to_pydatetime(),
    max_value=data.index[-1].to_pydatetime(),
    step=timedelta(hours=1),
    value=data.index[200].to_pydatetime(),
    format="MM/DD/YY - hh:mm", )

# plot the data
level_plot = alt.Chart(data.reset_index()).mark_line().encode(
    x="datetime",
    y="level"
)

rain_plot = alt.Chart(data.reset_index()).mark_line().encode(
    x="datetime",
    y="rain"
)

rules = alt.Chart(pd.DataFrame({
    'Date': [forecast_point],
    'color': ['red']
})).mark_rule().encode(
    x='Date:T',
    color=alt.Color('color:N', scale=None)
)

# data_plot = alt.vconcat(level_plot+rules, )

st.altair_chart(level_plot + rules, use_container_width=True)
st.altair_chart(rain_plot + rules, use_container_width=True)

current = np.argmax(data.index == pd.Timestamp(forecast_point))

if current >= 200:
    df_past = data[:current].copy()
    df_future = data[current:current + forecast_horizon].copy()
    future_rainfall = df_future["rain"].copy()

    rain_history = df_past["rain"].to_list()
    level_history = df_past["level"].to_list()
    level_diff_history = df_past["level_diff"].to_list()
    rain_future = df_future["rain"].to_list()

    level_future = df_future["level"].copy()

    index = data[:current + forecast_horizon].index

    preds, lower, upper = predictor.predict(level_history=pd.Series(level_history, index=df_past.index),
                                            rain_history=pd.Series(rain_history, index=df_past.index),
                                            level_diff_history=pd.Series(level_diff_history, index=df_past.index),
                                            future_rainfall=pd.Series(rain_future, index=df_future.index))

    pred_df = pd.DataFrame(index=level_future.index)
    pred_df["predicted"] = preds
    pred_df["upper"] = upper
    pred_df["lower"] = lower

    min_y = min(pred_df.min().min()-0.1, level_future.min()-0.1)
    max_y = max(pred_df.min().min()+0.1, level_future.min()+0.1)


    plot_data = pd.concat([level_future, pred_df[["predicted"]]], axis=1)


    # plot_data = level_future.copy()
    # plot_data["predicted"] = pred_df["predicted"]

    plot_data = plot_data.reset_index().melt("datetime")

    # fig = plt.figure(figsize=(6, 3))
    # plt.plot(pred_df["predicted"], color='b', linewidth=2, label="forecast")
    # plt.plot(pred_df["upper"], color='b', linewidth=1.)
    # plt.plot(pred_df["lower"], color='b', linewidth=1.)
    # plt.plot(level_future, color='k', linewidth=2, label="actual")
    # plt.grid(visible=True, which="major")
    #
    # plt.legend()
    #
    # st.pyplot(fig)

    predicted_plot = alt.Chart(plot_data, width=1200, height=600).mark_line().encode(
        x="datetime",
        y=alt.Y("value", scale=alt.Scale(domain=[data["level"].min()-0.1, data["level"].max()+0.1])),
        # y="Value",
        color="variable"
    )
    #
    prediction_interval = alt.Chart(pred_df.reset_index(), width=1200, height=600).mark_area().encode(
        x="datetime",
        y="lower",
        y2="upper",
        color=alt.value('blue'),
        opacity=alt.value(0.2)
    )
    #
    # actual_plot = alt.Chart(level_future.reset_index(), width=600, height=300).mark_line().encode(
    #     x="datetime",
    #     y="level",
    #     color=alt.value('red')
    # )
    #
    comp_plot = predicted_plot + prediction_interval
    #
    st.altair_chart(comp_plot, use_container_width=False)
else:
    st.markdown("forecast requires at least 200 historical data points, move slider further along")
