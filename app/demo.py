import streamlit as st
import pandas as pd
import os.path as Path
import altair as alt
from datetime import datetime, timedelta
import numpy as np

from river_forecasting.data import load_training_data
from river_forecasting.predict import Predictor

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
    predictor = Predictor(section_name=SECTION_NAME)
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
model_info = model_info[model_info["regression_model_type"]=="XGBOOST"].copy()

test_score_plot = alt.Chart(model_info, width=600, height=300).mark_line().encode(
    x='forecast_step',
    y='test score',
)

mae_plot = alt.Chart(model_info,width=600, height=300).mark_line().encode(
    x='forecast_step',
    y='test mae',
)

model_performance_plot = test_score_plot | mae_plot

st.markdown("## XGBoost model validation scores")
st.altair_chart(model_performance_plot, use_container_width=True)



## select forecast point


forecast_point = st.slider(
    "select forecast point",
    min_value=data.index[0].to_pydatetime(),
    max_value=data.index[-1].to_pydatetime(),
    step=timedelta(hours=1),
    value=data.index[200].to_pydatetime(),
    format="MM/DD/YY - hh:mm",)



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

st.altair_chart(level_plot+rules, use_container_width=True)
st.altair_chart(rain_plot+rules, use_container_width=True)

current = np.argmax(data.index==pd.Timestamp(forecast_point))

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

    preds, upper, lower = predictor.predict(level_history=pd.Series(level_history, index=df_past.index),
                                            rain_history=pd.Series(rain_history, index=df_past.index),
                                            level_diff_history=pd.Series(level_diff_history, index=df_past.index),
                                            future_rainfall=pd.Series(rain_future, index=df_future.index))

    pred_df = pd.DataFrame(index=level_future.index)
    pred_df["predicted"] = preds

    predicted_plot = alt.Chart(pred_df.reset_index(), width=600, height=300).mark_line().encode(
        x="datetime",
        y="predicted",
        color=alt.value("#FFAA00")
    )

    actual_plot = alt.Chart(level_future.reset_index(), width=600, height=300).mark_line().encode(
        x="datetime",
        y="level",
    )

    comp_plot = predicted_plot + actual_plot

    st.altair_chart(comp_plot, use_container_width=True)
else:
    st.markdown("forecast requires at least 200 historical data points, move slider further along")

