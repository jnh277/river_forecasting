import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

from river_forecasting import __version__ as model_version
from river_forecasting.predict import Predictor
from river_forecasting.models import RegressionModelType

from api import __version__, schemas
from api.config import settings
from api.schemas.predict import Sections
import time

api_router = APIRouter()

predictors = {
    Sections.franklin_at_fincham: Predictor(section_name=Sections.franklin_at_fincham.name,
                                            mean_predictor_type=RegressionModelType.XGBOOST,
                                            quantile_predictor_type=RegressionModelType.QUANTILE_GRADBOOST),
    Sections.collingwood_below_alma: Predictor(section_name=Sections.collingwood_below_alma.name,
                                               mean_predictor_type=RegressionModelType.GRADBOOST,
                                               quantile_predictor_type=RegressionModelType.QUANTILE_GRADBOOST)

}


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/forecast", response_model=schemas.ForecastResult, status_code=200)
async def forecast(forecast_input: schemas.ForecastInput) -> Any:
    """
    Make river level forecasts
    """

    past_data = pd.DataFrame(jsonable_encoder(forecast_input.past_data))
    past_data.index = pd.to_datetime(past_data["datetime"])
    future_rain = pd.DataFrame(jsonable_encoder(forecast_input.future_rain))
    future_rain.index = pd.to_datetime(future_rain["datetime"])
    section = forecast_input.section

    level_history = pd.Series(past_data["level"])
    rain_history = pd.Series(past_data["rain"])
    level_diff_history = pd.Series(past_data["level_diff"])
    future_rain = pd.Series(future_rain["rain"])

    # logger.info(f"Making forecasts on inputs: {forecast_input.inputs}")
    t1 = time.perf_counter()
    results = predictors[section].predict(level_history=level_history,
                                          rain_history=rain_history,
                                          level_diff_history=level_diff_history,
                                          future_rainfall=future_rain)
    t2 = time.perf_counter()
    results["comp_time"] = t2 - t1
    results["model_version"] = model_version

    if results["validation_errors"] is not None:
        logger.warning(f"Prediction validation error: {results.get('validation_errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["validation_errors"]))


    return results
