import pandas as pd
import numpy as np
from typing import Optional



def validate_inputs(*,
                level_history: pd.Series,
                rain_history: pd.Series,
                level_diff_history: pd.Series,
                future_rainfall: pd.Series,
                min_history: int) -> Optional[dict]:

    try:
        assert len(level_history) == len(rain_history), "past levels and rain must have same length"
        assert len(level_history) == len(level_diff_history), "past levels and level_diff must have same length"
        assert all(np.diff(level_history.index) == pd.Timedelta("1h")), "past levels must be every hour"
        assert all(np.diff(level_diff_history.index) == pd.Timedelta("1h")), "past level_diff must be every hour"
        assert all(np.diff(rain_history.index) == pd.Timedelta("1h")), "past rain must be every hour"
        assert all(np.diff(future_rainfall.index) == pd.Timedelta("1h")), "future rain must be every hour"
        assert (future_rainfall.index[0] - rain_history.index[-1]) == pd.Timedelta("1h"), "future rain must start 1 hour after past rain ends"
        assert rain_history.index[0] == level_history.index[0], "past rain and past levels must start at same datetime"
        assert rain_history.index[0] == level_diff_history.index[0], "past rain and past level_diff must start at same datetime"
        assert len(level_history) >= min_history, f"must provide at least {min_history}h of past data"

        errors = None
    except AssertionError as error:
        errors = error.args[0]

    return errors




