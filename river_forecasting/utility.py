from typing import Union
import numpy as np
import pandas as pd

def forecast_error(pred:Union[pd.DataFrame, np.ndarray],
                   true:Union[pd.DataFrame, np.ndarray],
                   mode:str='mse') -> np.ndarray:
    """
    Calculates the error as a function of steps into the future
    """
    error = np.array(pred) - np.array(true)
    if mode=='mse':
        score = np.mean(error**2, axis=0)
    elif mode=='rmse':
        score = np.sqrt(np.mean(error ** 2, axis=0))
    elif mode=='mae':
        score = np.sqrt(np.mean(np.abs(error), axis=0))
    else:
        raise NotImplementedError(f"forecast error is not implemented for {mode}")

    return score


def variable_expander(variable:Union[str,list], all_variables:list):
    l = []
    if isinstance(variable, list):
        for v in all_variables:
            if variable in v:
                l.append(v)
    if len(l) > 0:
        return l
    else:
        return variable