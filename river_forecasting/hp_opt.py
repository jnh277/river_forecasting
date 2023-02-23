import numpy as np
import pandas as pd
import contextlib
from typing import Union
import sklearn.pipeline as Pipeline
from sklearn.metrics import mean_squared_error, mean_pinball_loss
from tqdm import tqdm
from functools import partial
from hyperopt import fmin, hp, tpe, space_eval, STATUS_OK, Trials, STATUS_FAIL, progress
from hyperopt.pyll import scope

from river_forecasting.models import RegressionModelType
from river_forecasting.models import init_scikit_pipe

tuning_spaces = {
    RegressionModelType.RF: {
        'n_estimators': scope.int(hp.qloguniform('n_estimators', np.log(30), np.log(1000), 1)),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'max_depth': scope.int(hp.quniform('max_depth', 2, 30, 1)),
        'min_impurity_decrease': hp.loguniform('min_impurity_decrease', np.log(1e-10), 0),
        'max_features': hp.uniform('max_features', 0.5, 1.),
        "min_samples_leaf": scope.int(hp.qloguniform("min_samples_leaf", np.log(1), np.log(20), 1)),
    },
    RegressionModelType.GRADBOOST: {
        'learning_rate': hp.loguniform('learning_rate', -8, -2.3),  # 0.001 to 0.1,
        'n_estimators': scope.int(hp.qloguniform('n_estimators', np.log(30), np.log(1000), 1)),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'max_depth': scope.int(hp.quniform('max_depth', 2, 30, 1)),
        'min_impurity_decrease': hp.loguniform('min_impurity_decrease', np.log(1e-10), 0),
        'max_features': hp.uniform('max_features', 0.5, 1.),
        "min_samples_leaf":scope.int(hp.qloguniform("min_samples_leaf", np.log(1), np.log(20), 1)),
    },
    RegressionModelType.RIDGE: {
        "alpha": hp.uniform("alpha", 0, 10)
    },
    RegressionModelType.XGBOOST: {
        'learning_rate': hp.loguniform('learning_rate', -8, -2.3),  # 0.001 to 0.1
        'max_depth': scope.int(hp.quniform('max_depth', 2, 30, 1)),
        'min_child_weight': hp.loguniform('min_child_weight', np.log(1e-6), np.log(32.0)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'gamma': hp.loguniform('gamma', -16, 3),
        'eta': hp.loguniform('eta', np.log(1e-3), 0),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-6), np.log(10.0)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-6), np.log(2.0)),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'n_estimators': scope.int(hp.qloguniform('n_estimators', np.log(30), np.log(1000), 1)),
        # 'n_estimators': 100,  # allow early stopping instead of changing this
    },
    RegressionModelType.KNN: {
        "n_neighbors": hp.choice("n_neighbors", range(1, 100))
    },
    RegressionModelType.QUANTILE_GRADBOOST: {
        'learning_rate': hp.loguniform('learning_rate', -8, -2.3),  # 0.001 to 0.1,
        'n_estimators': scope.int(hp.qloguniform('n_estimators', np.log(30), np.log(1000), 1)),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'max_depth': scope.int(hp.quniform('max_depth', 2, 30, 1)),
        'min_impurity_decrease': hp.loguniform('min_impurity_decrease', np.log(1e-10), 0),
        'max_features': hp.uniform('max_features', 0.5, 1.),
        "min_samples_leaf": scope.int(hp.qloguniform("min_samples_leaf", np.log(1), np.log(20), 1)),
    }

}


@contextlib.contextmanager
def tqdm_progress_callback(initial, total):
    with tqdm(
        total=total,
        postfix={"best loss": "?"},
        disable=False,
        dynamic_ncols=True,
        unit="trial",
        initial=initial,
        position=3,
        leave=False
    ) as pbar:
        yield pbar

def minimization_function(selected_hp_values,
                          pipe: Pipeline,
                          X_train: pd.DataFrame,
                          y_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_test: pd.DataFrame,
                          quantile: Union[None, float]):

    try:
        pipe[-1].set_params(**selected_hp_values)
        pipe.fit(X_train, y_train)
        y_hat = pipe.predict(X_test)

        if quantile is None:
            loss = mean_squared_error(y_test, y_hat)
        else:
            loss = mean_pinball_loss(y_test, y_hat, alpha=quantile)
        status=STATUS_OK

    except ValueError:
        loss = 100
        status=STATUS_FAIL

    return {'loss': loss, 'status': status}


def run_hpopt(
        model_type: RegressionModelType,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        quantile: Union[float, None],
) -> Pipeline:

    pipe = init_scikit_pipe(model_type, quantile)
    pipe.fit(X_train, y_train)
    if quantile:
        loss = mean_pinball_loss(y_test, pipe.predict(X_test), alpha=quantile)
    else:
        loss = mean_squared_error(y_test, pipe.predict(X_test))

    trials = Trials()
    tuning = fmin(fn=partial(minimization_function,
                             pipe=pipe,
                             X_train=X_train,
                             y_train=y_train,
                             X_test=X_test,
                             y_test=y_test,
                             quantile=quantile,
                             ),
                  space=tuning_spaces[model_type],
                  algo=tpe.suggest,
                  max_evals=len(tuning_spaces[model_type].keys())*30,
                  trials=trials,
                  show_progressbar=tqdm_progress_callback
                  )


    # get the best run from above
    best_params = space_eval(tuning_spaces[model_type], tuning)
    best_loss = trials.best_trial["result"]["loss"]
    if best_loss < loss:
        pipe[-1].set_params(**best_params)
        pipe.fit(X_train, y_train)
    else:
        pipe = init_scikit_pipe(model_type, quantile)
        pipe.fit(X_train, y_train)

    return pipe
