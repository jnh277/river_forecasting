# standard
import numpy as np

np.random.seed(1)  # fix seed for notebook
import pandas as pd
import scipy

# ML
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

# utils
from functools import partial

# plotting libs
import matplotlib.pyplot as plt
import seaborn as sns

from river_forecasting.models import XGBQuantile


def generate_data():
    """
    Generates data sample as seen in "Prediction Intervals for Gradient Boosting Regression"
    (https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
    """
    np.random.seed(1)
    f = lambda u: u * np.sin(u)

    #  First the noiseless case
    X_train = np.atleast_2d(np.random.uniform(0, 10.0, size=100)).T
    X_train = X_train.astype(np.float32)

    # Observations
    y_train = f(X_train).ravel()
    dy = 1.5 + 1.0 * np.random.random(y_train.shape)
    noise = np.random.normal(0, dy)
    y_train += noise
    y_train = y_train.astype(np.float32)

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    X_test = np.atleast_2d(np.linspace(0, 10.0, 1000)).T
    X_test = X_test.astype(np.float32)
    y_test = f(X_test).ravel()

    return X_train, y_train, X_test, y_test


def collect_prediction(X_train, y_train, X_test, y_test, estimator, alpha, model_name):
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print("{model_name} alpha = {alpha:.2f},score = {score:.1f}".format(model_name=model_name, alpha=alpha,
                                                                        score=XGBQuantile.quantile_score(y_test, y_pred,
                                                                                                         alpha)))

    return y_pred


def plot_result(X_train, y_train, X_test, y_test, y_upper, y_lower):
    plt.plot(X_test, y_test, 'g:', label=u'$f(x) = x\,\sin(x)$')
    plt.plot(X_train, y_train, 'b.', markersize=10, label=u'Observations')
    plt.plot(X_test, y_pred, 'r-', label=u'Prediction')
    plt.plot(X_test, y_upper, 'k-')
    plt.plot(X_test, y_lower, 'k-')
    plt.fill(np.concatenate([X_test, X_test[::-1]]),
             np.concatenate([y_upper, y_lower[::-1]]),
             alpha=.5, fc='b', ec='None', label='90% prediction interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')


alpha = 0.95  # @param {type:"number"}

X_train, y_train, X_test, y_test = generate_data()

regressor = GradientBoostingRegressor(n_estimators=250, max_depth=3,
                                      learning_rate=.1, min_samples_leaf=9,
                                      min_samples_split=9)
y_pred = regressor.fit(X_train, y_train).predict(X_test)
regressor.set_params(loss='quantile', alpha=1. - alpha)
y_lower = collect_prediction(X_train, y_train, X_test, y_test, estimator=regressor, alpha=1. - alpha,
                             model_name="Gradient Boosting")
regressor.set_params(loss='quantile', alpha=alpha)
y_upper = collect_prediction(X_train, y_train, X_test, y_test, estimator=regressor, alpha=alpha,
                             model_name="Gradient Boosting")

fig = plt.figure(figsize=(12, 9))

plt.subplot(311)
plt.title("Prediction Interval Gradient Boosting")
plot_result(X_train, y_train, X_test, y_test, y_upper, y_lower)

regressor = XGBRegressor(n_estimators=250,max_depth=3,reg_alpha=5, reg_lambda=1,gamma=0.5)
y_pred = regressor.fit(X_train,y_train).predict(X_test)

regressor = XGBQuantile(n_estimators=100,max_depth = 3, reg_alpha =5.0,gamma = 0.5,reg_lambda =1.0 )
regressor.set_params(quant_alpha=1.-alpha,quant_delta=1.0,quant_thres=5.0,quant_var=3.2)

y_lower = collect_prediction(X_train,y_train,X_test,y_test,estimator=regressor,alpha=1.-alpha,model_name="Quantile XGB")
regressor.set_params(quant_alpha=alpha,quant_delta=1.0,quant_thres=6.0,quant_var = 4.2)
y_upper = collect_prediction(X_train,y_train,X_test,y_test,estimator=regressor,alpha=alpha,model_name="Quantile XGB")

plt.subplot(312)
plt.title("Prediction Interval XGBoost")
plot_result(X_train,y_train,X_test,y_test,y_upper,y_lower)



from river_forecasting.models import log_cosh_quantile

alpha = 0.05
model = XGBRegressor(objective=log_cosh_quantile(alpha),
                     n_estimators=100,
                     max_depth=3,
                     n_jobs=6,
                     learning_rate=.05)

y_lower = model.fit(X_train,y_train).predict(X_test)

alpha = 0.95
model2 = XGBRegressor(objective=log_cosh_quantile(alpha),
                     n_estimators=100,
                     max_depth=6,
                     n_jobs=6,
                     learning_rate=.05)

y_upper = model2.fit(X_train,y_train).predict(X_test)

plt.subplot(313)
plt.title("Prediction Interval my XGBoost")
plot_result(X_train,y_train,X_test,y_test,y_upper,y_lower)

plt.show()
