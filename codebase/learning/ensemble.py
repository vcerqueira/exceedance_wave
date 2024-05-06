import time

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

from codebase.learning.config import (METHODS,
                                      METHODS_PARAMETERS,
                                      MODELS_ON_SUBSET)
from codebase.utils import expand_grid_all

SUBSET_N = 1000


class HeterogeneousEnsemble:

    def __init__(self):
        self.models = {}
        self.err = {}
        self.time = {}
        self.failed = []
        self.selected_methods = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if isinstance(y, pd.Series):
            y = y.values

        for learning_method in METHODS:
            print(f'Creating {learning_method}')
            if len(METHODS_PARAMETERS[learning_method]) > 0:
                gs_df = expand_grid_all(METHODS_PARAMETERS[learning_method])

                n_gs = len(gs_df[[*gs_df][0]])
                for i in range(n_gs):

                    print(f'Training {i} out of {n_gs}')

                    pars = {k: gs_df[k][i] for k in gs_df}
                    pars = {p: pars[p] for p in pars if pars[p] is not None}
                    print(pars)

                    model = METHODS[learning_method](**pars)
                    try:
                        start = time.time()

                        if learning_method in MODELS_ON_SUBSET:
                            print(f'Training {learning_method} on a subset of {SUBSET_N} data points')
                            model.fit(X.tail(SUBSET_N), y[-SUBSET_N:])
                        else:
                            model.fit(X, y)
                        end_t = time.time() - start

                        self.models[f'{learning_method}_{i}'] = model
                        self.time[f'{learning_method}_{i}'] = end_t
                    except (ValueError, KeyError) as e:
                        continue
            else:
                model = METHODS[learning_method]()
                try:
                    start = time.time()
                    if learning_method in MODELS_ON_SUBSET:
                        model.fit(X.tail(SUBSET_N), y[-SUBSET_N:])
                    else:
                        model.fit(X, y)
                    end_t = time.time() - start

                    self.time[f'{learning_method}_0'] = end_t
                except (ValueError, KeyError) as e:
                    continue

    def fit_and_trim(self, X, y, select_percentile: float = .75):

        self.fit(X, y)

        preds = self.predict_all(X)

        for m in preds:
            self.err[m] = mean_absolute_error(y, preds[m])

        err_series = pd.Series(self.err)
        self.selected_methods = err_series[err_series < err_series.quantile(select_percentile)].index.tolist()

        self.models = {k: self.models[k] for k in self.selected_methods}

    def predict_all(self, X: pd.DataFrame):

        preds_all = {}
        for method_ in self.models:
            print(method_)
            preds_all[method_] = self.models[method_].predict(X).flatten()

        preds = pd.DataFrame(preds_all)

        return preds

    def predict(self, X: pd.DataFrame):
        preds_all = self.predict_all(X)

        preds_mean = preds_all.mean(axis=1).values

        return preds_mean

    def score(self, X, y):

        preds = self.predict_all(X)

        err = {}
        for k in preds:
            err[k] = mean_absolute_error(y, preds[k])

        return pd.Series(err)

    def predict_proba(self, X, thr):

        preds = self.predict_all(X)

        predsv = preds.apply(lambda k: np.mean(k > thr), axis=1).values

        return predsv

    def predict_exceedance(self, X, thr):

        preds = self.predict(X)

        predsv = (preds > thr).astype(int)

        return predsv
