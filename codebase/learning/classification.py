from copy import deepcopy

import numpy as np
import pandas as pd


class VanillaClassifier:
    """
    Binary classifier, no resampling
    """

    def __init__(self, model):
        self.model = deepcopy(model)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        yh_p = self.model.predict_proba(X)
        yh_prob = np.array([x[1] for x in yh_p])

        return yh_prob


class ResampledClassifier:
    """
    Binary classifier with resampling
    """

    def __init__(self, model, resampling_model):
        self.model = deepcopy(model)
        self.resampler = deepcopy(resampling_model)

    def fit(self, X, y):
        print(pd.Series(y).value_counts())
        X_r, y_r = self.resampler.fit_resample(X, y)

        self.model.fit(X_r, y_r)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        yh_p = self.model.predict_proba(X)
        yh_prob = np.array([x[1] for x in yh_p])

        return yh_prob
