from typing import List

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class ExceedanceRandomForest(RandomForestRegressor):

    def __init__(self):
        super().__init__()

    def predict_exceedance_proba(self, X: pd.DataFrame, thr: float) -> pd.DataFrame:
        """
        Extract exceedance probability from RF
        :param X: predictor variables
        :param thr: exceedance threshold
        :return: exceedance probability
        """

        per_tree_pred = [tree.predict(X) for tree in self.estimators_]

        preds_all = pd.DataFrame(per_tree_pred).T

        rf_prob_ = preds_all.apply(lambda x: np.mean(x > thr), axis=1).values

        return rf_prob_

    def predict_exceedance(self, X: pd.DataFrame, thr: float) -> pd.DataFrame:
        """
        Extract exceedance predictions from RF
        :param X: predictor variables
        :param thr: exceedance threshold
        :return: exceedance predictions
        """

        pred_num = self.predict(X)

        pred_exceedance = (pred_num > thr).astype(int)

        return pred_exceedance

    @staticmethod
    def remove_invalid_observations(X: pd.DataFrame,
                                    y: pd.Series,
                                    lag_columns: List[str],
                                    decision_thr: float):
        """
        removing observations in which the phenomena (y>=thr) already occurs in the input
        :param X: predictors as pd.DF
        :param y: target variable
        :param lag_columns: predictors relative to the target variable (lags)
        :param decision_thr: decision thr
        :return:
        """

        if isinstance(y, pd.Series):
            y = y.values

        idx_to_kp = ~(X[lag_columns] >= decision_thr).any(axis=1)

        X_t = X.loc[idx_to_kp, :].reset_index(drop=True).copy()
        y_t = y[idx_to_kp]

        return X_t, y_t
