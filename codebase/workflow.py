import copy

import pandas as pd
import numpy as np

from codebase.preprocessing.embed import series_as_supervised
from codebase.learning.rf import ExceedanceRandomForest

from sklearn.model_selection import train_test_split
from sktime.transformations.series.date import DateTimeFeatures
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (roc_auc_score,
                             r2_score, log_loss, brier_score_loss,
                             mean_squared_error as mse,
                             mean_absolute_percentage_error as mape,
                             mean_absolute_error as mae)

from codebase.learning.classification import VanillaClassifier, ResampledClassifier
from codebase.learning.ensemble import HeterogeneousEnsemble
from codebase.algorithms.nn import DeepNet
from codebase.cdf import CDFEngine


class WorkflowPreprocessing:

    @staticmethod
    def get_input_output_pairs(X, y,
                               train_index,
                               test_index,
                               threshold_perc,
                               lag_cols):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        y_std = y_train.std()
        thr = np.quantile(y_train, threshold_perc)

        X_train, y_train = \
            ExceedanceRandomForest.remove_invalid_observations(X=X_train, y=y_train,
                                                               lag_columns=lag_cols,
                                                               decision_thr=thr)
        X_test, y_test = \
            ExceedanceRandomForest.remove_invalid_observations(X=X_test, y=y_test,
                                                               lag_columns=lag_cols,
                                                               decision_thr=thr)

        y_train_clf = (y_train >= thr).astype(int)
        y_test_clf = (y_test >= thr).astype(int)

        return X_train, y_train, X_test, y_test, y_train_clf, y_test_clf, y_std

    @staticmethod
    def get_validation_set(X, y, threshold):
        X_dev, X_val, y_dev, y_val = train_test_split(X, y, test_size=0.3)

        y_val_clf = (y_val >= threshold).astype(int)
        y_dev_clf = (y_dev >= threshold).astype(int)

        return X_dev, X_val, y_dev, y_val, y_val_clf, y_dev_clf

    @staticmethod
    def series_to_xy(data: pd.DataFrame,
                     horizon: int,
                     k: int,
                     target_col: str):
        data_set, lag_cols, target_cols = \
            series_as_supervised(data=data,
                                 horizon=horizon,
                                 k=k,
                                 target_col=target_col)

        X = data_set.drop(target_cols, axis=1)
        y = data_set[target_cols].iloc[:, -1]

        date_features = DateTimeFeatures(ts_freq='H',
                                         keep_original_columns=False,
                                         feature_scope='efficient')

        dates = date_features.fit_transform(X.iloc[:, -1])
        dates = dates[['day_of_year']]
        X = pd.concat([X, dates], axis=1)

        return X, y, lag_cols, target_cols


class WorkflowModeling:

    def __init__(self, threshold: float, std_dev: float):
        self.models = {
            'RFC': VanillaClassifier(model=RandomForestClassifier()),
            'RFC+SMOTE': ResampledClassifier(model=RandomForestClassifier(), resampling_model=SMOTE()),
            'LR': VanillaClassifier(model=LogisticRegression()),
            'RFR': ExceedanceRandomForest(),
            'LASSO': LassoCV(),
            'HRE': HeterogeneousEnsemble(),
            'NN': DeepNet(batch_size=512, n_epochs=30, in_hre=False),
        }

        self.models_dev = copy.deepcopy(self.models)
        self.isotonic_models = {}

        self.threshold = threshold
        self.std_dev = std_dev

    def fit(self, X_train, y_train, y_train_clf):
        self.models['RFC'].fit(X_train, y_train_clf)
        self.models['RFC+SMOTE'].fit(X_train, y_train_clf)
        self.models['LR'].fit(X_train, y_train_clf)
        self.models['RFR'].fit(X_train, y_train)
        self.models['LASSO'].fit(X_train, y_train)
        self.models['HRE'].fit_and_trim(X_train, y_train, select_percentile=.5)
        self.models['NN'].fit(X_train, y_train)

    def fit_isotonic(self, X_dev, y_dev, y_dev_clf, X_val, y_val_clf):

        self.models_dev['RFC'].fit(X_dev, y_dev_clf)
        self.models_dev['RFC+SMOTE'].fit(X_dev, y_dev_clf)
        self.models_dev['LR'].fit(X_dev, y_dev_clf)
        self.models_dev['RFR'].fit(X_dev, y_dev)
        self.models_dev['LASSO'].fit(X_dev, y_dev)
        self.models_dev['HRE'].fit_and_trim(X_dev, y_dev, select_percentile=.5)
        self.models_dev['NN'].fit(X_dev, y_dev)

        probs = {
            'RFC': self.models['RFC'].predict_proba(X_val),
            'RFC+SMOTE': self.models['RFC+SMOTE'].predict_proba(X_val),
            'LR': self.models['LR'].predict_proba(X_val),
            'RFR': self.models['RFR'].predict_exceedance_proba(X_val, self.threshold),
            'HRE': self.models['HRE'].predict_proba(X_val, self.threshold),
            'RFR+CDF': self.models['RFR'].predict(X_val),
            'HRE+CDF': self.models['HRE'].predict(X_val),
            'LASSO+CDF': self.models['LASSO'].predict(X_val),
            'NN+CDF': self.models['NN'].predict(X_val),
        }

        cdf_eng = CDFEngine()
        cdf_eng.fit(y_dev)

        for mod in ['RFR+CDF', 'LASSO+CDF', 'NN+CDF', 'HRE+CDF']:
            probs[mod] = cdf_eng.get_probs(
                y_hat=probs[mod],
                distribution='gumbel_r',
                threshold=self.threshold,
            )

        for mod in probs:
            probs_mod = probs[mod]

            isotonic = IsotonicRegression(out_of_bounds='clip',
                                          y_min=probs_mod.min(),
                                          y_max=probs_mod.max())
            isotonic.fit(probs_mod, y_val_clf)

            self.isotonic_models[mod] = isotonic

    def predict_isotonic(self, test_probs):
        isotonic_preds = {}
        for mod in test_probs:
            isotonic_preds[mod] = self.isotonic_models[mod].predict(test_probs[mod])

        return isotonic_preds

    def predict_proba_direct(self, X_test):
        probs = {
            'RFC': self.models['RFC'].predict_proba(X_test),
            'RFC+SMOTE': self.models['RFC+SMOTE'].predict_proba(X_test),
            'LR': self.models['LR'].predict_proba(X_test),
            'RFR': self.models['RFR'].predict_exceedance_proba(X_test, self.threshold),
            'HRE': self.models['HRE'].predict_proba(X_test, self.threshold),
        }

        return probs

    def predict_num(self, X_test):
        forecasts = {
            'RFR': self.models['RFR'].predict(X_test),
            'HRE': self.models['HRE'].predict(X_test),
            'LASSO': self.models['LASSO'].predict(X_test),
            'NN': self.models['NN'].predict(X_test),
        }

        return forecasts

    def predict_proba_cdf(self, X_test, y_train):
        preds = self.predict_num(X_test)

        cdf_eng = CDFEngine()
        cdf_eng.fit(y_train)

        probs = {}
        for mod in preds:
            probs[f'{mod}+CDF'] = cdf_eng.get_probs(
                y_hat=preds[mod],
                distribution='gumbel_r',
                threshold=self.threshold,
            )

        return probs

    @staticmethod
    def evaluate(probs, forecasts, y_test, y_test_clf):

        clf_metrics = {}
        for mod in probs:
            clf_metrics[mod] = {
                'AUC': roc_auc_score(y_true=y_test_clf, y_score=probs[mod]),
                'LL': log_loss(y_true=y_test_clf, y_pred=probs[mod]),
                'BRIER': brier_score_loss(y_true=y_test_clf, y_prob=probs[mod]),
            }

        if forecasts is not None:

            num_scores = {}
            for mod in forecasts:
                num_scores[mod] = {
                    'R2': r2_score(y_true=y_test, y_pred=forecasts[mod]),
                    'MAE': mae(y_true=y_test, y_pred=forecasts[mod]),
                    'RMSE': mse(y_true=y_test, y_pred=forecasts[mod], squared=False),
                    'MAPE': mape(y_true=y_test, y_pred=forecasts[mod]),
                }

            num_scores_df = pd.DataFrame(num_scores).T
        else:
            num_scores_df = None

        clf_metrics_df = pd.DataFrame(clf_metrics).T

        return clf_metrics_df, num_scores_df
