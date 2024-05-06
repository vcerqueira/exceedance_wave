import numpy as np
import pandas as pd

from codebase.evaluation.cv import MonteCarloCV
from codebase.workflow import WorkflowModeling, WorkflowPreprocessing

from config import (DATA_DIR,
                    EMBED_DIM,
                    HORIZON_LIST,
                    THRESHOLD_PERCENTILE,
                    CV_N_FOLDS,
                    TRAIN_SIZE,
                    TARGET,
                    TEST_SIZE)

wave = pd.read_csv(DATA_DIR, parse_dates=['DATE'], index_col='DATE')
# wave = wave.head(2000)
# CV_N_FOLDS = 1
# THRESHOLD_PERCENTILE = .80

results_clf, results_iclf, results_num = {}, {}, {}
for horizon_ in HORIZON_LIST:
    print(f'Horizon: {horizon_}')

    cv = MonteCarloCV(n_splits=CV_N_FOLDS, train_size=TRAIN_SIZE, test_size=TEST_SIZE)

    X, y, lag_cols, target_cols = \
        WorkflowPreprocessing.series_to_xy(data=wave,
                                           horizon=horizon_,
                                           k=EMBED_DIM,
                                           target_col=TARGET)

    horizon_clf, horizon_iclf, horizon_num = [], [], []
    for train_index, test_index in cv.split(X):
        print('.')
        X_train, y_train, X_test, y_test, y_train_clf, y_test_clf, y_std = \
            WorkflowPreprocessing.get_input_output_pairs(X=X, y=y,
                                                         train_index=train_index,
                                                         test_index=test_index,
                                                         lag_cols=lag_cols,
                                                         threshold_perc=THRESHOLD_PERCENTILE)

        thr = np.quantile(y_train, THRESHOLD_PERCENTILE)

        X_dev, X_val, y_dev, y_val, y_val_clf, y_dev_clf = \
            WorkflowPreprocessing.get_validation_set(X_train, y_train, thr)

        models = WorkflowModeling(threshold=thr, std_dev=y_std)

        models.fit(X_train, y_train, y_train_clf)

        models.fit_isotonic(X_dev, y_dev, y_dev_clf, X_val, y_val_clf)

        probs_clf = models.predict_proba_direct(X_test)
        probs_cdf = models.predict_proba_cdf(X_test, y_train)
        forecasts = models.predict_num(X_test)

        exceedance_prob = {**probs_clf, **probs_cdf}
        exceedance_prob_iso = models.predict_isotonic(exceedance_prob)

        clf_metrics, _ = \
            models.evaluate(probs=exceedance_prob,
                            forecasts=forecasts,
                            y_test=y_test,
                            y_test_clf=y_test_clf)

        clf_metrics_iso, num_scores_iso = \
            models.evaluate(probs=exceedance_prob_iso,
                            forecasts=forecasts,
                            y_test=y_test,
                            y_test_clf=y_test_clf)

        horizon_clf.append(clf_metrics_iso)
        horizon_iclf.append(clf_metrics)
        horizon_num.append(num_scores_iso)

    horizon_clf_df = pd.concat(horizon_clf, axis=0).reset_index().groupby('index').mean()
    horizon_iclf_df = pd.concat(horizon_iclf, axis=0).reset_index().groupby('index').mean()
    horizon_num_df = pd.concat(horizon_num, axis=0).reset_index().groupby('index').mean()

    results_clf[horizon_] = horizon_clf_df
    results_iclf[horizon_] = horizon_iclf_df
    results_num[horizon_] = horizon_num_df

    clf_df = pd.concat(results_clf, axis=0).reset_index()
    iclf_df = pd.concat(results_iclf, axis=0).reset_index()
    num_df = pd.concat(results_num, axis=0).reset_index()

    clf_df.to_csv('assets/results/results_clf.csv', index=False)
    iclf_df.to_csv('assets/results/results_clf_iso.csv', index=False)
    num_df.to_csv('assets/results/results_num.csv', index=False)
