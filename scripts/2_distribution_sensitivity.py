import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression

from codebase.evaluation.cv import MonteCarloCV
from codebase.workflow import WorkflowModeling, WorkflowPreprocessing
from codebase.algorithms.nn import DeepNet
from codebase.cdf import CDFEngine

from config import (DATA_DIR,
                    EMBED_DIM,
                    HORIZON_LIST,
                    THRESHOLD_PERCENTILE,
                    TARGET)

wave = pd.read_csv(DATA_DIR, parse_dates=['DATE'], index_col='DATE')
# wave = wave.head(2000)
# CV_N_FOLDS = 1
# THRESHOLD_PERCENTILE = .80

results = {}
for horizon_ in HORIZON_LIST:
    # horizon_ = 6
    print(f'Horizon: {horizon_}')

    cv = MonteCarloCV(n_splits=1, train_size=0.69, test_size=0.3)

    X, y, lag_cols, target_cols = \
        WorkflowPreprocessing.series_to_xy(data=wave,
                                           horizon=horizon_,
                                           k=EMBED_DIM,
                                           target_col=TARGET)

    train_index, test_index = cv.split(X).__next__()

    X_train, y_train, X_test, y_test, y_train_clf, y_test_clf, y_std = \
        WorkflowPreprocessing.get_input_output_pairs(X=X, y=y,
                                                     train_index=train_index,
                                                     test_index=test_index,
                                                     lag_cols=lag_cols,
                                                     threshold_perc=THRESHOLD_PERCENTILE)

    thr = np.quantile(y_train, THRESHOLD_PERCENTILE)

    X_dev, X_val, y_dev, y_val, y_val_clf, y_dev_clf = \
        WorkflowPreprocessing.get_validation_set(X_train, y_train, thr)

    model = DeepNet(batch_size=512, n_epochs=30, in_hre=False)
    model_dev = DeepNet(batch_size=512, n_epochs=30, in_hre=False)

    #  training regressor on whole train dataset
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)

    # training calibrator
    model_dev.fit(X_dev, y_dev)
    pred_val = model_dev.predict(X_val)
    cdf_val = CDFEngine()
    cdf_val.fit(y_dev)

    probs_val = cdf_val.get_all_probs(
        y_hat=pred_val,
        threshold=thr)

    isotonic_models = {}
    for mod in probs_val:
        probs_mod = probs_val[mod]

        isotonic = IsotonicRegression(out_of_bounds='clip',
                                      y_min=probs_mod.min(),
                                      y_max=probs_mod.max())
        isotonic.fit(probs_mod, y_val_clf)

        isotonic_models[mod] = isotonic

    cdf_test = CDFEngine()
    cdf_test.fit(y_train)

    probs_test = cdf_test.get_all_probs(
        y_hat=preds_test,
        threshold=thr,
    )

    probs_test_iso = {}
    for mod in probs_test:
        probs_test_iso[mod] = \
            isotonic_models[mod].predict(probs_test[mod])

    clf_metrics_iso, _ = \
        WorkflowModeling.evaluate(probs=probs_test_iso,
                                  forecasts=None,
                                  y_test=y_test,
                                  y_test_clf=y_test_clf)

    clf_metrics, _ = \
        WorkflowModeling.evaluate(probs=probs_test,
                                  forecasts=None,
                                  y_test=y_test,
                                  y_test_clf=y_test_clf)

    results[horizon_] = clf_metrics_iso

    avg_results = pd.concat(results, axis=0).reset_index().groupby(['level_1']).mean()
    results_by_h = pd.concat(results).reset_index()

    results_by_h.to_csv('assets/results/distribution_analysis.csv', index=False)
