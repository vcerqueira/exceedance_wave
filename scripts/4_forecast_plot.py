import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from plotnine import *

from codebase.evaluation.cv import MonteCarloCV
from codebase.workflow import WorkflowModeling, WorkflowPreprocessing
from codebase.algorithms.nn import DeepNet
from codebase.cdf import CDFEngine

from config import (DATA_DIR,
                    EMBED_DIM,
                    THRESHOLD_PERCENTILE,
                    TARGET)

wave = pd.read_csv(DATA_DIR, parse_dates=['DATE'], index_col='DATE')

cv = MonteCarloCV(n_splits=1, train_size=0.69, test_size=0.3)

X, y, lag_cols, target_cols = WorkflowPreprocessing.series_to_xy(data=wave,
                                                                 horizon=6,
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

model = DeepNet(method='FFNN', batch_size=512, n_epochs=30, in_hre=False)
model_dev = DeepNet(method='FFNN', batch_size=512, n_epochs=30, in_hre=False)

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

# Plotting

df = pd.DataFrame({
    'Forecasts': preds_test,
    'Actual': y_test,
    'Exceedance Probability': probs_test['norm'],
    'Event': y_test_clf,
}).tail(-2000).head(2000)

df['index'] = wave.tail(df.shape[0]).index
dfm = df.melt('index')
dfm['Type'] = dfm['variable'].isin(['Forecasts', 'Actual']).map({True: 'Forecasts', False: 'Exceedance Probability'})

plot = \
    ggplot(dfm) + \
    aes(x='index',
        y='value',
        group='variable',
        color='variable') + \
    theme_538(base_family='Palatino', base_size=14) + \
    theme(plot_margin=.2,
          axis_text=element_text(size=10),
          axis_text_x=element_text(angle=30),
          legend_title=element_blank(),
          legend_position='top') + \
    scale_color_manual(values=['blue', 'blue', 'red', 'red'])

plot += geom_line(alpha=0.6, size=1.1)
plot += facet_wrap('~Type', ncol=1, scales='free_y')

plot = \
    plot + \
    xlab('') + \
    ylab('') + \
    ggtitle('')

plot.save(f'assets/plots/forecasting_example.pdf', height=7, width=15)
