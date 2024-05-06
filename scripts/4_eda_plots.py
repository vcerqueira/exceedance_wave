import pandas as pd
import numpy as np
from sktime.transformations.series.date import DateTimeFeatures
from plotnine import *
from statsmodels.tsa.stattools import acf, pacf

from config import DATA_DIR

wave = pd.read_csv(DATA_DIR, parse_dates=['DATE'], index_col='DATE')

date_features = DateTimeFeatures(ts_freq='H',
                                 keep_original_columns=False,
                                 feature_scope='efficient')

dates = date_features.fit_transform(wave.iloc[:, -1])
dates = dates[['day_of_year']]
wave = pd.concat([wave, dates], axis=1)
wave = wave.rename(columns={'day_of_year': 'DAY'})

mat = wave.corr()
mat = mat.reset_index().melt('index')
mat = mat.rename(columns={'value': 'Correlation'})

# HEATMAP PLOT
hm_plot = ggplot(mat, aes('index', 'variable', fill='Correlation')) + \
          geom_tile(aes(width=.95, height=.95)) + \
          theme_classic(base_family='Palatino',
                        base_size=12) + \
          scale_fill_gradient2(low='red',
                               mid='white',
                               high='blue',
                               midpoint=0) + \
          theme(
              axis_ticks=element_blank(),
              plot_margin=.2,
              panel_background=element_rect(fill='white'),
          ) + \
          labs(x='', y='')

# ACF PLOT

acf_x = pacf(
    wave['VCAR'].ffill(),
    nlags=24,
    alpha=.05,
)

acf_, conf_int = acf_x[:2]

acf_df = pd.DataFrame({
    'ACF': acf_,
    'ACF_low': conf_int[:, 0],
    'ACF_high': conf_int[:, 1],
})

acf_df['Lag'] = ['t'] + [f't-{i}' for i in range(1, 24 + 1)]
acf_df['Lag'] = pd.Categorical(acf_df['Lag'], categories=acf_df['Lag'].unique())

plot = ggplot(acf_df, aes(x='Lag', y='ACF'))
plot += geom_hline(yintercept=0, linetype='solid', color='black', size=1)

significance_thr = 2 / np.sqrt(wave.shape[0])

plot = \
    plot + geom_segment(
        aes(x='Lag',
            xend='Lag',
            y=0,
            yend='ACF'),
        size=1.5,
        color='steelblue'
    ) + \
    geom_point(
        size=4,
        color='steelblue',
    ) + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.125,
          axis_text_y=element_text(size=12),
          axis_text_x=element_text(size=10),
          panel_background=element_rect(fill='white'),
          plot_background=element_rect(fill='white'),
          strip_background=element_rect(fill='white'),
          legend_background=element_rect(fill='white'),
          legend_title=element_blank(),
          legend_position='none')

plot = plot + \
       xlab('Lag') + \
       ylab('ACF') + \
       ggtitle('')

hm_plot.save(f'assets/plots/heatmap2.pdf', height=7, width=7)
plot.save(f'assets/plots/acf.pdf', height=7, width=15)
