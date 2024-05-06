import matplotlib

matplotlib.use('TkAgg')
import pandas as pd
from baycomp import two_on_single
from plotnine import *

from codebase.data_reader import reading_data_results

perf_clf, perf_n_df, sens_result = reading_data_results()

# NUMERIC PERFORMANCE

num_metric_melt = perf_n_df.melt(['Metric', 'Horizon'])

num_metrics_plot = \
    ggplot(num_metric_melt, aes(x='index',
                                y='value',
                                fill='index')) + \
    geom_boxplot(width=0.7) + \
    theme_538(
        base_family='Palatino',
        base_size=12) + \
    theme(
        plot_margin=.2,
        axis_text_x=element_text(size=11),
        axis_text_y=element_text(size=12),
        legend_text=element_text(size=8),
        legend_title=element_text(size=8),
        legend_position='none') + \
    xlab('') + \
    ylab('Metric value') + facet_wrap('~Metric', scales='free_y')

# CLASSIFICATION METRICS

linetype_dict = {
    'LASSO+CDF': 'solid',
    'RFR+CDF': 'solid',
    'RFR+D': 'dotted',
    'HRE+D': 'dotted',
    'HRE+CDF': 'solid',
    'NN+CDF': 'solid',
    'RFC+SMOTE': 'longdash',
    'RFC': 'longdash',
    'LR': 'longdash'
}

ord = perf_clf.drop('Horizon', axis=1).mean().sort_values(ascending=False).index.to_list()

perf_clf_m = perf_clf.melt(['Horizon'])
perf_clf_m['index'] = pd.Categorical(perf_clf_m['index'], categories=ord)
perf_clf_m['Horizon'] = pd.Categorical(perf_clf_m['Horizon'], categories=perf_clf_m['Horizon'].unique())
perf_clf_m['linetype'] = perf_clf_m['index'].map(linetype_dict)
perf_clf_m['linetype'] = pd.Categorical(perf_clf_m['linetype'], categories=['solid', 'dotted', 'longdash'])

clf_auc_plot = \
    ggplot(perf_clf_m, aes(x=1, y='value')) + \
    facet_wrap('~ index', nrow=1, scales='free_x') + \
    geom_boxplot() + \
    theme_538() + \
    labs(x='', y='AUC') + \
    theme(axis_text_x=element_blank(),
          axis_text_y=element_text(size=12),
          axis_title_y=element_text(size=12),
          strip_text_x=element_text(angle=0, size=12))

reference = 'NN+CDF'
pd_clf_metric = perf_clf.drop(['Horizon'], axis=1)
for col in pd_clf_metric:
    pd_clf_metric[col] = 100 * ((perf_clf[reference] - perf_clf[col]) / perf_clf[reference])

corr_tt_results = {}
for col in pd_clf_metric:
    res = two_on_single(x=pd_clf_metric[col], y=pd_clf_metric[reference], rope=1)
    corr_tt_results[col] = {
        f'{reference} wins': res[0],
        'draw': res[1],
        f'{reference} loses': res[2],
    }

df = pd.DataFrame(corr_tt_results).T
df.drop('NN+CDF', inplace=True)
df_melted = df.reset_index().melt('index')
df_melted['variable'] = pd.Categorical(df_melted['variable'],
                                       categories=['NN+CDF loses', 'draw', 'NN+CDF wins'])

bayes_plot = \
    ggplot(df_melted, aes(fill='variable', y='value', x='index')) + \
    geom_bar(position='stack', stat='identity') + \
    theme_538(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.25,
          axis_text=element_text(size=12),
          axis_text_x=element_text(size=10, angle=0),
          legend_title=element_blank(),
          legend_position='top') + \
    labs(x='', y='') + scale_fill_hue()

plot_fh = \
    ggplot(data=perf_clf_m,
           mapping=aes(x='Horizon',
                       y='value',
                       color='index',
                       group='index', linetype='linetype')) + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.175,
          axis_text=element_text(size=12),
          strip_text=element_text(size=14),
          axis_text_x=element_text(size=12, angle=0),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_line(size=1) + \
    geom_point(group='variable') + \
    labs(x='Forecasting Horizon',
         y='AUC') + \
    guides(linetype=False)

## DISTRIBUTION SENSITIVITY

sens_result = sens_result.set_index('Horizon')
sens_result = sens_result.query('Distribution!="Laplace"')
# sens_result = sens_result.query('Distribution!="Cauchy"')
sens_result = sens_result.query('Distribution!="Weibull2"')
sens_result = sens_result.query('Distribution!="InvWeibull"')
sens_melted = sens_result.groupby('Distribution').mean().reset_index().melt('Distribution')

sens_metrics_plot = \
    ggplot(sens_melted, aes(x='Distribution', y='value', fill='Distribution')) + \
    geom_bar(position='dodge', stat='identity', width=0.6) + \
    theme_538(
        base_family='Palatino',
        base_size=12) + \
    theme(
        plot_margin=.2,
        axis_text_x=element_text(size=11, angle=60),
        axis_text_y=element_text(size=12),
        strip_text=element_text(size=12),
        legend_text=element_text(size=8),
        legend_title=element_text(size=8),
        legend_position='none') + \
    xlab('') + \
    ylab('') + \
    facet_wrap('~variable', scales='free_y', ncol=1)

sens_result_m = sens_result.drop('AUC', axis=1).melt(['Horizon', 'Distribution'])
sens_result_m['Horizon'] = pd.Categorical(sens_result_m['Horizon'], categories=sens_result_m['Horizon'].unique())

plot_fh_sens = \
    ggplot(data=sens_result_m,
           mapping=aes(x='Horizon',
                       y='value',
                       color='Distribution',
                       group='Distribution')) + \
    theme_classic(base_family='Palatino', base_size=12) + \
    theme(plot_margin=.175,
          axis_text=element_text(size=12),
          strip_text=element_text(size=14),
          axis_text_x=element_text(size=12, angle=0),
          legend_title=element_blank(),
          legend_position='top') + \
    geom_line(size=1) + \
    geom_point(group='variable') + \
    labs(x='Forecasting Horizon',
         y='AUC') + \
    guides(linetype=False)

sens_metrics_plot.save(f'assets/plots/sens_metrics_plot.pdf', height=7, width=15)
plot_fh.save(f'assets/plots/auc_over_fh.pdf', height=7, width=15)
clf_auc_plot.save(f'assets/plots/clf_auc_plot.pdf', height=5, width=13)
num_metrics_plot.save('assets/plots/num_metrics_plot.pdf', height=7, width=16)
bayes_plot.save('assets/plots/bayes_plot.pdf', height=5, width=13)
