import pandas as pd


def reading_data_results():
    perf_clf = pd.read_csv('assets/results/results_clf.csv')
    perf_num = pd.read_csv('assets/results/results_num.csv')
    sens_result = pd.read_csv('assets/results/distribution_analysis.csv')

    perf_clf = perf_clf.pivot(index='level_0', columns='index', values='AUC')
    perf_clf = perf_clf.reset_index().rename(columns={'level_0': 'Horizon',
                                                      'HRE': 'HRE+D',
                                                      'RFR': 'RFR+D'})
    perf_clf['Horizon'] = [f't+{i}' for i in perf_clf['Horizon']]

    perf_num_m = perf_num.melt(['level_0', 'index'])
    perf_g = perf_num_m.groupby('variable')
    perf_by_metric = {}
    for g, df in perf_g:
        df_pv = df.pivot(index='level_0', columns=['index'], values='value')
        df_pv = df_pv.reset_index().rename(columns={'level_0': 'Horizon'})

        perf_by_metric[g] = df_pv

    perf_n_df = pd.concat(perf_by_metric).reset_index(level=0)
    perf_n_df = perf_n_df.rename(columns={'level_0': 'Metric'})
    perf_n_df['Horizon'] = [f't+{i}' for i in perf_n_df['Horizon']]

    sens_result = sens_result.rename(columns={'level_1': 'Distribution',
                                              'LL': 'Log loss',
                                              'level_0': 'Horizon'})
    sens_result = sens_result.drop('BRIER', axis=1)
    sens_result['Distribution'] = sens_result['Distribution'].map({
        'cauchy': 'Cauchy',
        'gumbel_r': 'Gumbel',
        'laplace': 'Laplace',
        'invweibull': 'InvWeibull',
        'logistic': 'Logistic',
        'lognorm': 'Lognormal',
        'norm': 'Normal',
        'rayleigh': 'Rayleigh',
        'weibull_max': 'Weibull2',
        'weibull_min': 'Weibull'
    })

    sens_result['Horizon'] = [f't+{i}' for i in sens_result['Horizon']]

    return perf_clf, perf_n_df, sens_result
