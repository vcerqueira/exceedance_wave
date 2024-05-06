import numpy as np
import pandas as pd
from distfit import distfit

from scipy.stats import (norm,
                         laplace,
                         logistic,
                         gumbel_r,
                         lognorm,
                         cauchy,
                         rayleigh,
                         invweibull,
                         weibull_max,
                         weibull_min)

DISTRIBUTIONS = {
    'norm': norm,
    'laplace': laplace,
    'logistic': logistic,
    'gumbel_r': gumbel_r,
    'lognorm': lognorm,
    'cauchy': cauchy,
    'rayleigh': rayleigh,
    'invweibull': invweibull,
    'weibull_max': weibull_max,
    'weibull_min': weibull_min,
}


class CDFEngine:

    def __init__(self):
        self.distributions = [*DISTRIBUTIONS]
        self.scores = None
        self.params = {}

    def fit(self, series: pd.Series):
        # dist_list = ['cauchy', 'gumbel_r', 'laplace', 'lognorm', 'norm', 'logistic', 'rayleigh']

        dfit = distfit(method='parametric', todf=True, distr=self.distributions)
        dfit.fit_transform(series)
        self.scores = dfit.summary[['name', 'score', 'loc', 'scale']]

        self.params = {k: DISTRIBUTIONS[k].fit(series) for k in DISTRIBUTIONS}

        return self.scores

    def cdf_by_dist(self, distribution: str, location: float):
        DIST = {
            'norm': dict(loc=location, scale=self.params['norm'][1]),
            'laplace': dict(loc=location, scale=self.params['laplace'][1]),
            'logistic': dict(loc=location, scale=self.params['logistic'][1]),
            'gumbel_r': dict(loc=location, scale=self.params['gumbel_r'][1]),
            'lognorm': dict(loc=location,
                            s=self.params['lognorm'][0],
                            scale=self.params['lognorm'][2]),
            'cauchy': dict(loc=location, scale=self.params['cauchy'][1]),
            'rayleigh': dict(loc=location, scale=self.params['rayleigh'][1]),
            'invweibull': dict(loc=location,
                               c=self.params['invweibull'][0],
                               scale=self.params['invweibull'][2]),
            'weibull_max': dict(loc=location,
                                c=self.params['weibull_max'][0],
                                scale=self.params['weibull_max'][2]),
            'weibull_min': dict(loc=location,
                                c=self.params['weibull_min'][0],
                                scale=self.params['weibull_min'][2]),
        }

        return DIST[distribution]

    def point_exceedance_by_dist(self, distribution: str, location: float, threshold: float):
        PROBS = {
            'norm': 1 - norm.cdf(threshold, **self.cdf_by_dist('norm', location)),
            'laplace': 1 - laplace.cdf(threshold, **self.cdf_by_dist('laplace', location)),
            'logistic': 1 - logistic.cdf(threshold, **self.cdf_by_dist('logistic', location)),
            'gumbel_r': 1 - gumbel_r.cdf(threshold, **self.cdf_by_dist('gumbel_r', location)),
            'lognorm': 1 - lognorm.cdf(threshold, **self.cdf_by_dist('lognorm', location)),
            'cauchy': 1 - cauchy.cdf(threshold, **self.cdf_by_dist('cauchy', location)),
            'rayleigh': 1 - rayleigh.cdf(threshold, **self.cdf_by_dist('rayleigh', location)),
            'invweibull': 1 - invweibull.cdf(threshold, **self.cdf_by_dist('invweibull', location)),
            'weibull_max': 1 - weibull_max.cdf(threshold, **self.cdf_by_dist('weibull_max', location)),
            'weibull_min': 1 - weibull_min.cdf(threshold, **self.cdf_by_dist('weibull_min', location)),
        }

        return PROBS[distribution]

    def get_probs(self, distribution: str, y_hat: np.ndarray, threshold: float):
        p_exc = [self.point_exceedance_by_dist(distribution=distribution,
                                               location=x_,
                                               threshold=threshold)
                 for x_ in y_hat]

        p_exc = np.asarray(p_exc)

        return p_exc

    def get_all_probs(self, y_hat: np.ndarray, threshold: float):
        dists = [*DISTRIBUTIONS]

        dist_pe = {}
        for d_ in dists:
            p_exc = [self.point_exceedance_by_dist(distribution=d_,
                                                   location=x_,
                                                   threshold=threshold)
                     for x_ in y_hat]

            p_exc = np.asarray(p_exc)
            dist_pe[d_] = p_exc

        dist_pe = pd.DataFrame(dist_pe)

        return dist_pe
