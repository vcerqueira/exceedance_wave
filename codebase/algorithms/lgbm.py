import lightgbm as lgbm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin

from codebase.evaluation.holdout import Holdout

PARAMETER_SET = \
    dict(num_leaves=[5, 10, 15, 30],
         max_depth=[-1, 3, 5, 10],
         lambda_l1=[0, 0.1, 1, 100],
         lambda_l2=[0, 0.1, 1, 100],
         learning_rate=[0.05, 0.1, 0.2],
         min_child_samples=[15, 30, 50, 100],
         n_jobs=[1],
         linear_tree=[True, False],
         boosting_type=['gbdt'])


class LightGBMOptim(BaseEstimator, RegressorMixin):

    def __init__(self, iters: int = 50, params=None):
        self.model = None
        self.iters = iters
        self.estimator = lgbm.LGBMRegressor(n_jobs=1)
        self.params = params
        self.parameters = \
            dict(num_leaves=[5, 10, 15, 30],
                 max_depth=[-1, 3, 5, 10],
                 lambda_l1=[0, 0.1, 1, 100],
                 lambda_l2=[0, 0.1, 1, 100],
                 learning_rate=[0.05, 0.1, 0.2],
                 min_child_samples=[15, 30, 50, 100],
                 n_jobs=[1],
                 linear_tree=[True, False],
                 boosting_type=['gbdt'],
                 num_boost_round=[25, 50, 100])

    def fit(self, X, y=None):
        if self.params is None:
            self.model = RandomizedSearchCV(estimator=self.estimator,
                                            param_distributions=self.parameters,
                                            scoring='neg_mean_squared_error',
                                            n_iter=self.iters,
                                            n_jobs=1,
                                            refit=True,
                                            verbose=-1,
                                            cv=Holdout(n=X.shape[0]),
                                            random_state=123)

            self.model.fit(X, y)
        else:
            self.model = lgbm.LGBMRegressor(**self.params)
            self.model.fit(X, y)

    def predict(self, X):
        y_hat = self.model.predict(X)

        return y_hat
