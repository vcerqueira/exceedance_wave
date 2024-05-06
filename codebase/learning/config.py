from sklearn.ensemble \
    import (RandomForestRegressor,
            ExtraTreesRegressor,
            AdaBoostRegressor,
            BaggingRegressor)
from sklearn.linear_model \
    import (Lasso,
            Ridge,
            OrthogonalMatchingPursuit,
            ElasticNet)
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression, PLSCanonical

from codebase.algorithms.lgbm import LightGBMOptim
from codebase.algorithms.nn import DeepNet

from codebase.utils import expand_grid_from_dict

METHODS = \
    dict(
        DeepNet=DeepNet,
        LightGBMRegressor=LightGBMOptim,
        RandomForestRegressor=RandomForestRegressor,
        PLSRegression=PLSRegression,
        PLSCanonical=PLSCanonical,
        ExtraTreesRegressor=ExtraTreesRegressor,
        OrthogonalMatchingPursuit=OrthogonalMatchingPursuit,
        AdaBoostRegressor=AdaBoostRegressor,
        Lasso=Lasso,
        Ridge=Ridge,
        ElasticNet=ElasticNet,
        BaggingRegressor=BaggingRegressor,
    )

METHODS_PARAMETERS = \
    dict(
        RandomForestRegressor={
            'n_estimators': [50, 100],
            'max_depth': [None, 3],
        },
        DeepNet={
            'batch_size': [512],
            'n_epochs': [30],
            'in_hre': [True],
        },
        ExtraTreesRegressor={
            'n_estimators': [50, 100],
            'max_depth': [None, 3],
        },
        OrthogonalMatchingPursuit={},
        AdaBoostRegressor={
            'base_estimator': [DecisionTreeRegressor(max_depth=3),
                               DecisionTreeRegressor(max_depth=1)],
            'loss': ['linear'],
            'learning_rate': [0.3, 1],
        },
        Lasso={
            'alpha': [1, .5, .25, .75]
        },
        Ridge={
            'alpha': [1, .5, .25, .75]
        },
        ElasticNet={
        },
        PLSRegression={
            'n_components': [2, 5]
        },
        PLSCanonical={
            'n_components': [2, 5]
        },
        BaggingRegressor={
            'base_estimator': [DecisionTreeRegressor(max_depth=3),
                               # DecisionTreeRegressor(max_depth=5),
                               DecisionTreeRegressor(max_depth=1)
                               ],
            'n_estimators': [25, 50]
        },
        LightGBMRegressor={},
    )

MODELS_ON_SUBSET = ['GaussianProcessRegressor', 'SVR', 'LinearSVR', 'NuSVR', 'CubistR', 'ProjectionPursuitRegressor']
