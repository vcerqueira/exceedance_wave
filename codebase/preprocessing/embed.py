import pandas as pd


def series_as_supervised(data: pd.DataFrame,
                         k: int,
                         horizon: int,
                         target_col: str):
    """
    time delay embedding for mv time series

    :param data: multivariate time series as pd.DF
    :param k: embedding dimension (applied to all cols)
    :param horizon: forecasting horizon
    :param target_col: string denoting the target column

    :return: trainable data set
    """

    iter_over_k = list(range(k, 0, -1))

    X_cols = []
    for col in data.columns:
        # input sequence (t-n, ... t-1)
        X, col_iter = [], []
        for i in iter_over_k:
            X.append(data[col].shift(i))

        X = pd.concat(X, axis=1)
        X.columns = [f'{col}-{j}' for j in iter_over_k]
        X_cols.append(X)

    X_cols = pd.concat(X_cols, axis=1)

    # forecast sequence (t, t+1, ... t+n)
    y = []
    for i in range(0, horizon):
        y.append(data[target_col].shift(-i))

    y = pd.concat(y, axis=1)
    y.columns = [f'{target_col}+{i}' for i in range(1, horizon + 1)]

    data_set = pd.concat([X_cols, y], axis=1).dropna()

    lag_cols = [f'{target_col}-{i}' for i in range(1, k + 1)]
    target_cols = [f'{target_col}+{i}' for i in range(1, horizon + 1)]

    return data_set, lag_cols, target_cols
