import itertools
from typing import Dict

import pandas as pd


def expand_grid(*iters):
    product = list(itertools.product(*iters))
    return {i: [x[i] for x in product]
            for i in range(len(iters))}


def expand_grid_all(x: Dict) -> Dict:
    param_grid = expand_grid(*x.values())
    new_keys = dict(zip(param_grid.keys(), x.keys()))

    param_grid = {new_keys[k]: v for k, v in param_grid.items()}

    return param_grid

def expand_grid_from_dict(x: Dict) -> pd.DataFrame:
    param_grid = expand_grid(*x.values())
    param_grid = pd.DataFrame(param_grid)
    param_grid.columns = x.keys()

    return param_grid
