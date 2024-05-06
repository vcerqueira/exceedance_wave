import pandas as pd
import numpy as np
from plotnine import *

from codebase.cdf import CDFEngine
from config import DATA_DIR

wave = pd.read_csv(DATA_DIR, parse_dates=['DATE'], index_col='DATE')

# Exceedance prob. curve

series = wave['VCAR'].dropna()

cdf = CDFEngine()
cdf.fit(series)

pred = 2.55
thr = np.linspace(start=0.1, stop=9, num=100)

probs = [cdf.point_exceedance_by_dist(distribution='gumbel_r',
                                      location=pred,
                                      threshold=x)
         for x in thr]

df = pd.DataFrame(
    {
        'x': thr,
        'y': probs
    }
)

plt = ggplot(data=df,
             mapping=aes(x='x', y='y')) + \
      theme_classic(base_family='Palatino', base_size=12) + \
      geom_line(size=1.1) + \
      labs(x='', y='Exceedance prob.') + \
      geom_vline(xintercept=pred, color='red', linetype='dashed', size=1.5) + \
      geom_vline(xintercept=3.1, color='blue', size=1.5)

plt.save(f'assets/plots/epc.pdf', height=7, width=15)
