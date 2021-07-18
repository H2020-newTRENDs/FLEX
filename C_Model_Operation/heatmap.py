import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np




orders = np.array([[2000, 2000, 1887, 1764],
                   [1648, 1629, 1538,1432],
                   [832, 832, 822,791],
                   [2005, 1991, 1878,1749],
                   [1656, 1640, 1544,1431],
                   [832, 832, 824,794],
                   [1884, 1881, 1772,1637],
                   [1583, 1572, 1471,1345],
                   [830, 829, 819,782]])


plt.figure(figsize=(8, 5))
xlabels = ['10 kWh', '7 kWh', '3 kWh', '0 kWh']
ylabels = ['AG1 10 kWp', ' 7 kWp', ' 3 kWp', \
           'AG2 10 kWp', '7 kWp', '3 kWp', \
           'AG3 10 kWp', ' 7 kWp,', '3 kWp']

sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap = 'mako_r',\
            annot = False, fmt='g', linewidth = 0.75, linecolor = 'white', cbar_kws={'label': 'Savings compared to PV power = 0 kWp in €', 'orientation': 'horizontal'})



plt.tight_layout();

plt.savefig('BatteryAndPVHeatmap', dpi=600,  format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
            facecolor='w', edgecolor='w')
plt.show()