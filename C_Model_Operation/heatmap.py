import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# # PV and Battery with Reference to no PV
# orders = np.array([[2018, 1648, 832, 0],
#                    [1999, 1629, 832, 0],
#                    [1887, 1537, 822, 0],
#                    [1764, 1432, 791, 0],
#                    [2005, 1656, 832, 0],
#                    [1991, 1640, 832, 0],
#                    [1878, 1544, 824, 0],
#                    [1749, 1431, 794, 0],
#                    [1884, 1583, 830, 0],
#                    [1881, 1572, 829, 0],
#                    [1772, 1471, 819, 0],
#                    [1637, 1345, 782, 0]])
#
# plt.figure(figsize=(8, 5))
# xlabels = ['10 kWp', '7 kWp', '3 kWp', '0 kWp']
# ylabels = ['AG1 10 kWh', ' 7 kWh', ' 3 kWh', '0 kWh', \
#            'AG2 10 kWh', ' 7 kWh', ' 3 kWh', '0 kWh', \
#            'AG3 10 kWh', ' 7 kWh', ' 3 kWh', '0 kWh']
#
# sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap='mako_r', \
#             annot=False, fmt='g', linewidth=0.75, linecolor='white',
#             cbar_kws={'label': 'Savings € / year', 'orientation': 'vertical'})
#
# plt.tight_layout();
#
# plt.savefig('BatteryAndPVHeatmap_RefPV', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
# plt.show()
#
# # PV and Battery with Reference to no Battery
#
# orders = np.array([[253, 234, 123, 0],
#                    [216, 197, 105, 0],
#                    [40, 40, 30, 0],
#                    [0, 0, 0, 0],
#                    [256, 242, 128, 0],
#                    [224, 208, 112, 0],
#                    [38, 38, 30, 0],
#                    [0, 0, 0, 0],
#                    [247, 244, 135, 0],
#                    [237, 226, 125, 0],
#                    [47, 47, 36, 0],
#                    [0, 0, 0, 0]])
#
# plt.figure(figsize=(8, 5))
# xlabels = ['10 kWh', '7 kWh', '3 kWh', '0 kWh']
# ylabels = ['AG1 10 kWp', ' 7 kWp', ' 3 kWp', '0 kWp', \
#            'AG2 10 kWp', ' 7 kWp', ' 3 kWp', '0 kWp', \
#            'AG3 10 kWp', ' 7 kWp', ' 3 kWp', '0 kWp']
#
# sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap='mako_r', \
#             annot=False, fmt='g', linewidth=0.75, linecolor='white',
#             cbar_kws={'label': 'Savings € / year', 'orientation': 'vertical'})
#
# plt.tight_layout();
#
# plt.savefig('BatteryAndPVHeatmap_RefBattery', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
# plt.show()
#
# # Integration of EV
#
# orders = np.array([[289, 291, 552,552, 0],
#                    [445, 445, 652,653, 0],
#                    [781, 781, 808,808, 0],
#                    [824, 824, 824,824, 0],
#                    [262, 263, 508,537, 0],
#                    [437, 437, 649,653, 0],
#                    [784, 784, 807,807, 0],
#                    [824, 824, 824,824, 0],
#                    [244, 244, 475,509, 0],
#                    [378, 378, 623,627, 0],
#                    [774, 774, 805,805, 0],
#                    [824, 824, 824,824, 0]])
#
# plt.figure(figsize=(8, 5))
# xlabels = ['EV(V2B) + SBS',  'EV(no V2B) + SBS', 'EV(V2B)', 'EV(no V2B)', 'no EV, no SBS']
# ylabels = ['AG1 10 kWp', '7 kWp', '3kWp', '0 kWp', \
#            'AG2 10 kWp', '7 kWp', '3kWp', '0 kWp', \
#            'AG3 10 kWp', '7 kWp', '3kWp', '0 kWp']
#
# sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap='mako_r', \
#             annot=False, fmt='g', linewidth=0.75, linecolor='white',
#             cbar_kws={'label': 'Additional cost of the EV in € / year ', 'orientation': 'vertical'})
#
# plt.tight_layout();
#
# plt.savefig('Integration of EV', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
# plt.show()

# constant price and RTP

# Data for plotting
plt.figure(figsize=(20, 5))
x = ['Reference', 'Tank = 1500l', 'V2B =1', 'Smart App. = 1', 'Battery', 'Grid2Bat =1', 'All']

ConstantPrice = [1909,1880,1872,1845,1603,1603,1554]
Peak0 = [1893,1866,1852,1821,1585,1582,1529]
Peak5 = [1884,1854,1816,1806,1570,1558,1494]
Peak10= [1873,1838,1761,1789,1551,1525,1446]
Peak15 = [1857,1819,1703,1767,1527,1489,1397]

fig, ax = plt.subplots()
ax.plot(x,ConstantPrice, marker='o', linestyle='--', label ='Constant', color = 'dimgray', linewidth =0.5)
ax.plot(x, Peak0, marker='o', linestyle='--', label = '+/- 0% Peak', color = 'cadetblue', linewidth =0.5)
ax.plot(x, Peak5, marker='o', linestyle='--', label = '+/- 5% Peak', color = 'royalblue', linewidth =0.5)
ax.plot(x, Peak10, marker='o', linestyle='--', label = '+/- 10% Peak', color = 'steelblue', linewidth =0.5)
ax.plot(x, Peak15, marker='o', linestyle='--', label = '+/- 15% Peak', color = 'midnightblue', linewidth =0.5)
plt.xticks(rotation=90)

ax.set(ylabel='Yearly cost (€)')
ax.grid()

plt.tight_layout();
plt.legend(loc='upper right')
plt.savefig('Constant_RTP_Pricing', dpi=600, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
            facecolor='w', edgecolor='w')
plt.show()
