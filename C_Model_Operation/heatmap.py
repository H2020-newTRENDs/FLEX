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

# # constant price and RTP
#
# # Data for plotting
# plt.figure(figsize=(20, 5))
# x = ['Reference', 'V2B', 'Smart App', 'Heat/Cool', 'Battery', 'Grid2Bat', 'All']
#
# ConstantPrice = [2073,2031,1995,1880,1699,1699,1554]
# Peak0 = [2061,2015,1976,1866,1682,1679,1530]
# Peak5 = [2062,1982,1970,1854,1673,1659,1494]
# Peak10= [2064,1927,1966,1838,1662,1630,1446]
# Peak15 = [2065,1871,1961,1819,1650,1601,1398]
#
# fig, ax = plt.subplots()
# ax.plot(x,ConstantPrice, marker='o', linestyle='--', label ='Constant', color = 'black', linewidth =0.001)
# ax.plot(x, Peak0, marker='+', linestyle='--', label = '+/- 0% Peak', color = 'darkslateblue', linewidth =0.001)
# ax.plot(x, Peak5, marker='+', linestyle='--', label = '+/- 5% Peak', color = 'royalblue', linewidth =0.001)
# ax.plot(x, Peak10, marker='+', linestyle='--', label = '+/- 10% Peak', color = 'cornflowerblue', linewidth =0.001)
# ax.plot(x, Peak15, marker='+', linestyle='--', label = '+/- 15% Peak', color = 'lightseagreen', linewidth =0.001)
# plt.xticks(rotation=90)
#
# ax.set(ylabel='Yearly cost (€)')
# ax.grid()
#
# plt.tight_layout();
# plt.legend(loc='upper right')
# plt.savefig('Constant_RTP_Pricing', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
# plt.show()



############################


### Savings of Optimization no EV
orders = np.array([[657,413,336,82,0],
                   [585,426,261,85,0],
                   [526,422,210,89,0]])


plt.figure(figsize=(8, 3))
xlabels = ['All', 'Battery', 'Heat/Cool', 'Smart App', 'Reference']
ylabels = ['AG1', 'AG2', 'AG3']

sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap='mako_r', \
            annot=False, fmt='g', linewidth=0.75, linecolor='white',
            cbar_kws={'label': 'Savings in € per year', 'orientation': 'horizontal'})

plt.tight_layout();

plt.savefig('Saving potential no EV', dpi=600, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
            facecolor='w', edgecolor='w')
plt.show()


### Savings of Optimization with EV

orders = np.array([[622,355,298,70,40,0],
                   [562,360,228,74,34,0],
                   [518,374,254,78,41,0]])


plt.figure(figsize=(8, 3))
xlabels = ['All', 'Battery', 'Heat/Cool', 'Smart App', 'V2B', 'Reference']
ylabels = ['AG1', 'AG2', 'AG3']

sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap='mako_r', \
            annot=False, fmt='g', linewidth=0.75, linecolor='white',
            cbar_kws={'label': 'Savings in € per year', 'orientation': 'horizontal'})

plt.tight_layout();

plt.savefig('Saving potential EV', dpi=600, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
            facecolor='w', edgecolor='w')
plt.show()

### Savings of Optimization with EV and RTP

orders = np.array([[841,632,475,409,328,258,99,0],
                   [761,570,471,410,285,227,102,0],
                   [667,511,464,415,245,194,104,0]])


plt.figure(figsize=(8, 3))
xlabels = ['All', '(All no RTP)', 'Grid2Bat' ,'Battery', 'Heat/Cool', 'V2B', 'Smart App', 'Reference']
ylabels = ['AG1', 'AG2', 'AG3']

sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap='mako_r', \
            annot=False, fmt='g', linewidth=0.75, linecolor='white',
            cbar_kws={'label': 'Savings in € per year', 'orientation': 'horizontal'})

plt.tight_layout();

plt.savefig('Saving potential EV with RTP', dpi=600, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
            facecolor='w', edgecolor='w')
plt.show()