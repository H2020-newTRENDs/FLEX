import seaborn as sns

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# ## 6.1 PV
# ##PV and Battery with Reference to no PV
#
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
#             cbar_kws={'label': 'Savings in € / year', 'orientation': 'vertical'})
#
# plt.tight_layout();
#
# plt.savefig('Result_Various PV', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
# plt.show()
#
#
#
# ## 6.1 Battery
# ## PV and Battery with Reference to no Battery
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
#             cbar_kws={'label': 'Savings in € / year', 'orientation': 'vertical'})
#
# plt.tight_layout();
#
# plt.savefig('Result_Various SBS', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
# plt.show()
#
#
# ## 6.2 Integration of EV - Saving potential
#
# orders = np.array([[935,413,354,314,336,82],
#                    [887,426,359,325,261,85],
#                    [857,423,380,338,211,89]])
#
#
# plt.figure(figsize=(8, 3))
# xlabels = ['All','Battery', 'EV opt.(V2B)', 'EV opt.', 'Heat/Cool', 'SmartApp']
# ylabels = ['AG1', 'AG2', 'AG3']
#
# sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap='mako_r', \
#             annot=False, fmt='g', linewidth=0.75, linecolor='white',
#             cbar_kws={'label': 'Savings in € per year', 'orientation': 'horizontal'})
#
# plt.tight_layout();
#
# plt.savefig('Result_Saving potential with EV', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
# plt.show()
#
#
# ## 6.3 RTP 0%, 5%, 10% and 15%
#
# plt.figure(figsize=(20, 5))
# x = ['SmartApp','Heat/Cool','EV opt.', 'EV opt.+V2B', 'Battery', 'Grid2Bat', 'All']
#
# ConstantPrice = [89,210,338,379,423,423,856]
# Peak0 = [96,212,363,409,427,430,894]
# Peak5 = [102,225,379,459,447,450,947]
# Peak10= [107,241,395,532,445,477,1012]
# Peak15 = [112,260,411,605,455,504,1078]
#
# fig, ax = plt.subplots()
# ax.plot(x,ConstantPrice, marker='o', linestyle='--', label ='Static', color = 'black', linewidth =0.001)
# ax.plot(x, Peak0, marker='+', linestyle='--', label = '+/- 0% Peak', color = 'darkslateblue', linewidth =0.001)
# ax.plot(x, Peak5, marker='+', linestyle='--', label = '+/- 5% Peak', color = 'royalblue', linewidth =0.001)
# ax.plot(x, Peak10, marker='+', linestyle='--', label = '+/- 10% Peak', color = 'cornflowerblue', linewidth =0.001)
# ax.plot(x, Peak15, marker='+', linestyle='--', label = '+/- 15% Peak', color = 'lightseagreen', linewidth =0.001)
# plt.xticks(rotation=90)
#
# ax.set(ylabel='Saving potential for each components in € per year')
# ax.grid()
#
# plt.tight_layout();
# plt.legend(loc='upper left')
# plt.savefig('Result_Various dynamic prices', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
# plt.show()
#
# ## 6.3 RTP Saving potential
#
# orders = np.array([[1234,935,652,522,457,393,358,107],
#                    [1162,887,628,525,466,401,313,109],
#                    [1078,857,605,505,456,411,261,113]])
#
#
# plt.figure(figsize=(8, 3))
# xlabels = ['All', '(All sta.)', 'EV opt.(V2B)' ,'Grid2Bat', 'Battery', 'EV opt.', 'Heat/Cool','SmartApp']
# ylabels = ['AG1', 'AG2', 'AG3']
#
# sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap='mako_r', \
#             annot=False, fmt='g', linewidth=0.75, linecolor='white',
#             cbar_kws={'label': 'Savings in € per year', 'orientation': 'horizontal'})
#
# plt.tight_layout();
#
# plt.savefig('Result_Saving potential EV with RTP', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
# plt.show()
#

## Saving potential static and dynamic


orders = np.array([[935, 354, 413, 413, 314, 336, 82],
                   [1234, 652, 522, 457, 393, 358, 107],
                   [887, 359, 426, 426, 325, 261, 85],
                   [1162, 628, 525, 466, 401, 313, 109],
                   [857, 380, 423, 423, 338, 211, 89],
                   [1078, 605, 505, 456, 411, 261, 113]])

plt.figure(figsize=(8, 3))
xlabels = ['All', 'EV opt.(V2B)', 'Grid2Bat', 'Battery', 'EV opt.', 'Heat/Cool', 'SmartApp']
ylabels = ['AG1 static', 'dynamic', 'AG2 static', 'dynamic', 'AG3 static', 'dynamic']

sns.heatmap(orders, xticklabels=xlabels, yticklabels=ylabels, cmap='mako_r', \
            annot=False, fmt='g', linewidth=0.75, linecolor='white',
            cbar_kws={'label': 'Savings in € per year', 'orientation': 'horizontal'})

plt.tight_layout();

plt.savefig('Result_Saving potential_all', dpi=600, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
            facecolor='w', edgecolor='w')
plt.show()
