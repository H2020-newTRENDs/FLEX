import matplotlib.pyplot as plt
import numpy as np


#
# ## Baseload profile
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# Winter_Workday = [71, 51, 38, 36, 41, 59, 93, 122, 103, 92, 93, 108, 104, 104, 103, 97, 105, 143, 168, 176, 156, 133,
#                   127, 95]
# Winter_Saturday = [75, 55, 40, 36, 39, 49, 64, 92, 99, 107, 115, 131, 126, 122, 129, 136, 150, 187, 195, 181, 153, 130,
#                    133, 106]
# Winter_Sunday = [85, 65, 48, 43, 45, 53, 63, 93, 116, 137, 153, 177, 167, 147, 135, 130, 140, 177, 187, 177, 153, 129,
#                  127, 98]
#
# Trans_Workday = [69,52,44,43,47,65,102,126,111,105,109,124,124,122,117,107,107,127,155,180,173,152,141,98]
#
# Trans_Saturday = [76,55,44,42,45,55,74,95,102,120,132,149,146,140,144,148,153,171,187,192,172,149,149,115]
#
# Trans_Sunday = [86,65,53,50,53,61,74,101,128,154,173,196,186,165,149,140,141,159,173,180,166,146,140,101]
#
# Summer_Workday = [84,70,64,63,64,80,108,126,119,118,126,142,146,142,137,127,128,139,159,172,172,168,162,118]
#
# Summer_Saturday = [92,72,65,62,63,72,86,98,108,130,146,161,163,157,162,168,170,180,184,178,169,163,165,132]
#
# Summer_Sunday = [100,82,73,70,71,78,87,108,139,169,189,210,206,185,171,165,164,175,183,178,172,166,159,119]
#
#
# fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.set_size_inches(10,6)
#
# ax1.plot(x, Winter_Workday, label= 'Workday', color = 'darkred', linewidth = 0.5)
# ax1.plot(x, Winter_Saturday, label ='Saturday', color = 'darkgreen', linewidth = 0.5)
# ax1.plot(x, Winter_Sunday, label = 'Sunday', color = 'darkblue', linewidth = 0.5)
# ax1.legend(loc='upper right')
# ax1.set_title('Winter', fontsize = 10)
# ax1.set_xlabel('HoD (h)', fontsize = 8)
# ax1.set_ylabel('Power (W)', fontsize = 8)
#
# yticks = np.arange(0,250, 100)
# xticks = np.arange(0,24,4)
# ax1.set_yticks(yticks)
# ax1.set_xticks(xticks)
# ax1.grid()
#
#
# ax2.plot(x, Trans_Workday, color = 'darkred', linewidth = 0.5)
# ax2.plot(x, Trans_Saturday, color = 'darkgreen', linewidth = 0.5)
# ax2.plot(x, Trans_Sunday, color = 'darkblue', linewidth = 0.5)
# ax2.set_title('Transmission', fontsize = 10)
# ax2.set_xlabel('HoD (h)', fontsize = 8)
# ax2.set_ylabel('Power (W)', fontsize = 8)
# ax2.grid()
# ax2.set_yticks(yticks)
# ax2.set_xticks(xticks)
#
# ax3.plot(x, Summer_Workday, color = 'darkred', linewidth = 0.5)
# ax3.plot(x, Summer_Saturday, color = 'darkgreen', linewidth = 0.5)
# ax3.plot(x, Summer_Sunday, color ='darkblue', linewidth = 0.5)
# ax3.set_title('Summer', fontsize = 10)
# ax3.set_xlabel('HoD (h)', fontsize = 8)
# ax3.set_ylabel('Power (W)', fontsize = 8)
# ax3.set_yticks(yticks)
# ax3.set_xticks(xticks)
# ax3.grid()
#
#
#
# #plt.tight_layout();
# plt.subplots_adjust(hspace= 0.8)
#
#
# plt.savefig('Baseload profile', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
#
# plt.show()



# ### DHW
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# Winter_Workday = [40,16,3,7,23,89,192,251,242,185,152,142,139,139,126,112,99,146,175,208,172,172,132,102,]
# Winter_Saturday = [69,36,20,10,13,23,40,106,218,308,311,268,255,275,262,265,295,351,371,314,218,132,86,79]
# Winter_Sunday = [66,53,36,17,13,10,30,106,222,354,384,374,318,275,199,132,106,132,179,185,175,169,139,103]
#
#
#
# Trans_Workday =[38,15,6,7,22,76,159,208,207,160,132,119,121,122,112,99,91,117,149,172,157,151,121,89]
# Trans_Saturday = [58,27,16,6,9,22,49,109,195,255,250,217,204,220,205,204,222,270,301,273,207,139,93,76]
# Trans_Sunday = [55,38,23,8,7,7,31,91,189,290,329,324,283,235,174,118,98,113,146,156,146,141,113,83]
#
#
#
# Summer_Workday = [36,13,8,8,22,63,127,166,172,136,113,96,103,106,99,86,83,89,122,136,142,129,109,76]
# Summer_Saturday = [47,18,12,2,5,22,58,113,172,202,189,166,152,166,149,142,149,189,232,232,195,146,99,73]
# Summer_Sunday = [43,23,10,0,2,5,31,76,156,225,275,275,248,195,149,103,89,93,113,126,116,113,86,63,]
#
#
# fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.set_size_inches(10,6)
# locs, labels = plt.xticks()
#
#
# ax1.plot(x, Winter_Workday, label= 'Workday', color = 'darkred', linewidth = 0.5)
# ax1.plot(x, Winter_Saturday, label ='Saturday', color = 'darkgreen', linewidth = 0.5)
# ax1.plot(x, Winter_Sunday, label = 'Sunday', color = 'darkblue', linewidth = 0.5)
# ax1.legend(loc='upper right')
# ax1.set_title('Winter', fontsize = 10)
# ax1.set_xlabel('HoD (h)', fontsize = 8)
# ax1.set_ylabel('Power (W)', fontsize = 8)
#
# yticks = np.arange(0,450, 200)
# xticks = np.arange(0,24,4)
# ax1.set_yticks(yticks)
# ax1.set_xticks(xticks)
#
# ax1.grid()
#
#
# ax2.plot(x, Trans_Workday, color = 'darkred', linewidth = 0.5)
# ax2.plot(x, Trans_Saturday, color = 'darkgreen', linewidth = 0.5)
# ax2.plot(x, Trans_Sunday, color = 'darkblue', linewidth = 0.5)
# ax2.set_title('Transmission', fontsize = 10)
# ax2.set_xlabel('HoD (h)', fontsize = 8)
# ax2.set_ylabel('Power (W)', fontsize = 8)
# ax2.grid()
#
# ax2.set_yticks(yticks)
# ax2.set_xticks(xticks)
#
#
#
#
# ax3.plot(x, Summer_Workday, color = 'darkred', linewidth = 0.5)
# ax3.plot(x, Summer_Saturday, color = 'darkgreen', linewidth = 0.5)
# ax3.plot(x, Summer_Sunday, color ='darkblue', linewidth = 0.5)
# ax3.set_title('Summer', fontsize = 10)
# ax3.set_xlabel('HoD (h)', fontsize = 8)
# ax3.set_ylabel('Power (W)', fontsize = 8)
# ax3.grid()
#
# ax3.set_yticks(yticks)
# ax3.set_xticks(xticks)
#
#
#
#
# #plt.tight_layout();
# plt.subplots_adjust(hspace= 0.8)
#
#
# plt.savefig('DHW', dpi=600, format=None,
#             transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
#             facecolor='w', edgecolor='w')
#
# plt.show()



# Smart Appliances

plt.rcParams["figure.figsize"] = (10,3)
Dishwasher = [1, 0, 1, 1, 0,1,1]
Washingmachine = [1,0,1,0,1,0,1]
bars = ('Monday', 'Thuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
y_pos = np.arange(len(bars))

plt.bar(y_pos, Dishwasher, label = 'Dishwasher', color ='gainsboro', hatch ='/', alpha = 0.5)
plt.bar(y_pos, Washingmachine, label = 'Washing machine + dryer ', color ='gainsboro', hatch ='\\', alpha = 0.5)

plt.yticks([0,1], ['0', '1'])
plt.xticks(y_pos, bars)
plt.legend(loc='upper right')

plt.ylabel('Period for optimization')

plt.show()

plt.savefig('SmartApp', dpi=600, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None,
            facecolor='w', edgecolor='w')