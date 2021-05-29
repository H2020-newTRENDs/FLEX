import numpy as np
import matplotlib.pyplot as plt

data1=[30,20,10,0,0]
data2=[20,20,20,20,0]
data3=[50,60,70,80,100]

year=["2015","2016","2017","2018","2019"]

plt.figure(figsize=(9,7))
plt.bar(year,data3,color="green",label="Python")
plt.bar(year,data2,color="yellow",bottom=np.array(data3),label="JavaScript")
plt.bar(year,data1,color="red",bottom=np.array(data3)+np.array(data2),label="C++")

plt.legend(loc="lower left",bbox_to_anchor=(0.8,1.0))
plt.show()