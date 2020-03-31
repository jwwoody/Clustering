from sklearn.preprocessing import MinMaxScaler
from arff import Arff
from Kmeans import KMEANSClustering
import numpy as np
from HAC import HACClustering
mat = Arff("iris.arff",label_count=0) ## label_count = 0 because clustering is unsupervised.

raw_data = mat.data
data = raw_data[:,0:-1]
labels = raw_data[:,-1].reshape(-1,1)
labels = 1+labels

data = np.concatenate((data, labels), axis=1)
print(data)

### Normalize the data ###
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)
print(norm_data)

# ### KMEANS ###
kValues = [2,3,4,5,6,7]
SSEValues = []
for k in kValues:
    types = [1,1,1,1,1]
    KMEANS = KMEANSClustering(k=k,debug=False, columntype=types)
    while True:
        try:
            KMEANS.fit(norm_data)
            break
        except:
            KMEANS.fit(norm_data)

    KMEANS.save_clusters("iris_kmeans.txt")
    currentSSE = KMEANS.getTotalSSE()
    SSEValues.append(currentSSE)
    print(currentSSE)




### HAC SINGLE LINK ###
# HAC_single = HACClustering(k=5,link_type='single')
# HAC_single.fit(norm_data)
# HAC_single.save_clusters("debug_hac_single.txt")


### HAC COMPLETE LINK ###
# HAC_complete = HACClustering(k=5,link_type='complete')
# HAC_complete.fit(norm_data)
# HAC_complete.save_clusters("debug_hac_complete.txt")

import matplotlib.pyplot as plt

# An "interface" to matplotlib.axes.Axes.hist() method
value = np.array(SSEValues)
x_data= np.array(kValues)
plt.bar(x_data, value, tick_label=x_data)
plt.ylabel("Total SSE")
plt.xlabel("k-values")
plt.title("Total Clustering SSE versus K Values (Iris K-Means) With Labels")
# plt.axis(0,np.max(value)*1.25])

plt.show()
