import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import math
import random

import warnings
warnings.filterwarnings("error")

class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=2,tol = 0.001, debug=False, max_iter=300, columntype = []): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug
        self.tol = tol
        self.max_iter=max_iter
        self.columntype = columntype

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        data = X
        self.centroids = {}
        randList = random.sample(range(len(data)), self.k)
        for i in range(self.k):
            if self.debug:
                self.centroids[i] = data[i] #initializeCentroids
            else:
                self.centroids[i] = data[randList[i]]


        for i in range(self.max_iter):
            self.classifications = {} #Centroid with class
            for i in range(self.k):
                self.classifications[i] = []
            for featureSet in data:
        #         calculate distances
                distances = [np.linalg.norm(featureSet-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureSet)
            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                # pass
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                try:
                    if original_centroid[-1] == 0:
                        if current_centroid[-1] == 0:
                            original_centroid[-1] = 1
                        else:
                            original_centroid[-1] = current_centroid[-1]
                    if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
                        optimized=False
                except RuntimeWarning as r:
                    print(r)
                    print(original_centroid)
                    raise
            if optimized:
                break
        return self


    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


    def compute_sse(self,row1,row2):
        distance = 0.0
        for col in range(0, len(row1)):
            i1 = row1[col]
            i2 = row2[col]
            if math.isnan(i1) or math.isnan(i2):
                distance += 1
            elif self.columntype[col] == 1:
                distance += (i1 - i2) ** 2
            else:
                if i1 != i2:
                    distance += 1
        return distance

    def save_clusters(self,filename):
        """
            f = open(filename,"w+")
            # Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """

        totalSSE=0

        f = open(filename, "w+")
        # Used for grading.
        f.write("{:d}\n".format(self.k))
        for centroid, value in self.centroids.items():
            totalSSE += np.sum([self.compute_sse(i, value) for i in self.classifications[centroid]])

        self.totalSSE = totalSSE
        f.write("{:.4f}\n\n".format(totalSSE))
        # print(type(self.centroids))
        # print(self.centroids)
        # print(self.classifications)
        for centroid, value in self.centroids.items():
            f.write(np.array2string(value, precision=4, separator=","))
            f.write("\n")
            cluster_size=len(self.classifications[centroid])
            f.write("{:d}\n".format(cluster_size))
            cluster_sse = np.sum([self.compute_sse(i, value) for i in self.classifications[centroid]])
            f.write("{:.4f}\n\n".format(cluster_sse))
        f.close()

    def getTotalSSE(self):
        return self.totalSSE
