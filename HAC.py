import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
import math
import copy

class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.clusters = []
    def findMinimumValue(self, d):
        minV = float('inf')
        mini = 0
        minj = 0
        for i in range(len(d)):
            for j in range(len(d)):
                if i != j and d[i][j] < minV and j < i:
                    minV =d[i][j]
                    mini = i
                    minj = j
        return min(mini, minj), max(mini, minj), minV


    def minimumDistance(self, cluster1, cluster2, distanceMatrix):
        minimumD = float('inf')
        for i in cluster1:
            for j in cluster2:
                if distanceMatrix[i][j] < minimumD:
                    minimumD = distanceMatrix[i][j]
        return minimumD

    def maxDistance(self, cluster1, cluster2, distanceMatrix):
        # return max(self.minimumDistance(cluster1, cluster2, distanceMatrix), )




        maxD = -1*float('inf')
        for i in cluster1:
            for j in cluster2:
                if distanceMatrix[i][j]  > maxD and distanceMatrix[i][j] != float('inf'):
                    maxD = distanceMatrix[i][j]
        if maxD == -1*float('inf'):
            return float('inf')
        else:
            return maxD

    def updatedM(self, dM, m1, m2, distanceMatrix, clusterArray):
        for i in range(len(dM)):
            for j in range(len(dM)):

                if i >j:
                    if i == m1:
                        if self.link_type == 'single':
                            dM[i][j] = self.minimumDistance(clusterArray[m1], clusterArray[j], distanceMatrix)
                        else:
                            dM[i][j] = self.maxDistance(clusterArray[m1], clusterArray[j], distanceMatrix)
                    elif j == m1:
                        if self.link_type == 'single':
                            dM[i][j] = self.minimumDistance(clusterArray[i], clusterArray[j], distanceMatrix)
                        else:
                            dM[i][j] = self.maxDistance(clusterArray[i], clusterArray[j], distanceMatrix)
                    if i == m2:
                        dM[i][j] = float('inf')
                    if j == m2:
                        dM[i][j] = float('inf')
        return dM
    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        # X = np.array([[0.4, 0.53],
        #      [0.22, 0.38],
        #      [0.35, 0.32],
        #      [0.26, 0.19],
        #      [0.08, 0.41],
        #
        self.data = X
        distanceMatrix = np.linalg.norm(X - X[:,None], axis=-1)

        clusters = [[i] for i in range(len(X))]

        currentClusters = len(clusters)
        # dM = self.computeDistanceMatrix(clusters)
        dM = copy.deepcopy(distanceMatrix)
        while currentClusters > self.k:
            min1, min2, minV =  self.findMinimumValue(dM)

            clusters[min1] = clusters[min1] + clusters[min2]
            clusters[min2] = []
            # print('merging clusters ', min1, ' and ', min2, '      distance ', minV)
            dM = self.updatedM(dM, min1, min2, distanceMatrix, clusters)
            # dM = np.delete(dM, min2, axis=1)
            # dM = np.delete(dM, min2, axis=0)

            currentClusters -=1


        self.clusters = clusters
        return self

    def compute_sse(self,row1,row2):
        distance = 0.0
        row1 = row1.tolist()
        row2 = row2.tolist()
        for col in range(0, len(row1)):
            i1 = row1[col]
            i2 = row2[col]
            if math.isnan(i1) or math.isnan(i2):
                distance += 1
            # elif self.columntype[col] == 1:
            distance += (i1 - i2) ** 2
            # else:
            #     if i1 != i2:
            #         distance += 1
        return distance

    def save_clusters(self,filename):
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """
        totalSSE = 0

        f = open(filename, "w+")
        # Used for grading.
        f.write("{:d}\n".format(self.k))
        finalClusters  = []
        centroids = []
        clusterSize = []
        clusterSSE = []
        print(self.data[0])
        for i in range(len(self.clusters)):
            if self.clusters[i] !=[]:
                finalClusters.append(self.clusters[i])
                currentCluster = self.clusters[i]
                clusterSize.append(len(currentCluster))
                total = np.zeros(len(self.data[currentCluster[0]]))
                for j in range(0, len(currentCluster)):
                    if np.all(total == 0):
                        total = np.array(self.data[currentCluster[j]])
                    else:
                        total = np.vstack((total, self.data[currentCluster[j]]))
                cluster_sse = 0
                if len(currentCluster) == 1:
                    centroids.append(total)
                else:
                    centroidValue = np.average(total, axis = 0)
                    centroids.append(centroidValue)
                    cluster_sse = np.sum([self.compute_sse(i, centroidValue) for i in total])


                clusterSSE.append(cluster_sse)
                totalSSE += cluster_sse



        # for value in finalClusters:
        #
        #
        #
        #     totalSSE += np.sum([self.compute_sse(i, value) for i in self.classifications[centroid]])

        f.write("{:.4f}\n\n".format(totalSSE))

        for j in range(0, len(finalClusters)):
            f.write(np.array2string(centroids[j], precision=4, separator=","))
            f.write("\n")
            cluster_size = clusterSize[j]
            f.write("{:d}\n".format(cluster_size))
            cluster_sse = clusterSSE[j]
            f.write("{:.4f}\n\n".format(cluster_sse))
        f.close()
        #
        # for centroid, value in self.centroids.items():
        #     totalSSE += np.sum([self.compute_sse(i, value) for i in self.classifications[centroid]])
        #
        # f.write("{:.4f}\n\n".format(totalSSE))
        # print(type(self.centroids))
        # print(self.centroids)
        # print(self.classifications)
        # for centroid, value in self.centroids.items():
        #     f.write(np.array2string(value, precision=4, separator=","))
        #     f.write("\n")
        #     cluster_size = len(self.classifications[centroid])
        #     f.write("{:d}\n".format(cluster_size))
        #     cluster_sse = np.sum([self.compute_sse(i, value) for i in self.classifications[centroid]])
        #     f.write("{:.4f}\n\n".format(cluster_sse))
        # f.close()


