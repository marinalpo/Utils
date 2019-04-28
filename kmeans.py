import numpy as np
import matplotlib.pyplot as plt
import math


def euclidianDistance(p1, p2, Ndim):
    dist = 0
    for i_dim in range(Ndim):
        dist = dist + math.pow(p1[i_dim] - p2[i_dim], 2)
    dist = math.sqrt(dist)
    return dist


def plotScatter(data, idx, centroids):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots()
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 10))
    for k in range(len(centroids)):
        label = 'Class ' + str(k + 1)
        idx_class = np.asarray(np.where(idx == k))
        data_class = data[idx_class[0, :], :]
        ax.scatter(data_class[:, 0], data_class[:, 1], edgecolors=color[k], facecolors='None', label=label)
        ax.scatter(centroids[k, 0],centroids[k, 1], s=80, color=color[k])
    ax.axis('equal')
    plt.title('K-Means Final Distribution')
    plt.legend()
    plt.show()


def assignCentroid(data, centroids):
    idx = np.zeros(len(data))
    for i in range(len(data)):
        dist = np.zeros(len(centroids))
        for k in range(len(centroids)):
            dist[k] = euclidianDistance(data[i, :], centroids[k, :], 2)
        idx[i] = np.argmin(dist)
    return idx


def updateCentroids(data, idx, K):
    for k2 in range(K):
        if k2 in idx:
            idx_class = np.asarray(np.where(idx == k2))
            data_class = data[idx_class[0, :], :]
            centroids[k2, :] = np.mean(data_class, axis=0)
    return centroids


N = 1000  # Number of samples per class
K = 5  # Number of classes (maximum 7 for colour restrictions)
np.random.seed(2)
data = 10 * np.random.rand(N, 2)
centroids = 10 * np.random.rand(K, 2)
no_changes = 0
threshold = 5  # Number of additional iterations

while no_changes < threshold:
    idx = assignCentroid(data, centroids)
    old_centroids = np.around(centroids, 2)
    plotScatter(data, idx, centroids)
    centroids = updateCentroids(data, idx, K)
    new_centroids = np.around(centroids, 2)
    equal = np.count_nonzero(old_centroids == new_centroids)
    if equal >= K*2:
        no_changes = no_changes + 1

