import numpy as np
import matplotlib.pyplot as plt
import math


def euclidianDistance(p1, p2, Ndim):
    dist = 0
    for i_dim in range(Ndim):
        dist = dist + math.pow(p1[i_dim] - p2[i_dim], 2)
    dist = math.sqrt(dist)
    return dist


def plotScatter(Nclas, train, test, dmax):
    circle = plt.Circle((test[0], test[1]), dmax, color='r', fill=False)
    fig, ax = plt.subplots()
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 10))
    for i_class in range(Nclas):
        label = 'Class ' + str(i_class+1)
        ax.scatter(train[:, i_class * 2], train[:, i_class * 2 + 1], label=label)
    ax.scatter(test[0], test[1], label='Test')
    ax.add_artist(circle)
    plt.title('KNN 2D Distribution')
    plt.legend()
    plt.show()


Nclas = 3  # Number of classes
N = 10  # Number of samples per class
Ndim = 2  # Number of dimensions
K = 3  # Number of neighbours
# WARNING: To avoid unaccurate results due to voting draw, select only odd values of K

train = 10*np.random.rand(N, Ndim * Nclas)
test = 10*np.random.rand(Ndim)

distances = np.zeros((N, Nclas))
knn_class = np.zeros(K)

for i_class in range(Nclas):
    for x in range(N):
        distances[x, i_class] = euclidianDistance(train[x, i_class*Ndim:(i_class+1)*Ndim], test, Ndim)

dist_sort = np.sort(np.array(distances).flatten())
dmax = dist_sort[K-1]

for k in range(K):
    knn = np.where(distances == dist_sort[k])
    knn_class[k] = knn[1]

knn_class = np.bincount(knn_class.astype(np.int64))
knn_class = np.argmax(knn_class)

if Ndim == 2:
    plotScatter(Nclas, train, test, dmax)

print('Point ', np.around(test, decimals=2), ' is assigned to class', knn_class+1)
