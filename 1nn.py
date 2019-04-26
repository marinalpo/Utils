import numpy as np
import matplotlib.pyplot as plt
import math


def euclidianDistance(p1, p2, Ndim):
    dist = 0
    for i_dim in range(Ndim):
        dist = dist + math.pow(p1[i_dim] - p2[i_dim], 2)
    dist = math.sqrt(dist)
    return dist


def plotScatter(Nclas, train, test):
    for i_class in range(Nclas):
        label = 'Class ' + str(i_class+1)
        plt.scatter(train[:, i_class * 2], train[:, i_class * 2 + 1], label=label)
    plt.scatter(test[0], test[1], label='Test')
    plt.title('1NN Distribution')
    plt.legend()
    plt.show()


Nclas = 4  # Number of classes
N = 50  # Number of samples per class
Ndim = 2  # Number of dimensions

train = 10*np.random.rand(N, Ndim * Nclas)
test = 10*np.random.rand(Ndim)

distances = np.zeros((N, Nclas))

for i_class in range(Nclas):
    for x in range(N):
        distances[x, i_class] = euclidianDistance(train[x,i_class*Ndim:(i_class+1)*Ndim], test, Ndim)

class_test = np.where(distances == distances.min())

if Ndim == 2:
    plotScatter(Nclas, train, test)

print('Point ', np.around(test, decimals=2), ' is assigned to class', class_test[1]+1)
