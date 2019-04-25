import numpy as np
import matplotlib.pyplot as plt
import math


def euclidianDistance(p1, p2):
    dist = math.pow(p1[0, 0]-p2[0, 0], 2)+math.pow(p1[0, 1]-p2[0, 1], 2)
    dist = math.sqrt(dist)
    return dist


def plotScatter(Nclas, train, test):
    for i_class in range(Nclas):
        label = 'Class ' + str(i_class+1)
        plt.scatter(train[:, i_class * 2], train[:, i_class * 2 + 1], label=label)
    plt.scatter(test[0, 0], test[0, 1], label='Test')
    plt.title('1NN Distribution')
    plt.legend()
    plt.show()


Nclas = 5  # Number of classes
N = 10  # Number of samples per class

train = 10*np.random.rand(N, 2*Nclas)
test = 10*np.random.rand(1, 2)

distances = np.zeros((N, Nclas))

for i_class in range(Nclas):
    for x in range(N):
        distances[x, i_class] = euclidianDistance(np.reshape(train[x, (i_class * 2, i_class * 2 + 1)], (1, 2)), test)

class_test=np.where(distances == distances.min())

plotScatter(Nclas, train, test)
print('Point ', np.around(test, decimals=2), ' is assigned to class', class_test[1]+1)
