import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

np.random.seed(0)

# Import data
with open('/Users/marinaalonsopoal/PycharmProjects/Marina/centroids_tree_nhl.obj', 'rb') as f:
    data_pkl = pkl.load(f)

# Pre-process data
data = np.zeros((len(data_pkl), 3))
data[:, 2] = np.random.randint(1, 10, len(data_pkl))
for t, points in enumerate(data_pkl):
    if len(points) == 1:
        data[t, 0:2] = np.asarray(points)
    else:
        data[t, 0:2] = np.asarray(points[0])
# np.save('data.npy', data)

x = data[:, 0]
y = data[:, 1]
s = data[:, 2]
t = np.arange(data.shape[0])

plt.scatter(t, x, s=75, c='r', zorder=1, label='x component')
plt.scatter(t, y, s=75, c='b', zorder=1, label='y component')
for time in range(len(t)):
    c = s[time]/(np.max(s))
    if time==1:
        plt.axvline(x=time, color=(c, c, c), linestyle='-', linewidth=12, zorder=0, alpha=0.5, label='score value')
    else:
        plt.axvline(x=time, color=(c, c, c), linestyle='-', linewidth=12, zorder=0, alpha=0.5)
plt.legend(loc="center left")
plt.title('Trajectory Evolution')
plt.xlabel('Time [s]')
plt.show()
