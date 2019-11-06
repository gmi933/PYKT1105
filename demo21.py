import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

#np.random.seed(13579)
X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]
#print(type(X), X.shape)
print(X[:5, ], X[50:55, ], X[100:105])
KS = [2, 3, 4, 5]
for K in KS:
    kmeans1 = KMeans(n_init=1, n_clusters=K)
    kmeans1.fit(X)
    # total distance between node to center
    print(kmeans1.inertia_)
    # the center that the point belongs to
    #print(kmeans1.labels_)
    # 3 centers
    #print(kmeans1.cluster_centers_)
    colors = ['c', 'm', 'y', 'k', 'r']
    markers = ['.', '*', '^', 'x', 's']
    for i in range(K):
        dataX = X[kmeans1.labels_ == i]
        plt.scatter(dataX[:, 0], dataX[:, 1],
                    c=colors[i], marker=markers[i])
        #print(dataX.size)
    plt.show()