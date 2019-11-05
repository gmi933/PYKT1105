import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()
pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)
target = iris.target
print(data.shape)
print(data[:5, ])
# svc = svm.SVC(C=1) # default
# svc = svm.SVC(kernel='linear')
svc = svm.SVC(kernel='poly')
# svc = svm.SVC(kernel='rbf')
# svc = svm.SVC(kernel='sigmoid') # not proper here
#svc = svm.SVC(kernel="precomputed") # not work here
svc.fit(data, target)
# plot grid
datamax = data.max(axis=0) + 1
datamin = data.min(axis=0) - 1
print(datamax)
print(datamin)
n = 2000
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
print(X.shape)
print(Y.shape)
eachGrid = np.c_[X.ravel(), Y.ravel()]
print(eachGrid[:10])
print(type(eachGrid), eachGrid.shape)
Z = svc.predict(eachGrid)
print(type(Z), Z.shape)
plt.contour(X, Y, Z.reshape(X.shape), colors='k')
for c, s in zip([0, 1, 2], ['o', '^', '*']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)
plt.show()