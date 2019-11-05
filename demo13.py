import numpy as np
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [-3, -3], [1, 1], [2, 1], [3, 3]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = SVC()
clf.fit(X, Y)
print(type(clf))
print("predict:", clf.predict([[-0.8, -0.8], [-1, -0.8], [0, 0], [2, 2], [4, 4]]))