import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
classifier1 = GaussianNB()
classifier1.fit(X, Y)
print(classifier1.predict([[-0.8, -0.8],
                           [1, 1.5],
                           [-1, 1],
                           [1, -1]]))
classifier2 = GaussianNB()
classifier2.partial_fit(X, Y, np.unique(Y))
print(classifier2.predict([[-0.8, -0.8],
                           [1, 1.5],
                           [-1, 1],
                           [1, -1]]))
classifier2.partial_fit([[1,-0.8]], [2])
print(classifier2.predict([[-0.8, -0.8],
                           [1, 1.5],
                           [-1, 1],
                           [1, -1]]))