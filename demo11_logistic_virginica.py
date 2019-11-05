from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(list(iris.keys()))
print(iris.target_names)
print(iris.DESCR)
print(iris.feature_names)
print(iris.filename)

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
print(X)
print(y)

regression1 = LogisticRegression()
regression1.fit(X, y)

X_plot = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X_plot.shape)
y_porb = regression1.predict_proba(X_plot)

plt.plot(X, y, "g^")
plt.plot(X_plot, y_porb[:, 1], 'g--', label='iris-virginica')
plt.plot(X_plot, y_porb[:, 0], 'b--', label='Not iris-virginica')
plt.xlabel("Petal width", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.show()

print(regression1.predict([[1.7], [1.5], [2.2], [2.4], [3]]))