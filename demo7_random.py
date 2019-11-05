import matplotlib.pyplot as plt
from sklearn import datasets

data1 = datasets.make_regression(6, 5, noise=5)
X = data1[0]
y = data1[1]
print(X)
for i in range(5):
    plt.scatter(X[:, i], y)
    plt.show()