import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8]]
values = [1, 4, 5.5]
regression1 = linear_model.LinearRegression()
regression1.fit(features, values)
print(regression1)

plt.scatter([[0], [1], [2]], [1, 4, 5.5], c='g')
plt.scatter([[1], [3], [8]], [1, 4, 5.5], c='b')
plt.show()
print(f'coefficient={regression1.coef_}')
print(f'intercept={regression1.intercept_}')
print(f'x1 coef={regression1.coef_[0]},x2 coef={regression1.coef_[1]}')