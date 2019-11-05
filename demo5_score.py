import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8]]
values = [1, 4, 5.5]
regression1 = linear_model.LinearRegression()
regression1.fit(features, values)
print(regression1)

plt.scatter([[0], [1], [2]], [1, 4, 5.5], c='g')
plt.scatter([[1], [3], [8]], [1, 4, 5.5], c='b')
# plt.show()
print(f'coefficient={regression1.coef_}')
print(f'intercept={regression1.intercept_}')
print(f'x1 coef={regression1.coef_[0]},x2 coef={regression1.coef_[1]}')
print('predict1=', regression1.predict([[0.8, 0.8], [2, 4], [3, 5]]))
result = regression1.predict([[0.8, 0.8], [2, 4], [3, 5]])
print(regression1.score([[0.8, 0.8], [2, 4], [3, 5]], result))
print(regression1.score([[0.8, 0.8], [2, 4], [3, 5]], [4, 8, 11.3]))
print(regression1.score([[0.8, 0.8], [2, 4], [3, 5]], [-4, -8, -11.3]))
print(regression1.score([[0.8, 0.8], [2, 4], [3, 5]], [5, 7, 10]))