import matplotlib.pyplot as plt
from sklearn import linear_model

regression1 = linear_model.LinearRegression()
features = [[1], [2], [3], [4]]
values = [1, 4, 15, 20]
plt.scatter(features, values, c='green')
plt.show()
regression1.fit(features, values)
print(regression1)  # make a breakpoint here
print(f'coefficient={regression1.coef_[0]}')
print(f'intercept={regression1.intercept_}')
# score
print(regression1.score(features, values))
print(regression1.score(features, [2, 3, 13, 19]))
# plot
range1 = [0, 4]
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='gray')
plt.scatter(features, values, c='green', marker='.')
plt.scatter(features,  [2, 3, 13, 19], c='blue', marker='*')
#plt.plot()
plt.show()