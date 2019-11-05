import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(type(diabetes))
print(type(diabetes.data), diabetes.data.shape)
print(type(diabetes.target), diabetes.target.shape)
dataForTest = -50
data_train = diabetes.data[:dataForTest]
target_train = diabetes.target[:dataForTest]
print(f'data train shape:{data_train.shape}')
print(f'target train shape:{target_train.shape}')
data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]
print(f"data test shape:{data_test.shape}")
print(f"target test shape:{target_test.shape}")

regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)

print('score:', regression1.score(data_train, target_train))
print('score2:', regression1.score(data_test, target_test))

target_predict = regression1.predict(data_test)
print(type(target_predict), target_predict.shape)
for i in range(dataForTest,0):
    print('[I]predict:%.2f/%.2f' % (target_predict[i], target_test[i]))

for i in range(dataForTest, 0):
    data = np.array(data_test[i])
    # print("before reshape, data=", data.shape)
    dataArray = data.reshape(1, 10)
    # print("after reshape, data=", dataArray.shape)
    print('[II]predict:%.2f/%.2f' % (regression1.predict(dataArray)[0], target_test[i]))