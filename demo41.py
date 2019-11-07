import os

import numpy
from keras.layers import Dense
from keras.models import Sequential

print(os.getcwd())

# load data
dataset1 = numpy.loadtxt('data\\diabetes.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print("features size:", inputList.shape)
print("result size:", resultList.shape)

model = Sequential()
# 14, 8
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(inputList, resultList, epochs=200, batch_size=50)

scores = model.evaluate(inputList, resultList)
print("metrics name:", model.metrics_names)
print("\n")
print("\n %s: %.3f%%" % (model.metrics_names[1],
                         scores[1] * 100))
print("\n %s: %.3f" % (model.metrics_names[0],
                       scores[0]))