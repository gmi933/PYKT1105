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
history = model.fit(inputList, resultList, epochs=200, batch_size=50,
          validation_split=0.1)
scores = model.evaluate(inputList, resultList)
type(scores)
for s in scores:
    print(s)

type(model.metrics_names)

for s in model.metrics_names:
    print(s)

for s,m in zip(scores, model.metrics_names):
    print(f"{m}:{s}")

type(history)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'] )
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])
plt.show()