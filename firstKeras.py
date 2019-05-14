import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np


model = Sequential()
model.add(Dense(32, activation= 'relu',input_dim= 100))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

model.fit(x=data, y=labels, epochs=10, batch_size= 32)

print(model.get_weights())