# A SIMPLE MODEL TO USE LATER

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np


# HERE GOES DATASET
dataset = np.load('features.npy', allow_pickle=True)
# SHUFFLE DATA
np.random.shuffle(dataset)


# HERE WE SEPERATE FEATURES FROM LABELS
X = dataset[:, :-9]
y = dataset[:, -9:]

#print()
#exit(1)

model = Sequential()

# INPUT AND FIRST LAYER
model.add(Dense(2048, input_shape=(X.shape[1], ), activation='relu'))
model.add(Dropout(0.4))

# HIDDEN LAYERS
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

# OUTPUT LAYER
# CAN USE SIGMOID
model.add(Dense(y.shape[1], activation='sigmoid'))

# CAN USE https://keras.io/losses/
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, validation_split=0.2, epochs=50, batch_size=1024)

_, accuracy = model.evaluate(X, y)

print('Accuracy : ', accuracy * 100)

model.save('test_model.h5')

