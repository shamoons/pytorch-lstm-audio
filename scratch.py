import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed

VECTOR_SIZE = 3
SEQ_LENGTH = 1
BATCH_SIZE = 1000


def get_training_data(batches=1):
    inputs = tf.random.uniform(
        shape=(batches, SEQ_LENGTH, VECTOR_SIZE),
        minval=-1,
        maxval=1
    )

    outputs = inputs
    return np.array(inputs), np.array(outputs)


model = Sequential()
model.add(LSTM(256, input_shape=(SEQ_LENGTH, VECTOR_SIZE), return_sequences=True))
# model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(VECTOR_SIZE, activation='relu')))

model.compile(loss='mean_squared_error', optimizer='adam')

trainX, trainY = get_training_data(100)
model.fit(trainX, trainY, batch_size=32, verbose=1,
          epochs=100, validation_split=0.1)

testX, testY = get_training_data(100)

predY = model.predict([[testX[0]]], batch_size=1)

print(testX[0], '\t', predY, '\n=====\n')
