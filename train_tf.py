from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, TimeDistributed, Activation
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def get_test_data():
    inputs = tf.random.uniform(
        shape=(1, 100),
        minval=-1,
        maxval=1
    )

    outputs = inputs * 2
    return inputs, outputs


def get_training_data():
    inputs = tf.random.uniform(
        shape=(1, 100),
        minval=-1,
        maxval=1
    )

    outputs = inputs * 2
    return inputs, outputs


def main2():
    N = 1000
    Tp = 800
    t = np.arange(0, N)
    x = np.sin(0.2 * t) + 2 * np.random.rand(N)
    df = pd.DataFrame(x)

    values = df.values
    train, test = values[0: Tp, :], values[Tp:N, :]

    step = 20
    train = np.append(train, np.repeat(train[-1, ], step))
    test = np.append(test, np.repeat(test[-1, ], step))

    print('train.shape', train.shape)
    print('test.shape', test.shape)

    trainX, trainY = convertToMatrix(train, step)
    testX, testY = convertToMatrix(test, step)

    print('trainX.shape', trainX.shape, 'trainY.shape', trainY.shape)
    print('testX.shape', testX.shape, 'testY.shape', testY.shape)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(SimpleRNN(units=32, input_shape=(1, step), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    print(model.summary())

    model.fit(trainX, trainY, epochs=100, batch_size=16, verbose=1)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    predicted = np.concatenate((trainPredict, testPredict), axis=0)
    trainScore = model.evaluate(trainX, trainY, verbose=1)

    print('trainScore', trainScore)


def main():
    tf.random.set_seed(0)

    training_data = get_training_data()

    trainX, trainY = training_data

    trainX = np.expand_dims(trainX, 0)
    trainY = np.expand_dims(trainY, 0)

    print('trainX.shape', trainX.shape)
    print('trainY.shape', trainY.shape)

    model = Sequential()
    model.add(LSTM(100, input_shape=(1, 100), return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='linear')))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    print(model.summary())
    callbacks = [EarlyStopping(monitor='loss', patience=2)]
    model.fit(trainX, trainY, epochs=100, batch_size=1,
              verbose=2, callbacks=callbacks)

    testX, testY = get_test_data()
    testX = np.expand_dims(testX, 0)
    testY = np.expand_dims(testY, 0)
    print('trainX', trainX)
    scores = model.evaluate(trainX, trainY, verbose=1, batch_size=1)
    print('scores', scores)
    predicted = model.predict(trainX, batch_size=1, verbose=1)
    print('predicted', predicted)
    print('trainY', trainY)


if __name__ == '__main__':
    main()
