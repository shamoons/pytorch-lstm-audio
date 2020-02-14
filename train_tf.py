from keras.models import Sequential
from keras.layers import Input, SimpleRNN, Dense

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def get_training_data():
    inputs = tf.random.uniform(
        shape=(1, 100),
        minval=-1,
        maxval=1
    )

    inputs = tf.constant([[1, 2, 3], [4, 5, 6]])

    outputs = inputs * 2
    return inputs, outputs


def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d, ])
        Y.append(data[d, ])
    return np.asarray(X), np.asarray(Y)


def main():
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


def main2():
    tf.random.set_seed(0)

    training_data = get_training_data()

    hidden_size = 100

    encoder_inputs = Input(shape=(None, 200))
    encoder = LSTM(100, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(1, 100)))

    print(model.summary())
    # model.add(LSTM(20, batch_input_shape=(7, 5, 100), return_sequences=True, stateful=True))
    # print(training_data[0].eval())
    # print(training_data[1].eval())


if __name__ == '__main__':
    main()
