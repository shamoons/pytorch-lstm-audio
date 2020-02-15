import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, Dense, TimeDistributed, Activation
from keras.models import Sequential
import wandb
from wandb.keras import WandbCallback
wandb.init(project="pytorch-lstm-audio")


VECTOR_SIZE = 100
BATCH_SIZE = 5000


def get_test_data(batches=1):
    inputs = tf.random.uniform(
        shape=(batches, 10, VECTOR_SIZE),
        minval=-1,
        maxval=1
    )

    outputs = inputs * 2
    return inputs, outputs


def get_training_data(batches=1):
    inputs = tf.random.uniform(
        shape=(batches, 10, VECTOR_SIZE),
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

    model = Sequential()
    model.add(LSTM(256, input_shape=(10, VECTOR_SIZE), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(VECTOR_SIZE, activation='linear')))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    print(model.summary())
    callbacks = [WandbCallback(), EarlyStopping(
        monitor='val_loss', patience=10)]

    trainX, trainY = get_training_data(BATCH_SIZE)

    model.fit(trainX.numpy(), trainY.numpy(), epochs=250, batch_size=32,
              verbose=1, callbacks=callbacks, validation_split=0.1)

    testX, testY = get_test_data(BATCH_SIZE)

    # print('testX', testX)
    scores = model.evaluate(testX.numpy(), testY.numpy(), verbose=1)
    print('scores', scores)
    # predicted = model.predict(testX.numpy(), verbose=1)
    # print('predicted', predicted)
    # print('testY', testY)


if __name__ == '__main__':
    main()
