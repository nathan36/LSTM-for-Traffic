import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import callbacks


def buildModel(dataLength, time_step):
    # define input

    # define layer

    output = concatenate(
        [

        ]
    )

    output = Dense(time_step, activation='linear', name='weightedAverage_output')(output)

    model = Model(
        inputs=
        [

        ],
        outputs=
        [
            output
        ]
    )

    model.compile(optimizer='rmsprop', loss='mse')

    return model


if __name__ == '__main__':
    rnn = buildModel()
    rnn.fit(
        [
            trainingData["price"],
            trainingData['volume'],
        ],
        [
            trainingLabels["price"]
        ],
        validation_data=(
            [
                testingData["price"],
                testingData["volume"],
            ],
            [
                testingLabels["price"]
            ]),
        epochs=1,
        batch_size=3000,
        callbacks=[
            callbacks.TensorBoard('logs/Graph'),
        ])
