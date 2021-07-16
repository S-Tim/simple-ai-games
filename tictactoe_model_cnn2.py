import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import Dense, Conv2D, Flatten
from tensorflow import keras

from model import Model


class TicTacToeModelCnn2(Model):
    def __init__(self, model=None):
        super().__init__(ninput=9,
                         layers=None,
                         model=model)

    def build_model(self, ninput, layers):
        input_layer = Input(shape=(3, 3, 1))
        x = input_layer
        x = Conv2D(128, (2, 2), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(1)(x)

        model = tf.keras.Model(input_layer, output_layer)

        opt = keras.optimizers.RMSprop(learning_rate=0.0001)
        model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

    def preprocess(self, plays):
        features, targets = super().preprocess(plays)
        # print(features.shape)
        reshaped = features.reshape((-1, 3, 3))
        # print(reshaped.shape)

        return reshaped, targets

    def predict(self, states):
        reshaped = np.array(states).reshape((-1, 3, 3))
        return self.keras_model.predict(reshaped)
