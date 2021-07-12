import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import Dense, Conv2D, Flatten
from tensorflow import keras

from model import Model


class TicTacToeModelCnn(Model):
    def __init__(self, model=None):
        super().__init__(ninput=9,
                         layers=None,
                         model=model)

    def build_model(self, ninput, layers):
        input_layer = Input(shape=(3, 3, 1))
        x = input_layer
        x = Conv2D(64, (2, 2), activation='relu')(x)
        x = Conv2D(64, (2, 2), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', )(x)
        output_layer = Dense(3, activation='softmax')(x)

        model = tf.keras.Model(input_layer, output_layer)

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
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
