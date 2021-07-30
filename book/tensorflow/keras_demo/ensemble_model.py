import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def get_model():
    inputs = tf.keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return tf.keras.Model(inputs, outputs)


if __name__ == '__main__':
    model1 = get_model()
    model2 = get_model()
    model3 = get_model()

    inputs = tf.keras.Input(shape=(128,))
    y1 = model1(inputs)
    y2 = model2(inputs)
    y3 = model3(inputs)
    outputs = layers.average([y1, y2, y3])
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    ensemble_model.summary()
