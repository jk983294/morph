import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


if __name__ == '__main__':
    inputs = tf.keras.Input((4,))
    outputs = CustomDense(10)(inputs)
    model = tf.keras.Model(inputs, outputs)

    config = model.get_config()
    new_model = tf.keras.Model.from_config(config, custom_objects={"CustomDense": CustomDense})
