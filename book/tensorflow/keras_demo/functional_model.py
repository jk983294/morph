import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def basic_demo():
    inputs = tf.keras.Input(shape=(784,))
    print(inputs.shape)  # (None, 784)  batch size is not specified
    dense = layers.Dense(64, activation="relu")
    x = dense(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()
    # tf.keras.utils.plot_model(model, "my_first_model.png", show_shapes=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    path_ = "/tmp/my_model"
    model.save(path_)
    del model
    # Recreate the exact same model purely from the file:
    model = tf.keras.models.load_model(path_)


if __name__ == '__main__':
    basic_demo()
