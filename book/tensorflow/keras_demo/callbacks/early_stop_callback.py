import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

if __name__ == '__main__':
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

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        )
    ]

    history = model.fit(x_train, y_train, batch_size=64, epochs=20, callbacks=callbacks, validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
