import tensorflow as tf
import numpy as np
from book.tensorflow.keras_demo.train.training import get_compiled_model
import os


def make_or_restore_model(cp_dir):
    # Either restore the latest model, or create a fresh one if there is no checkpoint available.
    checkpoints = [cp_dir + "/" + name for name in os.listdir(cp_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


if __name__ == '__main__':
    checkpoint_dir = "./ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model = make_or_restore_model(checkpoint_dir)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            # The saved model name will include the current epoch.
            filepath=checkpoint_dir + "/mymodel_{epoch}",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            verbose=1,
        )
    ]

    history = model.fit(x_train, y_train, batch_size=64, epochs=5, callbacks=callbacks, validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
