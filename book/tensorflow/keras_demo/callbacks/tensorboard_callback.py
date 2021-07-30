import tensorflow as tf
from book.tensorflow.keras_demo.train.training import get_compiled_model

if __name__ == '__main__':
    model = get_compiled_model()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir="./tb",
            histogram_freq=0,  # How often to log histogram visualizations
            embeddings_freq=0,  # How often to log embedding visualizations
            update_freq="epoch",
        )
    ]

    history = model.fit(x_train, y_train, batch_size=64, epochs=20, callbacks=callbacks, validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])

    # tensorboard --logdir=./tb
