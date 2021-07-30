import tensorflow as tf
from book.tensorflow.keras_demo.train.training import get_compiled_model

if __name__ == '__main__':
    model = get_compiled_model()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    x_valid, y_valid = x_train[-1000:], y_train[-1000:]
    x_train, y_train = x_train[:-1000], y_train[:-1000]
    print(x_train.shape, x_valid.shape)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(64)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    val_dataset = val_dataset.batch(64)

    # Since the dataset already takes care of batching, we don't pass a `batch_size` argument.
    model.fit(train_dataset, epochs=3, validation_data=val_dataset)

    print("Evaluate")
    result = model.evaluate(test_dataset)
    dict(zip(model.metrics_names, result))
