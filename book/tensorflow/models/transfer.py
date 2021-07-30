import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import os

if __name__ == '__main__':
    tfds.disable_progress_bar()
    train_ds, validation_ds, test_ds = tfds.load(
        "cats_vs_dogs",
        data_dir=os.path.expanduser("~/junk/"),
        # Reserve 10% for validation and 10% for test
        split=[
            tfds.Split.TRAIN.subsplit(tfds.percent[:40]),
            tfds.Split.TRAIN.subsplit(tfds.percent[40:50]),
            tfds.Split.TRAIN.subsplit(tfds.percent[50:60])
        ],
        as_supervised=True,  # Include labels
    )

    print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
    print("Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds))
    print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))

    size = (150, 150)
    train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

    batch_size = 32
    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False,  # Do not include the ImageNet classifier at the top.
    )
    base_model.trainable = False  # Freeze the base_model

    # Create new model on top
    inputs = keras.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)

    # Pre-trained Xception weights requires that input be normalized from (0, 255) to a range (-1., +1.),
    # the normalization layer does the following, outputs = (inputs - mean) / sqrt(var)
    norm_layer = keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([127.5] * 3)
    var = mean ** 2
    x = norm_layer(x)  # Scale inputs to [-1, +1]
    norm_layer.set_weights([mean, var])

    # The base model contains batchnorm layers.
    # We want to keep them in inference mode when we unfreeze the base model for fine-tuning,
    # so we make sure that the base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    """train the top layer"""
    epochs = 20
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

    """do a round of fine-tuning of the entire model"""
    base_model.trainable = True
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate for fine tuning
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    epochs = 10
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
