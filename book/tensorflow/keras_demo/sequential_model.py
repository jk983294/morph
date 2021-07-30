import tensorflow as tf
from tensorflow.keras import layers


def basic_demo():
    model = tf.keras.Sequential(
        [
            layers.Dense(2, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(4, name="layer3"),
        ]
    )
    # Call model on a test input
    x = tf.ones((3, 3))
    y = model(x)

    print(model.layers)


def seq_add_demo():
    model = tf.keras.Sequential()
    model.add(layers.Dense(2, activation="relu", name="layer1"))
    model.add(layers.Dense(3, activation="relu", name="layer2"))
    model.add(layers.Dense(4, name="layer3"))
    print(len(model.layers))
    print(model.layers[0].weights)

    """It creates its weights the first time it is called on an input"""
    y = model(tf.ones((1, 4)))
    print(model.layers[0].weights)  # Now it has weights, of shape (4, 2) and (2,)
    model.summary()  # display its contents


def seq_add_with_input_dim_demo():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(4,)))
    model.add(layers.Dense(2, activation="relu"))
    # model.add(layers.Dense(2, activation="relu", input_shape=(4,)))  # alternative above
    model.summary()


def feature_extract_by_layers():
    initial_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(250, 250, 3)),
            layers.Conv2D(32, 5, strides=2, activation="relu"),
            layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
            layers.Conv2D(32, 3, activation="relu"),
        ]
    )
    feature_extractor1 = tf.keras.Model(
        inputs=initial_model.inputs,
        outputs=[layer.output for layer in initial_model.layers],
    )
    feature_extractor2 = tf.keras.Model(
        inputs=initial_model.inputs,
        outputs=initial_model.get_layer(name="my_intermediate_layer").output,
    )

    x = tf.ones((1, 250, 250, 3))
    features1 = feature_extractor1(x)
    features2 = feature_extractor2(x)
    print(features1)
    print(features2)


if __name__ == '__main__':
    basic_demo()
    seq_add_demo()
    seq_add_with_input_dim_demo()
    feature_extract_by_layers()
