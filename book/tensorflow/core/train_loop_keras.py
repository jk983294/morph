import tensorflow as tf
import matplotlib.pyplot as plt


def generate_data(TRUE_W, TRUE_B):
    NUM_EXAMPLES = 1000
    x = tf.random.normal(shape=[NUM_EXAMPLES])
    noise = tf.random.normal(shape=[NUM_EXAMPLES])
    y = x * TRUE_W + TRUE_B + noise
    return x, y


class MyModelKeras(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def call(self, x):
        return self.w * x + self.b


if __name__ == '__main__':
    TRUE_W = 3.0
    TRUE_B = 2.0
    """Given x and y, try to find the slope and offset of a line via simple linear regression"""
    x, y = generate_data(TRUE_W, TRUE_B)

    model = MyModelKeras()
    assert model(3.0).numpy() == 15.0  # Verify the model works

    # compile sets the training parameters
    model.compile(
        # By default, fit() uses tf.function().  You can
        # turn that off for debugging, but it is on now.
        run_eagerly=False,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=tf.keras.losses.mean_squared_error,
    )

    print(x.shape[0])
    model.fit(x, y, epochs=10, batch_size=x.shape[0])

    # Visualize how the trained model performs
    plt.scatter(x, y, c="b")
    plt.scatter(x, model(x), c="r")
    plt.show()
