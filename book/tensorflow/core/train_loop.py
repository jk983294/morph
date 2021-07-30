import tensorflow as tf
import matplotlib.pyplot as plt


def generate_data(TRUE_W, TRUE_B):
    NUM_EXAMPLES = 1000
    x = tf.random.normal(shape=[NUM_EXAMPLES])
    noise = tf.random.normal(shape=[NUM_EXAMPLES])
    y = x * TRUE_W + TRUE_B + noise
    return x, y


class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.w * x + self.b


def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))


def train(model, x, y, learning_rate):
    with tf.GradientTape() as t:
        # Trainable variables are automatically tracked by GradientTape
        current_loss = loss(y, model(x))

    # Use GradientTape to calculate the gradients with respect to W and b
    dw, db = t.gradient(current_loss, [model.w, model.b])

    # Subtract the gradient scaled by the learning rate
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


def training_loop(model, x, y, epochs):
    # Collect the history of W-values and b-values to plot later
    Ws, bs = [], []
    for epoch in epochs:
        # Update the model with the single giant batch
        train(model, x, y, learning_rate=0.1)

        # Track this before I update
        Ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(y, model(x))

        print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" % (epoch, Ws[-1], bs[-1], current_loss))
    return Ws, bs


if __name__ == '__main__':
    TRUE_W = 3.0
    TRUE_B = 2.0
    """Given x and y, try to find the slope and offset of a line via simple linear regression"""
    x, y = generate_data(TRUE_W, TRUE_B)

    model = MyModel()
    print("Variables:", model.variables)
    assert model(3.0).numpy() == 15.0  # Verify the model works

    # plt.scatter(x, y, c="b")
    # plt.scatter(x, model(x), c="r")
    # plt.show()
    print("Current loss: %1.6f" % loss(y, model(x)).numpy())
    print("Starting: W=%1.2f b=%1.2f, loss=%2.5f" % (model.w, model.b, loss(y, model(x))))
    epochs = range(10)
    Ws, bs = training_loop(model, x, y, epochs)

    # Plot it
    plt.plot(epochs, Ws, "r", epochs, bs, "b")
    plt.plot([TRUE_W] * len(epochs), "r--", [TRUE_B] * len(epochs), "b--")
    plt.legend(["W", "b", "True W", "True b"])
    plt.show()

    # Visualize how the trained model performs
    plt.scatter(x, y, c="b")
    plt.scatter(x, model(x), c="r")
    plt.show()
    print("Current loss: %1.6f" % loss(model(x), y).numpy())
