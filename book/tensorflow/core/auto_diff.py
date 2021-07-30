import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def scalar_demo():
    x = tf.Variable(3.0)

    with tf.GradientTape() as tape:
        loss = x ** 2

    grad = tape.gradient(loss, x)
    print(grad.numpy())  # dy/dx = 2*x = 6.0


def multi_dim_demo():
    w = tf.Variable(tf.random.normal((3, 2)), name='w')
    b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
    x = [[1., 2., 3.]]

    # persistent=True allows multiple calls to the gradient method
    with tf.GradientTape(persistent=True) as tape:
        y = x @ w + b
        loss = tf.reduce_mean(y ** 2)

    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
    print('dl_db', dl_db.numpy())
    print('dl_dw', dl_dw.numpy())


def model_demo():
    layer = tf.keras.layers.Dense(2, activation='relu')
    x = tf.constant([[1., 2., 3.]])

    with tf.GradientTape() as tape:  # Forward pass
        y = layer(x)
        loss = tf.reduce_mean(y ** 2)

    # Calculate gradients with respect to every trainable variable
    grad = tape.gradient(loss, layer.trainable_variables)
    for var, g in zip(layer.trainable_variables, grad):
        print(f'{var.name}, shape: {g.shape}')


def manual_control_watch():
    """tf.Tensor is not "watched" by default, and the tf.constant is not trainable"""
    x0 = tf.Variable(3.0, name='x0')
    x1 = tf.Variable(3.0, name='x1', trainable=False)  # Not trainable
    x2 = tf.Variable(2.0, name='x2') + 1.0  # Not a Variable: A variable + tensor returns a tensor.
    x3 = tf.constant(3.0, name='x3')  # Not a variable

    with tf.GradientTape() as tape:
        y = (x0 ** 2) + (x1 ** 2) + (x2 ** 2)

    grad = tape.gradient(y, [x0, x1, x2, x3])
    for g in grad:
        print(g)
    print([var.name for var in tape.watched_variables()])  # ['x0:0']


def manual_watch_tensor():
    x = tf.constant(3.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x ** 2

    dy_dx = tape.gradient(y, x)  # dy/dx = 2x
    print(dy_dx.numpy())


def manual_disable_watch_variable():
    x0 = tf.Variable(0.0)
    x1 = tf.Variable(10.0)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x1)
        y0 = tf.math.sin(x0)
        y1 = tf.nn.softplus(x1)
        y = y0 + y1
        ys = tf.reduce_sum(y)

    # dys/dx1 = exp(x1) / (1 + exp(x1)) = sigmoid(x1)
    grad = tape.gradient(ys, {'x0': x0, 'x1': x1})
    print('dy/dx0:', grad['x0'])
    print('dy/dx1:', grad['x1'].numpy())


def multi_y_demo():
    x = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        y0 = x ** 2
        y1 = 1 / x
    print(tape.gradient({'y0': y0, 'y1': y1}, x).numpy())  # the sum of the gradients of each target


def control_flow_demo():
    x = tf.constant(1.0)

    v0 = tf.Variable(2.0)
    v1 = tf.Variable(2.0)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        if x > 0.0:
            result = v0
        else:
            result = v1 ** 2

    dv0, dv1 = tape.gradient(result, [v0, v1])

    print(dv0)
    print(dv1)  # None, the gradient only connects to the variable that was used


if __name__ == '__main__':
    scalar_demo()
    multi_dim_demo()
    model_demo()
    manual_control_watch()
    manual_watch_tensor()
    manual_disable_watch_variable()
    multi_y_demo()
    control_flow_demo()
