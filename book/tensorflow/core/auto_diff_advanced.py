import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def stop_record_demo():
    """useful to reduce overhead if you do not wish to differentiate a complicated operation"""
    x = tf.Variable(2.0)
    y = tf.Variable(3.0)

    with tf.GradientTape() as t:
        x_sq = x * x
        with t.stop_recording():
            y_sq = y * y
        z = x_sq + y_sq

    grad = t.gradient(z, {'x': x, 'y': y})

    print('dz/dx:', grad['x'])  # 2*x => 4
    print('dz/dy:', grad['y'])


def stop_gradient_demo():
    """used to stop gradients from flowing along a particular path, without needing access to the tape itself"""
    x = tf.Variable(2.0)
    y = tf.Variable(3.0)

    with tf.GradientTape() as t:
        y_sq = y ** 2
        z = x ** 2 + tf.stop_gradient(y_sq)

    grad = t.gradient(z, {'x': x, 'y': y})

    print('dz/dx:', grad['x'])  # 2*x => 4
    print('dz/dy:', grad['y'])


# Establish an identity operation, but clip during the gradient pass.
@tf.custom_gradient
def clip_gradients(y):
    def backward(dy):
        return tf.clip_by_norm(dy, 0.5)

    return y, backward


def custom_gradient_demo():
    v = tf.Variable(2.0)
    with tf.GradientTape() as t:
        output = clip_gradients(v * v)
    print(t.gradient(output, v))  # calls "backward", which clips 4 to 2


def multiple_tapes_demo():
    x0 = tf.constant(0.0)
    x1 = tf.constant(0.0)

    with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
        tape0.watch(x0)
        tape1.watch(x1)
        y0 = tf.math.sin(x0)
        y1 = tf.nn.sigmoid(x1)
        y = y0 + y1
        ys = tf.reduce_sum(y)
    print(tape0.gradient(ys, x0).numpy())  # cos(x) => 1.0
    print(tape1.gradient(ys, x1).numpy())  # sigmoid(x1)*(1-sigmoid(x1)) => 0.25


def higher_order_gradients():
    x = tf.Variable(1.0)

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            y = x * x * x
        # Compute the gradient inside the outer `t2` context manager
        # which means the gradient computation is differentiable as well.
        dy_dx = t1.gradient(y, x)
    d2y_dx2 = t2.gradient(dy_dx, x)

    print('dy_dx:', dy_dx.numpy())  # 3 * x**2 => 3.0
    print('d2y_dx2:', d2y_dx2.numpy())  # 6 * x => 6.0


def jacobian_demo():
    """a vector-target with respect to a scalar-source"""
    x = tf.linspace(-10.0, 10.0, 200 + 1)
    delta = tf.Variable(0.0)

    with tf.GradientTape() as tape:
        y = tf.nn.sigmoid(x + delta)

    dy_dx = tape.jacobian(y, delta)
    print(y.shape)
    print(dy_dx.shape)


def jacobian_tensor_demo():
    x = tf.random.normal([7, 5])  # batch = 7, features = 5
    layer = tf.keras.layers.Dense(10, activation=tf.nn.relu)  # output = 10

    with tf.GradientTape(persistent=True) as tape:
        y = layer(x)

    j = tape.jacobian(y, layer.kernel)
    print(j.shape)  # (7, 10, 5, 10)
    print(y.shape)  # (7, 10)
    print(layer.kernel.shape)  # (5, 10)

    g = tape.gradient(y, layer.kernel)  # gradient is sum of all batch's gradient
    print('g.shape:', g.shape)  # (5, 10)

    j_sum = tf.reduce_sum(j, axis=[0, 1])
    delta = tf.reduce_max(abs(g - j_sum)).numpy()
    assert delta < 1e-3
    print('delta:', delta)


def hessian_demo():
    x = tf.random.normal([7, 5])
    layer1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)
    layer2 = tf.keras.layers.Dense(6, activation=tf.nn.relu)

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            x = layer1(x)
            x = layer2(x)
            loss = tf.reduce_mean(x ** 2)

        g = t1.gradient(loss, layer1.kernel)
    h = t2.jacobian(g, layer1.kernel)
    print(f'layer.kernel.shape: {layer1.kernel.shape}')  # (5, 8)
    print(f'h.shape: {h.shape}')  # (5, 8, 5, 8)

    n_params = tf.reduce_prod(layer1.kernel.shape)
    g_vec = tf.reshape(g, [n_params, 1])
    h_mat = tf.reshape(h, [n_params, n_params])
    eps = 1e-3
    eye_eps = tf.eye(h_mat.shape[0]) * eps
    # X(k+1) = X(k) - (∇²f(X(k)))^-1 @ ∇f(X(k))
    # h_mat = ∇²f(X(k))
    # g_vec = ∇f(X(k))
    update = tf.linalg.solve(h_mat + eye_eps, g_vec)

    # Reshape the update and apply it to the variable.
    _ = layer1.kernel.assign_sub(tf.reshape(update, layer1.kernel.shape))


if __name__ == '__main__':
    stop_record_demo()
    stop_gradient_demo()
    custom_gradient_demo()
    multiple_tapes_demo()
    higher_order_gradients()
    jacobian_demo()
    jacobian_tensor_demo()
    hessian_demo()
