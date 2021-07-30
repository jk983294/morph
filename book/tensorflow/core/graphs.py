import timeit
import tensorflow as tf


def my_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


@tf.function
def outer_function(x):
    y = tf.constant([[2.0], [3.0]])
    b = tf.constant(4.0)
    return my_function(x, y, b)


def function_demo():
    a_function_that_uses_a_graph = tf.function(my_function)
    x1 = tf.constant([[1.0, 2.0]])
    y1 = tf.constant([[2.0], [3.0]])
    b1 = tf.constant(4.0)

    orig_value = my_function(x1, y1, b1).numpy()
    tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()
    print(orig_value)
    print(tf_function_value)


def decorate_demo():
    print(outer_function(tf.constant([[1.0, 2.0]])).numpy())


def simple_relu(x):
    if tf.greater(x, 0):
        return x
    else:
        return 0


def convert_py_method_to_graph():
    tf_simple_relu = tf.function(simple_relu)  # TensorFlow `Function` that wraps `simple_relu`
    print("First branch, with graph:", tf_simple_relu(tf.constant(1)).numpy())
    print("Second branch, with graph:", tf_simple_relu(tf.constant(-1)).numpy())
    # print(tf.autograph.to_code(simple_relu))  # graph-generating output of AutoGraph
    # print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())


@tf.function
def get_MSE(y_true, y_pred):
    sq_diff = tf.pow(y_true - y_pred, 2)
    return tf.reduce_mean(sq_diff)


def power(x, y):
    result = tf.eye(10, dtype=tf.dtypes.int32)
    for _ in range(y):
        result = tf.matmul(x, result)
    return result


def speed_up_demo():
    x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)
    print("Eager execution:", timeit.timeit(lambda: power(x, 100), number=1000))
    power_as_graph = tf.function(power)
    print("Graph execution:", timeit.timeit(lambda: power_as_graph(x, 100), number=1000))


if __name__ == '__main__':
    function_demo()
    decorate_demo()
    speed_up_demo()
