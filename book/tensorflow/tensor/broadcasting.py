import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    """
    smaller tensors are "stretched" automatically to fit larger tensors when running combined operations on them
    """
    x = tf.constant([1, 2, 3])
    y = tf.constant(2)
    z = tf.constant([2, 2, 2])

    # All of these are the same computation
    print(tf.multiply(x, 2))
    print(x * y)
    print(x * z)

    # A broadcasted add: a [3, 1] times a [1, 4] gives a [3,4]
    x = tf.reshape(x, [3, 1])
    y = tf.range(1, 5)
    print(x, "\n")
    print(y, "\n")
    print(tf.multiply(x, y))

    # see what broadcasting looks like for dim (3, 3)
    print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
