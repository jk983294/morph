import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    """
    TensorFlow uses C-style "row-major" memory ordering
    with the requested shape, pointing to the same data
    """
    x = tf.constant([[1], [2], [3]])
    print(x.shape)  # (3, 1)
    print(x.shape.as_list())  # [3, 1]

    reshaped = tf.reshape(x, [1, 3])
    print(x.shape)  # (3, 1)
    print(reshaped.shape)  # (1, 3)

    rank_3_tensor = tf.constant([
        [[0, 1, 2, 3, 4],
         [5, 6, 7, 8, 9]],
        [[10, 11, 12, 13, 14],
         [15, 16, 17, 18, 19]],
        [[20, 21, 22, 23, 24],
         [25, 26, 27, 28, 29]], ])
    print(tf.reshape(rank_3_tensor, [-1]))  # check row major in memory
    print(tf.reshape(rank_3_tensor, [3 * 2, 5]).shape)  # (6, 5)
    print(tf.reshape(rank_3_tensor, [3, -1]).shape)  # (3, 10)
