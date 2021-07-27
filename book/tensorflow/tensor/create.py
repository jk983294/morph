import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    matrix1 = np.array([(2, 2, 2), (2, 2, 2), (2, 2, 2)], dtype='int32')
    matrix2 = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], dtype='int32')
    matrix_product = tf.matmul(matrix1, matrix2)
    print(matrix_product)

    print(tf.add(1, 2))
    print(tf.add([1, 2], [3, 4]))
    print(tf.square(5))
    print(tf.reduce_sum([1, 2, 3]))
    print(tf.square(2) + tf.square(3))
