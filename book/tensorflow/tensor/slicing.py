import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    t1 = tf.constant([0, 1, 2, 3, 4, 5, 6, 7])
    print(tf.slice(t1, begin=[1], size=[3]))  # [1 2 3]
    print(t1[1:4])  # [1 2 3]
    print(t1[-3:])  # [5 6 7]

    t2 = tf.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])
    print(t2[:-1, 1:3])

    t3 = tf.constant([[[1, 3, 5, 7], [9, 11, 13, 15]],
                      [[17, 19, 21, 23], [25, 27, 29, 31]]])
    print(tf.slice(t3, begin=[1, 1, 0], size=[1, 1, 2]))

    # strided
    print(tf.gather(t1, indices=[0, 3, 6]))
    print(t1[::3])

    t4 = tf.constant([[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]])
    print(tf.gather_nd(t4, indices=[[2], [3], [0]]))  # by row

    t5 = np.reshape(np.arange(18), [2, 3, 3])
    print(tf.gather_nd(t5, indices=[[0, 0, 0], [1, 2, 1]]))  # two elements
    print(tf.gather_nd(t5, indices=[[[0, 0], [0, 2]], [[1, 0], [1, 2]]]))  # Return a list of two matrices
    print(tf.gather_nd(t5, indices=[[0, 0], [0, 2], [1, 0], [1, 2]]))  # Return one matrix
