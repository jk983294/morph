import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    t2 = tf.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])
    t6 = tf.constant([10])
    indices = tf.constant([[1], [3], [5], [7], [9]])
    data = tf.constant([2, 4, 6, 8, 10])
    """the tensor into which you insert values is zero-initialized"""
    print(tf.scatter_nd(indices=indices, updates=data, shape=t6))

    # Gather values from one tensor by specifying indices
    new_indices = tf.constant([[0, 2], [2, 1], [3, 3]])
    t7 = tf.gather_nd(t2, indices=new_indices)

    # Add these values into a new tensor
    t8 = tf.scatter_nd(indices=new_indices, updates=t7, shape=tf.constant([4, 5]))
    print(t8)

    """insert data into a tensor with pre-existing values"""
    t11 = tf.constant([[2, 7, 1], [9, 1, 1], [1, 3, 8]])
    t12 = tf.tensor_scatter_nd_add(t11, indices=[[0, 2], [1, 1], [2, 0]], updates=[6, 5, 4])
    print(t12)
    t13 = tf.tensor_scatter_nd_sub(t12, indices=[[0, 2], [1, 1], [2, 0]], updates=[6, 5, 4])
    print(t13)
