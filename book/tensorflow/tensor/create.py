import tensorflow as tf
import numpy as np


def math_demo():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[1, 1], [1, 1]])

    print(tf.add(a, b), "\n")
    print(tf.multiply(a, b), "\n")
    print(tf.matmul(a, b), "\n")
    print(a + b, "\n")  # element-wise addition
    print(a * b, "\n")  # element-wise multiplication
    print(a @ b, "\n")  # matrix multiplication

    c = tf.constant([1., 2., 3.])
    print(tf.square(5))
    print(tf.reduce_sum(c))
    print(tf.reduce_max(c))  # 3 the largest value
    print(tf.argmax(c))  # 2 the index of the largest value
    print(tf.nn.softmax(c))  # [0.09003057 0.24472848 0.66524094]


if __name__ == '__main__':
    rank_0_tensor = tf.constant(4)
    print(rank_0_tensor)  # tf.Tensor(4, shape=(), dtype=int32)
    rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
    print(rank_1_tensor)  # tf.Tensor([2. 3. 4.], shape=(3,), dtype=float32)
    rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
    print(rank_2_tensor)  # tf.Tensor([...], shape=(3, 2), dtype=float16)

    rank_4_tensor = tf.zeros([3, 2, 4, 5])  # (batch, width, height, features)
    print("Type of every element:", rank_4_tensor.dtype)  # float32
    print("Number of axes:", rank_4_tensor.ndim)  # 4
    print("Shape of tensor:", rank_4_tensor.shape)  # (3, 2, 4, 5)
    print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])  # 3
    print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])  # 5
    print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())  # 120

    math_demo()
