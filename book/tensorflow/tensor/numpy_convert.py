import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    matrix1 = np.array([(2, 2, 2), (2, 2, 2), (2, 2, 2)], dtype='int32')
    matrix2 = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], dtype='int32')
    matrix_product_tensor = tf.matmul(matrix1, matrix2)
    print(matrix_product_tensor)

    ndarray = np.random.rand(4, 3)
    print("TensorFlow operations convert numpy arrays to Tensors automatically")
    tensor = tf.multiply(ndarray, 42)
    print(tensor)

    print("And NumPy operations convert Tensors to numpy arrays automatically")
    print(np.add(tensor, 1))

    print("The .numpy() method explicitly converts a Tensor to a numpy array")
    print(tensor.numpy())
    print(np.array(tensor))  # to numpy
