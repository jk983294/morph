import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    ndarray = np.random.rand(4, 3)
    print("TensorFlow operations convert numpy arrays to Tensors automatically")
    tensor = tf.multiply(ndarray, 42)
    print(tensor)

    print("And NumPy operations convert Tensors to numpy arrays automatically")
    print(np.add(tensor, 1))

    print("The .numpy() method explicitly converts a Tensor to a numpy array")
    print(tensor.numpy())
