import tensorflow as tf
import numpy as np
import tempfile


def from_tensor_demo():
    ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
    ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

    print('Elements of ds_tensors:')
    for x in ds_tensors:
        print(x)


def from_file_demo():
    # Create a CSV file
    _, filename = tempfile.mkstemp()

    with open(filename, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3")

    ds_file = tf.data.TextLineDataset(filename)
    ds_file = ds_file.batch(2)

    print('\nElements in ds_file:')
    for x in ds_file:
        print(x)


if __name__ == '__main__':
    from_tensor_demo()
    from_file_demo()
