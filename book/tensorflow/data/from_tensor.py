import tensorflow as tf

if __name__ == '__main__':
    ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
    ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

    print('Elements of ds_tensors:')
    for x in ds_tensors:
        print(x)
