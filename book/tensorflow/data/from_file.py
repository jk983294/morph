import tensorflow as tf
import tempfile

if __name__ == '__main__':
    # Create a CSV file
    _, filename = tempfile.mkstemp()

    with open(filename, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3")

    ds_file = tf.data.TextLineDataset(filename)
    ds_file = ds_file.batch(2)

    print('\nElements in ds_file:')
    for x in ds_file:
        print(x)
