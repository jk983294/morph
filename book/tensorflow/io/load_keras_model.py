import tensorflow as tf
from book.tensorflow.core.models import MySequentialModel

if __name__ == '__main__':
    reconstructed_model = tf.keras.models.load_model("exname_of_file")
    print("my_sequential_model results:", reconstructed_model(tf.constant([[2.0, 2.0, 2.0]])))
