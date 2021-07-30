from datetime import datetime
import tensorflow as tf
from book.tensorflow.core.models import SequentialModule

if __name__ == '__main__':
    new_model = tf.saved_model.load("/tmp/the_saved_model")
    print(new_model([[2.0, 2.0, 2.0]]))
    print(new_model([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]))

    # new_model is an internal TensorFlow user object without any of the class knowledge.
    print(isinstance(new_model, SequentialModule))
