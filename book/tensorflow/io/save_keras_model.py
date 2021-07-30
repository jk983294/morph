import tensorflow as tf
from book.tensorflow.core.models import MySequentialModel

if __name__ == '__main__':
    my_sequential_model = MySequentialModel(name="the_model")
    print("my_sequential_model results:", my_sequential_model(tf.constant([[2.0, 2.0, 2.0]])))
    my_sequential_model.save("exname_of_file")
