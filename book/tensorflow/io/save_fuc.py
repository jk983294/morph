from datetime import datetime

import tensorflow as tf
from book.tensorflow.core.models import Dense


class MySequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    @tf.function
    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


if __name__ == '__main__':
    my_model = MySequentialModule(name="the_model")
    print(my_model([[2.0, 2.0, 2.0]]))
    print(my_model([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]))
    tf.saved_model.save(my_model, "/tmp/the_saved_model")

    # ls -l /tmp/the_saved_model
