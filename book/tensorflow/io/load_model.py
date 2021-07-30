import tensorflow as tf
from book.tensorflow.core.models import SequentialModule

if __name__ == '__main__':
    my_model = SequentialModule()
    checkpoint = tf.train.Checkpoint(model=my_model)
    chkp_path = "/tmp/my_checkpoint"
    checkpoint.restore(chkp_path)

    print(my_model(tf.constant([[2.0, 2.0, 2.0]])))
