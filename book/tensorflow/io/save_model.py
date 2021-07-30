import tensorflow as tf
from book.tensorflow.core.models import SequentialModule

if __name__ == '__main__':
    my_model = SequentialModule(name="the_model")
    chkp_path = "/tmp/my_checkpoint"
    checkpoint = tf.train.Checkpoint(model=my_model)
    checkpoint.write(chkp_path)

    print(tf.train.list_variables(chkp_path))
