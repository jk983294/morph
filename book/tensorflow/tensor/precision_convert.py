import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
    the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)

    # Now, cast to an uint8 and lose the decimal precision
    the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
    print(the_u8_tensor)
