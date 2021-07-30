import tensorflow as tf
import os
import cProfile


def variable_turn_off_gradient():
    step_counter = tf.Variable(1, trainable=False)
    print(step_counter)


def variable_placing():
    with tf.device('CPU:0'):
        # Create some tensors
        a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    print(c)


if __name__ == '__main__':
    """
    recommended way to represent shared, persistent state your program manipulates
    Higher level libraries like tf.keras use tf.Variable to store model parameters
    A variable looks and acts like a tensor, and, in fact, is a data structure backed by a tf.Tensor
    """
    my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    my_variable = tf.Variable(my_tensor)
    print("Shape: ", my_variable.shape)
    print("DType: ", my_variable.dtype)
    print("As NumPy: ", my_variable.numpy())
    print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
    print("\nIndex of highest value:", tf.argmax(my_variable))

    # This creates a new tensor; it does not reshape the variable.
    print("\nCopying and reshaping: ", tf.reshape(my_variable, [1, 4]))

    # Variables can be all kinds of types, just like tensors
    bool_variable = tf.Variable([False, False, False, True])
    complex_variable = tf.Variable([5 + 4j, 6 + 1j])

    # assign
    a = tf.Variable([2.0, 3.0])
    a.assign([1, 2])  # This will keep the same dtype, float32
    # a.assign([1.0, 2.0, 3.0])  # Not allowed as it resizes the variable

    # copy, two variables will not share the same memory
    a = tf.Variable([2.0, 3.0])
    b = tf.Variable(a)  # Create b based on the value of a
    a.assign([5, 6])
    print(a.numpy())  # [5. 6.]
    print(b.numpy())  # [2. 3.]
    print(a.assign_add([2, 3]).numpy())  # [7. 9.]
    print(a.assign_sub([7, 9]).numpy())  # [0. 0.]

    variable_turn_off_gradient()
    variable_placing()
