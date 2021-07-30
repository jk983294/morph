import tensorflow as tf
from datetime import datetime


class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")

    def __call__(self, x):
        return self.a_variable * x + self.non_trainable_variable


class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class SequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


def simple_module_demo():
    simple_module = SimpleModule(name="simple")
    print(simple_module(tf.constant(5.0)))
    print("trainable variables:", simple_module.trainable_variables)
    print("all variables:", simple_module.variables)


def sequential_module_demo():
    my_model = SequentialModule(name="the_model")
    print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))
    print("Submodules:", my_model.submodules)
    for var in my_model.variables:
        print(var, "\n")


class MyDense(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


class FlexibleDense(tf.keras.layers.Layer):
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.w = None
        self.b = None

    def build(self, input_shape):
        """build is called exactly once, and it is called with the shape of the input"""
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.out_features]), name='w')
        self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MySequentialModel(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = FlexibleDense(out_features=3)
        self.dense_2 = FlexibleDense(out_features=2)

    def call(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


def keras_model_demo():
    simple_layer = MyDense(name="simple", in_features=3, out_features=3)
    print("simple_layer results:", simple_layer(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))
    flexible_dense = FlexibleDense(out_features=3)
    print("flexible_dense results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))
    my_sequential_model = MySequentialModel(name="the_model")
    print("my_sequential_model results:", my_sequential_model(tf.constant([[2.0, 2.0, 2.0]])))


if __name__ == '__main__':
    simple_module_demo()
    sequential_module_demo()
    keras_model_demo()
