import tensorflow as tf
import os
import cProfile


def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy() + 1):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print('FizzBuzz')
        elif int(num % 3) == 0:
            print('Fizz')
        elif int(num % 5) == 0:
            print('Buzz')
        else:
            print(num.numpy())
        counter += 1


if __name__ == '__main__':
    print(tf.executing_eagerly())
    x = [[2.]]
    m = tf.matmul(x, x)
    print("hello, {}".format(m))

    fizzbuzz(5)
