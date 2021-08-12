mport tensorflow as tf

if __name__ == '__main__':
    print(tf.sysconfig.get_include())
    print(tf.sysconfig.get_lib())
    print(tf.sysconfig.get_compile_flags())
    print(tf.sysconfig.get_link_flags())
