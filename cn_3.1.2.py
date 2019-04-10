import tensorflow as tf
#import matplotlib.pyplot as plt
#import numpy as np
def main():
    a = tf.constant([1.0,2.0],name='a')
    b = tf.constant([2.0,3.0],name='b')
    result=a+b

    print(a.graph is tf.get_default_graph())

    g1 = tf.Graph()
    with g1.as_default():
        v = tf.get_variable("a",shape=[1],initializer=tf.zeros_initializer())

    g2 = tf.Graph()
    with g2.as_default():
        v = tf.get_variable("a",shape=[1],initializer=tf.ones_initializer())

    with tf.Session(graph=g1) as sess:
        tf.global_variables_initializer().run()
        with tf.variable_scope("",reuse=True):
            print(sess.run(tf.get_variable('a')))

    with tf.Session(graph=g2) as sess:
        tf.global_variables_initializer().run()
        with tf.variable_scope("",reuse=True):
            print(sess.run(tf.get_variable('a')))

    a1 = tf.get_variable(name='a1', shape=[2,3], initializer=tf.random_normal_initializer(mean=0, stddev=1))
    a2 = tf.get_variable(name='a2', shape=[1], initializer=tf.constant_initializer(1))
    a3 = tf.get_variable(name='a3', shape=[2,3], initializer=tf.ones_initializer())

    # zbh: not done
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #sess.run()
        #sess.run(tf.initialize_all_variables())
        print(sess.run(a1))
        print(sess.run(a2))
        print(sess.run(a3))

if __name__ == '__main__':
    main()