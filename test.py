#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
input_data = tf.Variable( [[[1,2],[3,4],[5,6]],[[6,5],[4,3],[2,1]]], dtype = tf.float32 )
output = tf.nn.l2_normalize(input_data, dim = 3)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
sess.run(input_data)
sess.run(output)
print(sess.run(tf.shape(output)))