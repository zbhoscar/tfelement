import tensorflow as tf;  
import numpy as np;  
import matplotlib.pyplot as plt;  
import math
  
learning_rate = 1.  
decay_rate1 = 0.99  
decay_rate2 = 0.9
global_steps = 140000
#decay_steps = 60000/128  
decay_steps1 = global_steps/math.log(0.005,decay_rate1)
decay_steps2 = global_steps/math.log(0.005,decay_rate2)
  
global_ = tf.Variable(tf.constant(0))  
# c = tf.train.exponential_decay(learning_rate, global_, decay_steps1, decay_rate1, staircase=True)  
BATCH_SIZE = 128
num_batcher_per_epoch = 50000 / BATCH_SIZE
decay_step = int(num_batcher_per_epoch * 350.0)
c = tf.train.exponential_decay(0.1, global_, 100, 0.96)
d = tf.train.exponential_decay(learning_rate, global_, decay_steps2, decay_rate2)#, staircase=False)  
  
T_C = []  
F_D = []  
  
with tf.Session() as sess:  
    for i in range(global_steps):  
        T_c = sess.run(c,feed_dict={global_: i})  
        T_C.append(T_c)  
        F_d = sess.run(d,feed_dict={global_: i})  
        F_D.append(F_d)  
  
  
plt.figure(1)  
# plt.plot(range(global_steps), F_D, 'r-')  
plt.plot(range(global_steps), T_C, 'b-')  
      
plt.show()  