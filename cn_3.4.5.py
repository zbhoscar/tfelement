import tensorflow as tf
# import matplotlib.pyplot as plt
#import numpy as np
from numpy.random import RandomState

batch_size=8
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
x  = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

corss_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(corss_entropy)

rdm = RandomState(1)
dataset_size=128
X=rdm.rand(dataset_size,2)
Y=[[int(x1+x2<1)] for (x1,x2) in X ]

with tf.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)

	print(sess.run(w1))
	print(sess.run(w2))

	STEPS=150
	for i in range(STEPS):
		start=(i*batch_size) % dataset_size
		end= min(start+batch_size,dataset_size)

		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

		if i % 1000 == 0:
			total_cross_entropy = sess.run(corss_entropy,feed_dict={x:X,y_:Y})
			print("%d steps,corss_entropy on all data is %g" % (i,total_cross_entropy))

	print(sess.run(w1))
	print(sess.run(w2))

	# print(sess.run(y,feed_dict={x:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
# print(a.graph is tf.get_default_graph())

# g1 = tf.Graph()
# with g1.as_default():
# 	v = tf.get_variable("a",shape=[1],initializer=tf.zeros_initializer())

# g2 = tf.Graph()
# with g2.as_default():
# 	v = tf.get_variable("a",shape=[1],initializer=tf.ones_initializer())

# with tf.Session(graph=g1) as sess:
# 	tf.global_variables_initializer().run()
# 	with tf.variable_scope("",reuse=True):
# 		print(sess.run(tf.get_variable('a')))

# with tf.Session(graph=g2) as sess:
# 	tf.global_variables_initializer().run()
# 	with tf.variable_scope("",reuse=True):
# 		print(sess.run(tf.get_variable('a')))

# a1 = tf.get_variable(name='a1', shape=[2,3], initializer=tf.random_normal_initializer(mean=0, stddev=1))  
# a2 = tf.get_variable(name='a2', shape=[1], initializer=tf.constant_initializer(1))  
# a3 = tf.get_variable(name='a3', shape=[2,3], initializer=tf.ones_initializer())  

# # zbh: not done  
# with tf.Session() as sess:
# 	tf.global_variables_initializer().run()
# 	sess.run()
#     #sess.run(tf.initialize_all_variables())  
#     print(sess.run(a1))
#     print(sess.run(a2))
#     print(sess.run(a3))
