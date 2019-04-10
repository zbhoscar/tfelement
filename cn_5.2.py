import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
BATCH_SIZE=100

LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99

REGULARIZATION_RATE=0.0001
TRAINING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
	if avg_class==None:
		#layer1=tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
		layer1=tf.matmul(input_tensor,weights1)+biases1
		return tf.matmul(layer1,weights2)+biases2
	else:
		#layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
		layer1=tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1)
		return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)

def train(mnist):
	x  = tf.placeholder(tf.float32,shape=(None,INPUT_NODE),name='x-input')
	y_ = tf.placeholder(tf.float32,shape=(None,OUTPUT_NODE),name='y-input')

	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=1))
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=1))
	biases1  = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
	biases2  = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

	y = inference(x,None,weights1,biases1,weights2,biases2)

	global_step = tf.Variable(0,trainable=False)
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	variable_averages_op=variable_averages.apply(tf.trainable_variables())
	average_y=inference(x,variable_averages,weights1,biases1,weights2,biases2)

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	regularization=regularizer(weights1)+regularizer(weights2)
	loss=cross_entropy_mean+regularization

	learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
	train_step=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,global_step=global_step)
	# train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

	with tf.control_dependencies([train_step,variable_averages_op]):
		train_op=tf.no_op(name='train')

	correct_prediction=tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	with tf.Session() as sess:
		init_op=tf.global_variables_initializer()
		sess.run(init_op)

		validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
		test_feed={x:mnist.test.images,y_:mnist.test.labels}

		for i in range(TRAINING_STEPS):
			xs,ys=mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op,feed_dict={x:xs,y_:ys})

			if i % 1000 == 0:
				validate_acc=sess.run(accuracy,feed_dict=validate_feed)
				print("%d steps,%g" % (i,validate_acc))

		test_acc=sess.run(accuracy,feed_dict=test_feed)
		print("final:%d steps,%g" % (i,test_acc))

def main(argv=None):
	mnist=input_data.read_data_sets('D:\Desktop\Deep_Learning_with_TensorFlow\datasets\MNIST_data',one_hot=True)
	train(mnist)

if __name__=='__main__':
	tf.app.run()


# def get_weight(shape, lamb):
# 	var=tf.Variable(tf.random_normal(shape),dtype = tf.float32)
# 	tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamb)(var))
# 	return var

# batch_size=8

# x  = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
# y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')
# layer_dimension=[2,10,10,10,1]
# n_layers=len(layer_dimension)

# cur_layer=x
# in_dimension=layer_dimension[0]

# for i in range(1,n_layers):
# 	out_dimension = layer_dimension[i]
# 	weight = get_weight([in_dimension,out_dimension],0.001)
# 	bias   = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
# 	cur_layer=tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
# 	in_dimension = layer_dimension[i]

# #a = tf.matmul(x,w1)
# y = tf.matmul(x,w1)

# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
# mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
# tf.add_to_collection('losses',mse_loss)
# loss = tf.add_n(tf.get_collection('losses'))

# train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# rdm = RandomState(1)
# dataset_size=128
# X=rdm.rand(dataset_size,2)
# Y=[[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X ]

# with tf.Session() as sess:
# 	init_op=tf.global_variables_initializer()
# 	sess.run(init_op)

# 	print(sess.run(w1))

# 	STEPS=15000
# 	for i in range(STEPS):
# 		start=(i*batch_size) % dataset_size
# 		end= min(start+batch_size,dataset_size)

# 		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})

# 		if i % 1000 == 0:
# 			total_cross_entropy = sess.run(loss,feed_dict={x:X,y_:Y})
# 			print("%d steps,cross_entropy on all data is %g" % (i,total_cross_entropy))

# 	print(sess.run(w1))

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

