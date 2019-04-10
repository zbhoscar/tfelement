import tensorflow as tf 
config = tf.ConfigProto()  

# config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
config.log_device_placement = True
# allow_soft_placement=True,log_device_placement=True   
sess=tf.Session(config = config)
with tf.device('/cpu:0'):
	a = tf.constant([1.,2.,3.],shape=[3],name='a')
	b = tf.constant([1.,2.,3.],shape=[3],name='b')
with tf.device('/gpu:0'):
	c = a+b

print(sess.run(c))
sess.close()
