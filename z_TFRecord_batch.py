# -*- coding:utf-8 -*- 

import tensorflow as tf

files = tf.train.match_filenames_once(r'D:\Code_win\Deep_Learning_with_TensorFlow\1.0.0\Chapter07\Records\output.tfrecords')
filename_queue = tf.train.string_input_producer(files, shuffle=False) 

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
	serialized_example,
	features={
		'image_raw':tf.FixedLenFeature([],tf.string),
		'pixels':tf.FixedLenFeature([],tf.int64),
		'label':tf.FixedLenFeature([],tf.int64)
	})

decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
pixels = tf.cast(features['pixels'],tf.int32)
images = tf.reshape(retyped_images, [20])
labels = tf.cast(features['label'],tf.int32)

print(decoded_images)
print(retyped_images)
print(pixels)
print(images)
print(labels)

min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch([images, labels], 
    batch_size=batch_size,capacity=capacity,num_threads=12,
    min_after_dequeue=min_after_dequeue)

def inference(input_tensor, weights1, biases1, weights2, biases2):
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 5000        

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch, weights1, biases1, weights2, biases2)
    
# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
# 损失函数的计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularaztion

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = (tf.global_variables_initializer(), tf.local_variables_initializer())
# 初始化回话并开始训练过程。
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 循环的训练神经网络。
    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            print("After %d training step(s), loss is %g " % (i, sess.run(loss)))
                  
        sess.run(train_step) 
    coord.request_stop()
    coord.join(threads)       
# q=tf.RandomShuffleQueue(4,min_after_dequeue=2,dtypes='int32')
# init=q.enqueue_many(([0,10,5,3],))
# x=q.dequeue()
# y=x+1
# q_inc=q.enqueue([y])

# with tf.Session() as sess:
# 	init.run()
# 	for _ in range(20):
# 		v,_=sess.run([x,q_inc])
# 		print(v)
# 		
# image_raw_data=tf.gfile.FastGFile(r'D:\Code_win\Deep_Learning_with_TensorFlow\datasets\cat.jpg','rb').read()
# height=300
# width=300

# def  distort_color(image,color_ordering=0):
# 	if color_ordering==0:
# 		image=tf.image.random_brightness(image,32./255.)
# 		image=tf.image.random_saturation(image,0.5,1.5)
# 		image=tf.image.random_hue(image,0.2)
# 		iamge=tf.image.random_contrast(image,0.5,1.5)
# 	else:
# 		image=tf.image.random_saturation(image,0.5,1.5)
# 		image=tf.image.random_brightness(image,32./255.)
# 		image=tf.image.random_hue(image,0.2)
# 		iamge=tf.image.random_contrast(image,0.5,1.5)
# 	return tf.clip_by_value(image,0.0,1.0)

# def preprosess_for_main(image,height,width,bbox):
# 	if bbox == None:
# 		bbox=tf.constant([0.,0.,1.,1.],dtype=tf.float32,shape=[1,1,4])
# 	if image.dtype != tf.float32:
# 		image=tf.image.convert_image_dtype(image,tf.float32)
# 	bbox_begin,bbox_size,_=tf.image.sample_distorted_bounding_box(
# 		tf.shape(image),bbox)
# 	distort_image=tf.slice(image,bbox_begin,bbox_size)
# 	distort_image=tf.image.resize_images(distort_image,[height,width],np.random.randint(4))
# 	distort_image=tf.image.random_flip_left_right(distort_image)
# 	distort_image=distort_color(distort_image,np.random.randint(2))
# 	return distort_image

# with tf.Session() as sess:
# 	img_data=tf.image.decode_jpeg(image_raw_data)
# 	boxes=tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
# 	for i in range(6):
# 		result=preprosess_for_main(img_data,height,width,boxes)
# 		plt.imshow(result.eval())
# 		plt.show()

# with tf.Session() as sess:
# 	img_data=tf.image.decode_jpeg(image_raw_data)
# 	print(img_data.eval()[0])
# 	plt.imshow(img_data.eval())

# 	resized = tf.image.resize_images(img_data, [300, 300], method=0)

# 	# TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
# 	print("Digital type: ", resized.dtype)

# 	cat = np.asarray(resized.eval(), dtype='uint8')
	
# 	cat3 =  tf.image.convert_image_dtype(resized, dtype=tf.uint8)
# 	print(cat[0])
# 	print(cat3.eval()[0])

# 	# tf.image.convert_image_dtype(rgb_image, tf.float32)
# 	plt.imshow(cat)
# 	plt.show()
# 	plt.imshow(cat3.eval())


# reader=tf.TFRecordReader()
# filename_squeue=tf.train.string_input_producer(
# 	[r'D:\Code_win\tf_test\temp\output.tfrecords'])

# _,serialized_example=reader.read(filename_squeue)

# features=tf.parse_single_example(
# 		serialized_example,
# 		features={
# 		'image_raw':tf.FixedLenFeature([],tf.string),
# 		'pixels':tf.FixedLenFeature([],tf.int64),
# 		'label':tf.FixedLenFeature([],tf.int64)
# 	})

# images=tf.decode_raw(features['image_raw'],tf.uint8)
# labels=tf.cast(features['label'],tf.int32)
# pixels=tf.cast(features['pixels'],tf.int32)

# sess = tf.Session()
# coord=tf.train.Coordinator()
# threads=tf.train.start_queue_runners(sess=sess,coord=coord)
# for i in range(2):
# 	ort=sess.run(serialized_example)
# 	test=sess.run(features)
# 	image,label,pixel=sess.run([images,labels,pixels])
# 	print(ort)
# 	print(test)
# 	print(image,label,pixel)
# def _int64_feature(value):
# 	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def _bytes_feature(value):
# 	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# mnist=input_data.read_data_sets(
# 	r'D:\Code_win\Deep_Learning_with_TensorFlow\datasets\MNIST_data',
# 	dtype=tf.uint8,one_hot=True)
# images=mnist.train.images
# labels=mnist.train.labels
# pixels=images.shape[1]
# num_examples=images.shape[0]

# filename=r'D:\Code_win\tf_test\temp\output.tfrecords'
# writer=tf.python_io.TFRecordWriter(filename)
# for index in range(num_examples):
# 	image_raw=images[index].tostring()

# 	example = tf.train.Example(features=tf.train.Features(feature={
# 			'pixels': _int64_feature(pixels),
# 			'label': _int64_feature(np.argmax(labels[index])),
# 			'image_raw': _bytes_feature(image_raw)
# 	}))
# 	writer.write(example.SerializeToString())
# writer.close()
# print('Done')

# BOTTLENECK_TENSOR_SIZE=2048
# BOTTLENECK_TENSOR_NAME='pool_3/_reshape:0'
# JPEG_DATA_TENSOR_NAME='DecodeJpeg/contents:0'

# MODEL_DIR=r'D:\Code_win\Deep_Learning_with_TensorFlow\datasets\inception_dec_2015'
# MODEL_FILE='tensorflow_inception_graph.pb'

# CACHE_DIR=r'D:\Code_win\tf_test\bottleneck'
# INPUT_DATA=r'D:\Code_win\Deep_Learning_with_TensorFlow\datasets\flower_photos'
# #INPUT_DATA=r'D:\Code_win\tf_test'

# VALIDATION_PCT=10
# TEST_PCT=10

# BATCH=100
# STEPS=4000
# LEARNING_RATE=0.01


# def create_image_lists(testing_percentage,validation_percentage):
# 	result={}
# 	sub_dirs=[x[0] for x in os.walk(INPUT_DATA)][1:]
# 	for sub_dir in sub_dirs:
# 		extensions=['jpg', 'jpeg', 'JPG', 'JPEG','py']
# 		file_list=[]
# 		for extension in extensions:
# 			file_glob=os.path.join(sub_dir,'*.'+extension)
# 			file_list.extend(glob.glob(file_glob))
# 		if not file_list: continue

# 		#print(file_list)

# 		dir_name=os.path.basename(sub_dir)
# 		label_name=dir_name.lower()
# 		training_images	=[]
# 		testing_images=[]
# 		validation_images=[]

# 		for file_name in file_list:
# 			base_name = os.path.basename(file_name)

# 			chance = np.random.randint(100)
# 			if chance < validation_percentage:
# 				validation_images.append(base_name)
# 			elif chance < (testing_percentage + validation_percentage):
# 				testing_images.append(base_name)
# 			else:
# 				training_images.append(base_name)

# 		result[label_name]={
# 		'dir':dir_name,
# 		'training':training_images,
# 		'testing':testing_images,
# 		'validation':validation_images,
# 		}
# 	return result

# def get_image_path(image_lists,image_dir,label_name,index,category):
# 	label_lists=image_lists[label_name]
# 	category_list=label_lists[category]
# 	mod_index=index % len(category_list)
# 	base_name=category_list[mod_index]
# 	sub_dir=label_lists['dir']
# 	full_path=os.path.join(image_dir,sub_dir,base_name)
# 	return full_path

# def get_bottleneck_path(image_lists,label_name,index,category):
# 	return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'

# def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
# 	bottleneck_values=sess.run(bottleneck_tensor,{image_data_tensor:image_data})
# 	bottleneck_values=np.squeeze(bottleneck_values)
# 	return bottleneck_values

# def get_or_create_bottleneck(sess,image_lists,label_name,
# 			index,category,jpeg_data_tensor,bottleneck_tensor):
# 	label_lists=image_lists[label_name]
# 	sub_dir=label_lists['dir']
# 	sub_dir_path=os.path.join(CACHE_DIR,sub_dir)
# 	if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
# 	bottleneck_path=get_bottleneck_path(image_lists,label_name,index,category)

# 	if not os.path.exists(bottleneck_path):
# 		image_path=get_image_path(image_lists,INPUT_DATA,label_name,index,category)
# 		image_data=gfile.FastGFile(image_path,'rb').read()
# 		bottleneck_values=run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
# 		bottleneck_string=','.join(str(x) for x in bottleneck_values)
# 		with open(bottleneck_path,'w') as bottleneck_file:
# 			bottleneck_file.write(bottleneck_string)
# 	else:
# 		with open(bottleneck_path,'r') as bottleneck_file:
# 			bottleneck_string = bottleneck_file.read()
# 		bottleneck_values=[float(x) for x in bottleneck_string.split(',')]
# 	return bottleneck_values

# def get_random_cached_bottlenecks(sess, n_classes, image_lists, 
# 		how_many, category, jpeg_data_tensor, bottleneck_tensor):
# 	bottlenecks = []
# 	ground_truths = []
# 	for _ in range(how_many):
# 		label_index = random.randrange(n_classes)
# 		label_name = list(image_lists.keys())[label_index]
# 		image_index = random.randrange(65536)
# 		bottleneck = get_or_create_bottleneck(
# 		sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
# 		ground_truth = np.zeros(n_classes, dtype=np.float32)
# 		ground_truth[label_index] = 1.0
# 		bottlenecks.append(bottleneck)
# 		ground_truths.append(ground_truth)
# 	return bottlenecks, ground_truths

# def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
# 	bottlenecks = []
# 	ground_truths = []
# 	label_name_list = list(image_lists.keys())
# 	for label_index, label_name in enumerate(label_name_list):
# 		category = 'testing'
# 		for index, unused_base_name in enumerate(image_lists[label_name][category]):
# 			bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,jpeg_data_tensor, bottleneck_tensor)
# 			ground_truth = np.zeros(n_classes, dtype=np.float32)
# 			ground_truth[label_index] = 1.0
# 			bottlenecks.append(bottleneck)
# 			ground_truths.append(ground_truth)
# 	return bottlenecks, ground_truths

# def main():
# 	image_lists=create_image_lists(TEST_PCT,VALIDATION_PCT)
# 	n_classes=len(image_lists.keys())

# 	with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
# 		graph_def = tf.GraphDef()
# 		graph_def.ParseFromString(f.read())
# 	bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
# 		graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

# 	bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
# 	ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

# 	with tf.name_scope('final_training_ops'):
# 		weights=tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,n_classes],stddev=0.001))
# 		biases=tf.Variable(tf.zeros([n_classes]))
# 		logits=tf.matmul(bottleneck_input,weights)+biases
# 		final_tensor=tf.nn.softmax(logits)

# 	cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
# 	cross_entropy_mean=tf.reduce_mean(cross_entropy)
# 	train_step=tf.train.GradientDescentOptimizer(LEARNING_RATE)\
# 			.minimize(cross_entropy_mean)

# 	with tf.name_scope('evaluation'):
# 		correct_prediction=tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))
# 		evaluation_step=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 	with tf.Session() as sess:
# 		init_op=tf.global_variables_initializer()
# 		sess.run(init_op)

# 		for i in range(STEPS):
# 			train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(sess, 
# 				n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
# 			sess.run(train_step, feed_dict={
# 				bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

# 			if i%100 == 0 or i+1 == STEPS:
# 				validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
# 					sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
# 				validation_accuracy = sess.run(evaluation_step, feed_dict={
# 					bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
# 				print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
# 					(i, BATCH, validation_accuracy * 100))

# 		test_bottlenecks, test_ground_truth = get_test_bottlenecks(
# 			sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
# 		test_accuracy = sess.run(evaluation_step, feed_dict={
# 			bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
# 		print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

# if __name__=='__main__':	
# 	main()


# MODEL_SAVE_PATH=r'D:\Code_win\tf_test\models'
# MODEL_NAME='m64.ckpt'
# mnist_path=r'D:\Code_win\Deep_Learning_with_TensorFlow\datasets\MNIST_data'

# def train(mnist):
# 	x =tf.placeholder(tf.float32,[BATCH_SIZE,mnist_inference.IMAGE_SIZE,\
# 		mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS],name='x-input')
# 	y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')
# 	regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
# 	y=mnist_inference.inference(x,False,regularizer)
# 	global_step=tf.Variable(0,trainable=False)

# 	variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
# 	variable_averages_op=variable_averages.apply(tf.trainable_variables())
# 	cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(\
# 		logits=y, labels=tf.argmax(y_, 1))
# 	cross_entropy_mean=tf.reduce_mean(cross_entropy)
# 	loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
# 	learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,\
# 		mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
# 	train_step=tf.train.GradientDescentOptimizer(learning_rate)\
# 		.minimize(loss,global_step=global_step)
# 	with tf.control_dependencies([train_step,variable_averages_op]):
# 		train_op=tf.no_op(name='train')

# 	correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# 	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# 	saver=tf.train.Saver()
# 	init_op=tf.global_variables_initializer()

# 	with tf.Session() as sess:
# 		sess.run(init_op)

# 		for i in range(TRAINING_STEPS):
# 			xs,ys=mnist.train.next_batch(BATCH_SIZE)
# 			reshaped_xs=np.reshape(xs,(BATCH_SIZE,mnist_inference.IMAGE_SIZE,\
# 				mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))
# 			_, loss_value,validate_acc,step=sess.run([train_op,loss,accuracy,global_step],\
# 				feed_dict={x:reshaped_xs,y_:ys})

# 			if step % 1000 == 0:
# 				print('After %d steps,loss = %g,acc= %g' % (step,loss_value,validate_acc))
# 				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

# def main(argv=None):
# 	mnist=input_data.read_data_sets(mnist_path,one_hot=True)
# 	train(mnist)

# if __name__=='__main__':	
# 	main()


# INPUT_NODE=784
# OUTPUT_NODE=10
# LAYER1_NODE=500

# def get_weights_variables(shape,regularizer):
# 	weights=tf.get_variables('weights',shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
# 	if regularizer != None:
# 		tf.add_to_collection('losses',regularizer(weights))
# 	return weights

# def inference(input_tensor,regularizer):
# 	with tf.variable_scope('layer1'):
# 		weights=get_weights_variables([INPUT_NODE,LAYER1_NODE],regularizer)
# 		biases=tf.get_variables('biases',[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
# 		layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
# 	with tf.variable_scope('layer2'):
# 		weights=get_weights_variables([LAYER1_NODE,OUTPUT_NODE],regularizer)
# 		biases=tf.get_variables('biases',[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
# 		layer2=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
# 	return layer2


# v1=tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
# v2=tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
# result=v1+v2

# saver=tf.train.Saver()
# saver.export_meta_graph(r'D:\Code_win\tf_test\models\m54.ckpt.meta.json',as_text=True)



# init_op=tf.global_variables_initializer()
# with tf.Session() as sess:
# 	sess.run(init_op)

# 	graph_def=tf.get_default_graph().as_graph_def()
# 	output_graph_def=graph_util.convert_variables_to_constants(sess,graph_def,['add'])
# 	with tf.gfile.GFile(r'D:\Code_win\tf_test\models\test.pb','wb') as f:
# 		f.write(output_graph_def.SerializeToString())


# v1=tf.Variable(tf.constant(1.0,shape=[1],name='v1'))
# v2=tf.Variable(tf.constant(2.0,shape=[1],name='v2'))
# result=v1+v2
# saver=tf.train.Saver()
# with tf.Session() as sess:
# 	init_op=tf.global_variables_initializer()
# 	sess.run(init_op)
# 	saver.save(sess,r'D:\Code_win\tf_test\models\m54.ckpt')

# v1=tf.Variable(tf.constant(2.0,shape=[1],name='v1'))
# v2=tf.Variable(tf.constant(2.0,shape=[1],name='v2'))
# result=v1+v2	
# saver=tf.train.Saver([v1])
# with tf.Session() as sess:
# 	saver.restore(sess,r'D:\Code_win\tf_test\models\m54.ckpt')
# 	print(sess.run(result))

# saver=tf.train.import_meta_graph(r'D:\Code_win\tf_test\models\m54.ckpt.meta')
# with tf.Session() as sess:
# 	saver.restore(sess,r'D:\Code_win\tf_test\models\m54.ckpt')
# 	print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))

# v1=tf.Variable(tf.constant(2.0,shape=[1]),name='v1')
# v2=tf.Variable(tf.constant(1.0,shape=[1]),name='v2')
# for variable in tf.global_variables():
# 	print(variable.name)
# ema=tf.train.ExponentialMovingAverage(0.99)
# maintain_averages_op=ema.apply(tf.global_variables())
# for variable in tf.global_variables():
# 	print(variable.name)

# saver=tf.train.Saver()
# with tf.Session() as sess:
# 	init_op=tf.global_variables_initializer()
# 	sess.run(init_op)
# 	sess.run(tf.assign(v1,[10]))
# 	sess.run(maintain_averages_op)
# 	saver.save(sess,r'D:\Code_win\tf_test\models\m54_1.ckpt')
# 	print(sess.run([v1,v2,ema.average(v1),maintain_averages_op,init_op]))

# v3=tf.Variable(tf.constant(2.0,shape=[1],name='v1'))
# saver=tf.train.Saver({'v1/ExponentialMovingAverage':v3})
# with tf.Session() as sess:
# 	saver.restore(sess,r'D:\Code_win\tf_test\models\m54_1.ckpt')
# 	print(sess.run(v3))

##################################################
# INPUT_NODE=784
# OUTPUT_NODE=10
# LAYER1_NODE=500
# BATCH_SIZE=100

# LEARNING_RATE_BASE=0.8
# LEARNING_RATE_DECAY=0.99

# REGULARIZATION_RATE=0.0001
# TRAINING_STEPS=40000
# MOVING_AVERAGE_DECAY=0.99

# def inference(input_tensor,reuse=False):
# 	with tf.variable_scope('layer1',reuse=reuse):
# 		weights = tf.get_variable('weights',[INPUT_NODE,LAYER1_NODE],\
# 			initializer=tf.truncated_normal_initializer(0.1))
# 		biases  = tf.get_variable('baises',[LAYER1_NODE],\
# 			initializer=tf.constant_initializer(0.0))
# 		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
# 		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(weights))
# 	with tf.variable_scope('layer2',reuse=reuse):
# 		weights = tf.get_variable('weights',[LAYER1_NODE,OUTPUT_NODE],\
# 			initializer=tf.truncated_normal_initializer(0.1))
# 		biases  = tf.get_variable('baises',[OUTPUT_NODE],\
# 			initializer=tf.constant_initializer(0.0))
# 		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(weights))
# 		layer2 = tf.nn.relu(tf.matmul(layer1,weights)+biases)

# 	return layer2

# def train(mnist):
# 	x  = tf.placeholder(tf.float32,shape=(None,INPUT_NODE),name='x-input')
# 	y_ = tf.placeholder(tf.float32,shape=(None,OUTPUT_NODE),name='y-input')

# 	y = inference(x)

# 	global_step = tf.Variable(0,trainable=False)
# 	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
# 	variable_averages_op=variable_averages.apply(tf.trainable_variables())
# 	#average_y=inference(x)

# 	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
# 	cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 	tf.add_to_collection('losses',cross_entropy_mean)
# 	loss = tf.add_n(tf.get_collection('losses'))

# 	learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
# 	#train_step=tf.train.AdamOptimizer(0.001).minimize(loss,global_step=global_step)
# 	train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

# 	with tf.control_dependencies([train_step,variable_averages_op]):
# 		train_op=tf.no_op(name='train')

# 	correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# 	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 	with tf.Session() as sess:
# 		init_op=tf.global_variables_initializer()
# 		sess.run(init_op)

# 		validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
# 		test_feed={x:mnist.test.images,y_:mnist.test.labels}

# 		for i in range(TRAINING_STEPS):
# 			xs,ys=mnist.train.next_batch(BATCH_SIZE)
# 			sess.run(train_op,feed_dict={x:xs,y_:ys})

# 			if i % 1000 == 0:
# 				validate_acc=sess.run(accuracy,feed_dict=validate_feed)
# 				print("%d steps,%g" % (i,validate_acc))

# 		test_acc=sess.run(accuracy,feed_dict=test_feed)
# 		print("final:%d steps,%g" % (i,test_acc))

# def main(argv=None):
# 	mnist=input_data.read_data_sets('D:\Desktop\Deep_Learning_with_TensorFlow\datasets\MNIST_data',one_hot=True)
# 	train(mnist)

# if __name__=='__main__':
# 	tf.app.run()
# 	
######################################################

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

