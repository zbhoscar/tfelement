# -*- coding: UTF-8 -*- #
# python3 __file__ -set_gpu=0 -timestamp=`date '+%y%m%d%H%M%S'` | tee ~/Dropbox/droplog/`date '+%y%m%d%H%M%S'`.txt
import z_flags
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# FOR PLACEHOLDER get_input
tf.app.flags.DEFINE_integer('sample_num', 6, 'numbers to sample in a video clip')
tf.app.flags.DEFINE_string('shuffle_string', '0,0,1', 'stream num of siamese net')
tf.app.flags.DEFINE_bool('constrain', False, 'sample frames by constrain?')
tf.app.flags.DEFINE_bool('consecutive', False, 'sample frames by consecutive?')
tf.app.flags.DEFINE_string('encoder', 'dynamic', 'how to encode the sampled frames')
tf.app.flags.DEFINE_string('specific_size', 'random_crop', 'how to get specific size of net input')
tf.app.flags.DEFINE_list('aaa',['te','te','te'],'')

print(('* ' + 'FLAGS' + ' *').center(60, '*'))
_ = [print('-', i.ljust(20),' : ',eval('FLAGS.' + i)) for i in dir(FLAGS)]