import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import time
import os

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_string('ckpt_file', r'D:\Desktop\Deep_Learning_with_TensorFlow\models\vgg_16_2016_08_28\vgg_16.ckpt',
                           """Path to the ckpt_file.""")
tf.app.flags.DEFINE_boolean('show_details', False,
                            """If show var in detail.""")


def main(_):
    checkpoint_path = FLAGS.ckpt_file  
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:  
        print("tensor_name: ", key, reader.get_tensor(key).shape)
        if FLAGS.show_details == True:  
            print(reader.get_tensor(key).shape) 


if __name__ == '__main__':
    tf.app.run()
    