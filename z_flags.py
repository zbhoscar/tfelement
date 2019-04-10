# ###### TESTING ######
# import tensorflow as tf

# test = tf.app.flags
# ttt = test.FLAGS

# # Basic model parameters.
# test.DEFINE_integer('batch_size', 128,
#                             """Number of images to process in a batch.""")
# test.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                            """Path to the CIFAR-10 data directory.""")
# test.DEFINE_boolean('use_fp16', False,
#                             """Train the model using fp16.""")
# # tf.app.flags.DEFINE_string('str_name', 'cifar10_data',"descrip1")
# # tf.app.flags.DEFINE_integer('int_name', 128,"descript2")
# # tf.app.flags.DEFINE_boolean('bool_name', True, "descript3")
# print(ttt.batch_size)
# print(ttt.data_dir)
# print(ttt.use_fp16)
# # print(FLAGS.str_name)
# # print(FLAGS.int_name)
# # print(FLAGS.bool_name)
# print('###########')


# def flags_editor(test):    
#     test.DEFINE_string('str_name', 'cifar10_data',"descrip1")
#     test.DEFINE_integer('int_name', 128,"descript2")
#     test.DEFINE_boolean('bool_name', True, "descript3")
#     # return tf.app.flags.FLAGS

# def main(_):
#     # FLAGS1 = flags_editor(FLAGS)
#     # print(FLAGS1.batch_size)
#     # print(FLAGS1.data_dir)
#     # print(FLAGS1.use_fp16)
#     # print(FLAGS1.str_name)
#     # print(FLAGS1.int_name)
#     # print(FLAGS1.bool_name)
#     flags_editor(test)
#     print(ttt.batch_size)
#     print(ttt.data_dir)
#     print(ttt.use_fp16)
#     print(ttt.str_name)
#     print(ttt.int_name)
#     print(ttt.bool_name)

# if __name__ == '__main__':
#     # main()
#     tf.app.run()

##### TESTED ######
import tensorflow as tf
import os 

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
tf.app.flags.DEFINE_string('str_name', 'cifar10_data',"descrip1")
tf.app.flags.DEFINE_integer('int_name', 128,"descript2")
tf.app.flags.DEFINE_boolean('bool_name', True, "descript3")
tf.app.flags.DEFINE_string('data_dir2', os.path.dirname(FLAGS.data_dir),
                           """Path to the CIFAR-10 data directory.""")
print(FLAGS.batch_size)
print(FLAGS.data_dir)
print(FLAGS.use_fp16)
print(FLAGS.str_name)
print(FLAGS.int_name)
print(FLAGS.bool_name)
print(FLAGS.data_dir2)
print('###########')


def flags_editor(FLAGS):    
    FLAGS.str_name = 'def_v_1'
    FLAGS.int_name = 10
    FLAGS.bool_name = False
    return FLAGS

def main(_):
    flags = flags_editor(FLAGS)     
    print(flags.batch_size)
    print(flags.data_dir)
    print(flags.use_fp16)
    print(flags.str_name)
    print(flags.int_name)
    print(flags.bool_name)

if __name__ == '__main__':
    # main()
    tf.app.run()
