import tensorflow as tf

sess = tf.Session()
# ???????
saver = tf.train.import_meta_graph('/home/zbh/Dropbox/tf_models/180929175948/180929175948.ckpt-29400.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/zbh/Dropbox/tf_models/180929175948'))

# ??placeholders???????feed-dict???placeholders???
graph = tf.get_default_graph()
# graph.
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict = {w1: 13.0, w2: 17.0}

# ????????????op
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

# ?????????op
add_on_op = tf.multiply(op_to_restore, 2)

print(sess.run(add_on_op, feed_dict))
# ??120.0==>(13+17)*2*2
