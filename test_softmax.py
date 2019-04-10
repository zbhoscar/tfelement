import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
with tf.Session(config=config) as sess:
    _logits = [[0.3, 0.2, 0.2, 0.1], [0.5, 0.7, 0.3, 0.2], [0.1, 0.2, 0.3, 0.2], [0., 0., 0., 1.]]
    _labels = [0, 1, 2, 0]
    # Softmax本身的算法很简单，就是把所有值用e的n次方计算出来，求和后算每个值占的比率，保证总和为1，一般我们可以认为Softmax出来的就是confidence也就是概率
    # [[0.35591307  0.32204348  0.32204348]
    #  [0.32893291  0.40175956  0.26930749]
    #  [0.30060959  0.33222499  0.36716539]]
    # print(sess.run(tf.nn.softmax(_logits)))
    # 对 _logits 进行降维处理，返回每一维的合计
    # [1.  1.  0.99999994]
    # print(sess.run(tf.reduce_sum(tf.nn.softmax(_logits), 1)))

    # 传入的 lables 需要先进行 独热编码 处理。
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=_logits, labels=tf.one_hot(_labels,depth=len(_labels)))
    los2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=_labels)
    # [ 1.03306878  0.91190147  1.00194287]
    print(sess.run(loss))
    print(sess.run(los2))

    _labels = [0, 1, 2, 3]
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=_logits, labels=tf.one_hot(_labels,depth=len(_labels)))
    los2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=_labels)
    # [ 1.03306878  0.91190147  1.00194287]
    print(sess.run(loss))
    print(sess.run(los2))