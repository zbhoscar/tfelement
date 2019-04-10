#!/usr/bin/env python  
# coding=utf-8  
  
import tensorflow as tf  
logits=tf.Variable(tf.truncated_normal([10,5],mean=0.0,stddev=1.0,dtype=tf.float32))  
labels=tf.Variable([1,1,1,1,1,1,1,1,1,1])  
eval_correct=tf.nn.in_top_k(logits,labels,3)  
  
init=tf.initialize_all_variables()  
sess=tf.Session()  
sess.run(init)  
print(sess.run(logits))
print(sess.run(labels)) 
print(sess.run(eval_correct))
sess.close()  