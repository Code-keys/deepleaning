# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:00:24 2019

@author: Administrator
"""


#REGRESSION MODEL
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    

#SET VAR
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#model
y = tf.nn.softmax(tf.matmul(x,W) + b)

#training
y_ = tf.placeholder("float", [None,10]) #CROSS IE

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
#EVAULATE
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

print( sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
