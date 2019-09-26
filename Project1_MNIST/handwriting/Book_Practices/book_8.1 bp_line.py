#!/usr/bin/python
#*-.UTF-8.-*#
'''__author:我的第一个神经网络程序  算法：线性回归算法'''
import tensorflow as tf
import numpy as  np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def add_layer(inputs,in_size,out_size,activation_function = None):
    weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,weights)+ biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,3000)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

h1 = add_layer(xs,1,20,activation_function=tf.nn.relu)
prediction = add_layer(h1,20,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
i = 1
while 1:
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    i+=1
    if i%100  ==0:
        print('已经训练了 %d 次，误差loss为 %f '   %(i,sess.run(loss,feed_dict={xs:x_data,ys:y_data})))
    if i%100000  ==1:
        break