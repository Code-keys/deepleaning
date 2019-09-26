#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
import  os
from read import *
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#读取数据
trX, trY, teX, teY = readdata()

g_Rnn = tf.Graph()
with g_Rnn.as_default():
    tf.sum
    lr = 0.001
    tr_num = 100000
    batch_size = 100

    #rnn_参数
    n_inputs = 28
    n_step = 28
    n_hidden_uints = 128
    n_classes = 10

    x = tf.placeholder(tf.float32,[None,n_step,n_inputs])
    y = tf.placeholder(tf.float32,[None,n_classes])
    weights = {
        #28*128
        "in" : tf.Variable(tf.random_normal([n_inputs,n_hidden_uints])),
        #128*10
        "out":tf.Variable(tf.random_normal([n_hidden_uints,n_classes]))
    }
    biases = {
        #  128,
        "in":tf.Variable(tf.constant(0.1,shape=[n_hidden_uints,])),
        #  10,
        "out":tf.Variable(tf.constant(0.1,shape=[n_classes,]))
    }

    def RNN_model(X,weights,biases):
        X  = tf.reshape(X,[-1,n_inputs])

        #hidden layer
        X_in = tf.matmul(X,weights["in"])+biases["in"]
        X_in = tf.reshape(X_in,[-1,n_step,n_hidden_uints])
        #基恩的循环网络单元 lstm
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_uints,forget_bias = 1.0,state_is_tuple = True)
        #初始化为0
        init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
        outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)
        results = tf.matmul(final_state[1],weights["out"])+biases["out"]
        return results

    pred = RNN_model(x,weights,biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
    accuracuy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

def RNN():
    with tf.Session(graph=g_Rnn) as sess:
        tf.global_variables_initializer().run()

        step = 0
        while (step +1)* batch_size < len(trX):
            batch_xs = trX.reshape([-1,28,28])
            teX1 = teX.reshape([-1,28,28])
            sess.run([train_op],feed_dict={x:batch_xs[batch_size*(step):(step+1)*batch_size],
                                           y:trY[batch_size*(step):(step+1)*batch_size]})
           # if step % 20 == 0:
            print(sess.run(accuracuy,feed_dict={x:teX1[:batch_size],y:teY[:batch_size]}))
            step = step+1

if __name__ == '__main__':
    RNN()
    """
    0.96875
    0.960938
    0.960938
    0.960938
    0.953125
    0.945313
    0.945313
    0.945313
    0.945313
    0.929688
    0.929688
    0.945313
    0.945313
    0.945313
    0.945313
    """