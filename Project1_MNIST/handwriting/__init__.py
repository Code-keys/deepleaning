from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import  os
import time
import tensorflow as tf
from Project1_MNIST.handwriting.read import  *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#读取数据
trX, trY, teX, teY = readdata()

#超参数
batch_size = 180
test_size = 2500

#保存地址
MAC = "/volumes/CO_OS/"
WINDOWS = "g:"

log_dir = MAC + 'PYTHON_PRO/Project1_MNIST/handwriting/log_dir'  # 输出日志保存的路径
ckpt_dir = MAC + "PYTHON_PRO/Project1_MNIST/handwriting/ckpt_dir_Cnn"
if not os.path.exists(ckpt_dir) :
    os.makedirs(ckpt_dir)
    if not  os.path.exists(log_dir):
        os.makedirs(ckpt_dir)


g_CNN = tf.Graph()
with g_CNN.as_default():

    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape,stddev=0.1))
    def Model(X,w,w2,w21,w3,w4,w_o,p_keep_conv,p_keep_hidden):

        with tf.name_scope("layer1"):
            lla  = tf.nn.relu(tf.nn.conv2d(X,w,strides=[1,1,1,1],padding="SAME"))
            ll = tf.nn.max_pool(lla,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            ll = tf.nn.dropout(ll,p_keep_conv)
        with tf.name_scope("layer2"):
            l2a = tf.nn.relu(tf.nn.conv2d(ll, w2, strides=[1, 1, 1, 1], padding="SAME"))
            l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            l2 = tf.nn.dropout(l2, p_keep_conv)
        with tf.name_scope("layer3"):
            l21a = tf.nn.relu(tf.nn.conv2d(l2, w21, strides=[1, 1, 1, 1], padding="SAME"))
           # l21 = tf.nn.max_pool(l21a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")#无法pool 后面有问题的
            l21 = tf.nn.dropout(l21a, p_keep_conv)
        with tf.name_scope("layer4"):
            l3a  = tf.nn.relu(tf.nn.conv2d(l21,w3,strides=[1,1,1,1],padding="SAME"))
            l3 = tf.nn.max_pool(l3a,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
            l3 = tf.reshape(l3,[-1,w4.get_shape().as_list()[0]])
            l3 = tf.nn.dropout(l3,p_keep_conv)
        with tf.name_scope("layer5"):
            l4  =  tf.nn.relu(tf.matmul(l3,w4))
            l4 = tf.nn.dropout(l4,p_keep_hidden)
        with tf.name_scope("FC"):
            pyx = tf.matmul(l4,w_o)
        return pyx

    def Verify(tey,pre_y,test_size = 2500):
        num = 0
        for j in range(test_size):
            if tey[j] ==pre_y[j]:
                num +=1
        with tf.name_scope('accuracy'):
            a= float(num * 100 / test_size)
        tf.summary.scalar('accuracy', a)
        return a


    with tf.name_scope('input'):
        X = tf.placeholder('float',[None,28,28,1],name='x-input')
        Y = tf.placeholder("float",[None,10],name='y-input')
        tf.summary.image('input', tf.reshape(X,[-1,28,28,1]), 10)

    with tf.name_scope('model'):
        w = init_weights([3,3,1,16])
        w2 = init_weights([3,3,16,32])
        w21 = init_weights([3, 3, 32, 64])
        w3 = init_weights([3,3,64,64])
        w4 = init_weights([64*4*4,64])
        w_o = init_weights([64,10])

    leaning_rate = tf.placeholder("float")
    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    with tf.name_scope('model'):
        py_x = Model(X,w,w2,w21,w3,w4,w_o,p_keep_conv,p_keep_hidden)#预测值

    with tf.name_scope('loss'):     # 计算交叉熵损失（每个样本都会有一个损失）
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y))
    tf.summary.scalar('loss', cost)

    with tf.name_scope('train'):
        train_op = tf.train.GradientDescentOptimizer(leaning_rate).minimize(cost)
    with tf.name_scope('predict'):
        predict_op = tf.argmax(py_x,1)

    global_step = tf.Variable(0, name="global_step",trainable=False)  # 保存模型的计数器
    saver = tf.train.Saver() #保存之前的变量

with tf.Session(graph = g_CNN) as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    tf.global_variables_initializer().run()
    global_step.eval()
    for i in range(30):
    # 训练 先打包 再一次次训练（填数据，自动校正 W 参数）
        time_begin = time.time()
        training_batch = zip(range(0,len(teX),batch_size),
                                   range(batch_size,len(teX)+1,batch_size))
        for start ,end in training_batch:
            sess.run(train_op,feed_dict={
                X:trX[start:end],Y:trY[start:end],
                leaning_rate :0.01,
                p_keep_hidden : 0.7,p_keep_conv : 0.9})
# tensorborad
            if i % 30 == 0:
                result = sess.run(merged, feed_dict={X:trX[start:end],Y:trY[start:end],
                                                     leaning_rate :0.01,p_keep_hidden : 0.7,p_keep_conv : 0.9})  # merged也是需要run的  
                writer.add_summary(result, i)  # result是summary类型的，需要放入writer中，i步数（x轴）
#验证
        pre_y = onehot_mun(sess.run(predict_op, feed_dict={X: teX[:],p_keep_hidden:1,p_keep_conv :1}))
        result = Verify(teY,pre_y,test_size)
        print("第%d次:精度为%.2f%%,耗时%.2fs" % (i + 1, result,(-time_begin+time.time())))
#模型保存# 存储
        global_step.assign(i).eval()  # 更新计数器
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)
    writer.close()