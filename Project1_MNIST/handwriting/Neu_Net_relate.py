import tensorflow as tf

'''
神经网络:
激活函数: 可微分：tf.nn.sigmoid、tanh、、、
          不可微分：relu
'''

def tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None):
''' 输入图像Tensor : [图片数量float32/64, 图片高度, 图片宽度, 图像通道数]
    CNN中的卷积核Tensor:  [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
    步长:一维的向量 长度4
    卷积方式：string类型的量，只能是”SAME”, ”VALID”其中之一
    输出：[图片数量float32/64, 图片高度, 图片宽度, 图像通道数]'''
    tf.nn.depthwise_conv2d()...
    tf.nn.convolution()
    tf.nn.separable_conv2d()
    tf.nn.atrous_conv2d()
    tf.nn.conv2d_transpose()
    tf.nn.conv1d()
    tf.nn.conv3d()
    return 0

def tf.nn.avg_pool(value, ksize, strides, padding, name=None):
 '''输入图像  [batch, in_height, in_width, in_channels]：
    池化窗口:[1, height, width, 1]
    步长:[ 1, strides, strides, 1]
    边界: "SAME”or“VALID”'''
    value = tf.Variable([1,2,3,4,5,6,7,8,9.0]).reshape(value,[1,3,3,1])
    value = tf.reshape(value,[1,3,3,1])
    ksize = [1, 2, 2, 1]
    pool = tf.nn.max_pool(value, ksize, strides=[1, 1, 1, 1], padding='VALID')

    tf.nn.max_pool()
    tf.nn.max_pool3d()
    tf.nn.

 '''
  
import tensorflow as tf
import numpy as np
value = tf.Variable([1,2,3,4,5,6,7,8,9.0])
value = tf.reshape(value,[1,3,3,1])
ksize = [1, 2, 2, 1]
pool = tf.nn.max_pool(value, ksize, strides=[1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    # 初始化变量
    op_init = tf.global_variables_initializer()
    sess.run(op_init)

    print("value的值为：")
    print(sess.run(value))
    print("池化值为：")
    print(sess.run(pool)) 
'''


def tf.nn.sigmoid_cross_entry_with_logits(logits,targets,name = None):
    """分类函数
输入：logits:[batch_size,num_size]最后一层的输入即可
     terget:[batch_size,size]
outputs：loss ：[batch_size,num_size】保存的交叉熵
    """
    tf.nn.softmax(logits, dim= -1, name=None):
    tf.nn.logsoftmax(logits,targets,name = None):
    tf.nn.softmax_cross_entry_with_logits(_sentinel = None,lable = None,logits = None,dim= -1,name = None):
    tf.nn.sparse_softmax_cross_entry_with_logits(logits,lable = None,name = None):

损失函数loss 交叉熵 PSNR ect.....
def loss(y_pre,y_ture):
    return tf.inner(y_pre-y_ture)**2


training 函数介绍：各种优化类提供了为损失函数
                    (损失函数由交叉熵损失函数，L1 loss,L2 loss 等)计算梯度的方法
    tf.train.GradientDesentOptimizer(learning_rate):
        """优化方法：批梯度下降
    误差平均化 作为真实误差 参与参数的更新
    Training	
    Optimizers，
    Gradient Computation
    Gradient Clipping
    Distributed execution
    输入：learning_rate
    输出：实例函数     可接:train_op = optimizer.minimize(loss, global_step=global_step)
                         
        """
    tf.train.MomentumOptimizer()
    tf.train.AdamOptimizer()
    return 0