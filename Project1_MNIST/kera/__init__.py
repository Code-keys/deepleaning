#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from keras.models import Sequential

from keras.layers.core import Dense,Activation,Dropout  # BP 正则化  dropout

from keras.layers.recurrent import Recurrent,SimpleRNN,GRU,LSTM # RNN

from keras.layers.convolutional import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D # CNN

"""
Dense层  FC层

Activation

Dropout 

model.add(Flatten())# #张量化  

keras.layers.core.Reshape(target_shape) model.output_shape ==

keras.layers.core.Permute(dims)   维度从新排列

keras.layers.core.RepeatVector(n)   2D张量，输出3D张量

keras.engine.topology.Merge(layers=None, mode='sum',  #一个张量列表中的若干张量合并为一个单独的张量
                            concat_axis=-1, dot_axes=-1, output_shape=None, node_indices=None, tensor_indices=None, name=None)

keras.layers.core.Lambda(function, output_shape=None, arguments={})

keras.layers.core.ActivityRegularization(l1=0.0, l2=0.0) # 经过本层的数据不会有任何变化，但会基于其激活值更新损失函数值

keras.layers.core.Masking(mask_value=0.0)  使用给定的值对输入的序列信号进行“屏蔽”，用以定位需要跳过的时间步。

keras.layers.core.Highway(init='glorot_uniform', transform_bias=-2, activation='linear', weights=None, W_regularizer=None,
 #全连接FC                           b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)

MaxoutDense层 全连接的Maxout层

"""

