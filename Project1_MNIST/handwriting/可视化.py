
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse
import sys

# Tensorboard可以记录与展示以下数据形式：
# （1）标量Scalars
# （2）图片Images
# （3）音频Audio
# （4）计算图Graph
# （5）数据分布Distribution
# （6）直方图Histograms
# （7）嵌入向量Embeddings

'''API
tf.summary.FileWriter.__init__(logdir, graph=None, max_queue= 10, flush_secs=120, graph_def=None)
#创建FileWriter 和事件文件，会在logdir中创建一个新的事件文件
tf.summary,.FileWriter.add summary(summary, global step=None)  将摘要添加到事件文件
tf.summary.FileWriter.add event(event)  向事件文件中添加一个事件
tf.summary. FileWriter.add_ graph(graph, global step= None, graph_def = None)
tf.summary.FileWriter.get_logdir()
tf.summary FileWriter.flush()
tf.summary.FileWriter.close()
tf.summary.scalar(name, tensor, collections None)
tf.summary.histogram(name, values, collections=None)
tf. summary.audio( name, tensor, sample_ rate, max_ outputs= 3,collections=None)
tf.summary. image( name, tensor, max_ outputs= =3, collections= None)
tf.summary.merge(inputs, collections=None, name=None)
'''
'''可视化过程'''
#（1）首先肯定是先建立一个graph,你想从这个graph中获取某些数据的信息


#（2）确定要在graph中的哪些节点放置summary operations以记录信息
''' 使用tf.summary.scalar记录标量 
    使用tf.summary.histogram记录数据的直方图 
    使用tf.summary.distribution记录数据的分布图 
    使用tf.summary.image记录图像数据 '''



#（3）operations并不会去真的执行计算，除非你告诉他们需要去run,或者它被其他的需要run的operation所依赖。而我们上一步创建的这些summary operations其实并不被其他节点依赖，因此，我们需要特地去运行所有的summary节点。但是呢，一份程序下来可能有超多这样的summary 节点，要手动一个一个去启动自然是及其繁琐的，因此我们可以使用tf.summary.merge_all去将所有summary节点合并成一个节点，只要运行这个节点，就能产生所有我们之前设置的summary data。

#（4）使用tf.summary.FileWriter将运行后输出的数据都保存到本地磁盘中

#（5）运行整个程序，并在命令行输入运行tensorboard的指令，之后打开web端可查看可视化的结果
