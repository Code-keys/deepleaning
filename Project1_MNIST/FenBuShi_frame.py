#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import tensorflow as tf
#命令行参数解析，获取集群信息与工作的节点hosts  以及工作节点和任务下标
tf.app.flags.DEFINE_string("ps hosts", "", "coma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker. hosts", "", "Corma- separated list of hostname :portpairs")

tf.app.flags.DEFINE_string("job name", "","One of 'ps', 'worker'"l
tf.app.flags.DeFINE_integer("task index", o，"Index of task within the job"〉
FLAGS = tf.app.flags.FLAGS
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts(",")

# 第2歩:創建当前任努芦点的服努器

cluster = tf.train.ClusterSpec("ps": ps_hosts, "worker": worker_hosts))

server = tf.train.Server(cluster, job_name = FLAGs.job_name, task_index = FLAGS.task_index)

#第3歩:如果当前芍点是参数服多器，則調用server.join〈无休止等待;如果是エ作苓点，則抉行第4歩
if FLAGS.job name == "ps":

    server.join()

# 第4歩:枸建要訓的模型，杓建汁算圏
elif FLAGS.job name =- "worker":
    # build tensorflow graph modeI  pmD

# 第5歩:創建tf.train. Supervisor米管理模型的訓祢辻程
# #創建一个supervisor来監督訓祢辻程

sv = tf.train.Supervisor( is_chief=(FLAGS.task_index == 0)，logdir - "/tmp/train logs")

#supervisor 負責会蛞初始化和从橙査点恢夏樸型
sess = sv.prepare_or_wait_for_session(server.target)

+幵始循坏，宜到supervisor停止
while not sv.should stop():
    # 訓綜模型
