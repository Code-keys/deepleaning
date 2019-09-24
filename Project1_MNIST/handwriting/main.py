#_author_ :CX
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #使用cpu未加速警告
"""QUEUE MANERGEMENT  """
# 1：FIFOQueue 先进先出
import tensorflow as tf

q= tf.FIFOQueue(capacity=1000,dtypes = "float")
counter = tf.Variable(0.0,dtype="float")
increse = tf.assign_add(counter,tf.constant(1.0))
enqueue_op = q.enqueue([counter])

qr = tf.train.QueueRunner(q,enqueue_ops=[increse,enqueue_op]*1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coor = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess,coord=coor,start=True
                                        )
    #主线程
    coor.request_stop()#通知其他线程关闭
    for i in range(0,10):
        try:
            print(sess.run(q.dequeue()))
        except tf.errors.OutOfRangeError:
            break
    coor.join(enqueue_threads)#关闭后 进行此函数






    enqueue_threads = qr.create_threads(sess,start=True)
    for i in range(10):
        print(sess.run(q.dequeue()))

    '''
#init = Q.enqueue(1.0)
init0 = Q.enqueue([0,1,2,3,4,5,6,7,8,9])
init = Q.enqueue_many([[10, 20, 30,40,50,60,70,80,90],])

deq1 = Q.dequeue()
y = deq1- 1
q_inc = Q.enqueue([y])

with tf.Session() as sess:
    sess.run(init)
    #sess.run(init0)

    assert sess.run(deq1) == 10.0
    sess.run(q_inc)
    assert sess.run(deq1) == 30.0
    sess.run(q_inc)
    assert sess.run(deq1) == 50.0

    queue = sess.run(Q.size())
    for i in range(queue):
        print(sess.run(Q.dequeue())''
        )
    '''