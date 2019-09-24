#day 1
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#添加图的  节点  和之间的逻辑运算  控制依赖  各个传递子函数
a = tf.constant([[1., 2.]
                 ])#边
b = tf.constant([[2.],
                 [1.]
                 ])#边
c = tf.placeholder(tf.float32,(1,28,28,1))
d = tf.placeholder(tf.float16)#张量 内置函数name、dtype、value_index......

output = tf.add(c, d)#节点 操作符号 #type、inputs、outputs


# 会话只用来进行  计算 赋值 数据依赖  传递函数（输入 输出两项）
with tf.Session() as sess0:
    # with tf.device('/cpu:1'):#设备
    y_p = 1
    init = tf.global_variables_initializer()

    out = tf.subtract(c,d)
    print(out.op.name)

    #sess0._extend_graph()
    a = sess0.run([output], feed_dict={c: [0], d: [1]})#进行计算
    sess0.run(init)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

tf.Graph.as_default(sess0)
print(out.graph)


'''添加前缀 variable_scope   name_scope
'''
with tf.variable_scope('funny',reuse=True):
    with tf.name_scope('haha'):
        T = tf.get_variable('E', [1])
        # name参数,给op指定标识
        f = tf.Variable(tf.zeros([1]), name='zero')
        g = 1 + e  # 等价于 f = tf.add(1, e)

print ( 'T.name    '   ,e.name)
print ('f.name    '   ,d.name )
print ('f.op.name ',d.op.name)
print ('g.name     '  ,f.name)
print ('g.op.name  ',f.op.name)

with tf.variable_scope('var4') as var4_scope:
    v4 = tf.get_variable('v', [1])
    with tf.name_scope('var5') as var5_scope:
        v5 = tf.get_variable('w', [1])
assert var4_scope.name == 'var4'
assert v4.name == 'var4/v:0'
print(v5)


