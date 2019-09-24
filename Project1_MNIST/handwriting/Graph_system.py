#_author_ :CX
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #使用cpu未加速警告
import tensorflow as tf
import numpy as np
import tensorboard


g1 = tf.Graph()#函数创建新的计算子图  包含一种算法
g2 = tf.Graph()
g3 = tf.Graph()
g4 = tf.Graph()
g1.__init__()
g2.__init__()
g3.__init__()
g4.__init__()

tf.Graph.as_default(g1) #方便其他py文件调用

#语句下定义属于计算子图g的张量和操作
with g1.as_default():
    ''' tensorflow 算子、op、'''
    input = tf.Variable(tf.random_normal([10,9,9,3],0.0,1.0,tf.float32))
    filter = tf.Variable(np.random.rand(2,2,3,2))
    w= tf.placeholder(tf.float32,[None,28,28,1])
    b = tf.placeholder(tf.float32,[None,10])
    c = tf.placeholder(tf.float32)
    d = tf.multiply(c,c)
    init = tf.global_variables_initializer()
    print(tf.get_default_graph())

    with tf.Session(graph = g1) as Gsess:#中通过参数 graph = xxx指定当前会话所运行的计算图
        #  with tf.device()
        init.run()
        assert Gsess.run(d,{c:3.0}) == 9.0
    print(tf.get_default_graph() ,'\n')

with g2.as_default():
    ''' 命名加前缀'''
    with tf.variable_scope('top-',reuse=False) :#添加名字前缀  用在RNN里比较多
        with tf.name_scope("sub-"):
            top_val = tf.constant([3,6,5])
            Sta_ = tf.get_variable('Const',[1])
    assert  (top_val.name) == 'top-/sub-/Const:0'
    assert  (Sta_.name) == 'top-/Const:0'
    with tf.Session() as top_sess:
        print(top_val.eval())

with g3.as_default():
    """模型的加载 与存储"""
    ckpt_dir = "./ckpt_dir"
    if not os.path.exists():
        os.makedirs(ckpt_dir)

    global_step = tf.Variable(0,trainable=False,name="global_step") #计数器

    defined_all_variable = tf.placeholder(dtype="float",[-1,-1])
    #

    Saver = tf.train.Saver()  # 保存之前的变量

    #之后的数据不会被存储
    session().run() 之后
    start = global_step.eval()

    for i in range(100)：

        global_step.assign(i).eval() #更新计数器
        saver.save(sess,ckpt_dir + "/model.ckpt",global_step = global_step) #存储


    """XXXXXXXX"""
    #load_Model
    with tf.Session() as sess:
        tf.initialize_all_variables.run()

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess,ckpt.model_checkpoint_path)

    # load_Graph

    #write 2 txt
    v = tf.Varibale(0,name = "my_variable")
    with  tf.Session() as sess:
        tf.train.write_graph(graph = sess.graph_def,"/tmp/tf/tfmodel","train.pbtxt")
    #read.
    with tf.Session() as sess:
        with gfile.FastGFile("/temp/tfmodel/train.pbtxt","rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _sess.graph.as_default()
            tf.import_graph_def(graph_def,name="tfgraph")


with g4.as_default() :
    """QUEUE  队列 """
    # 1：FIFOQueue 先进先出
    def fifoQ():
        Q = tf.FIFOQueue(100, "float")
        #     ([
        # [[10, 20, 30,40,50,60,70,80,90],[100,200,300,400,500,600,700,800,900]],
        # [[1000,2000,3000,4000],[]]
        #                    ])

        # init = Q.enqueue(1.0)
        init = Q.enqueue_many(([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],))

        x = Q.dequeue()
        y = x - 1
        q_inc = Q.enqueue([y])

        with tf.Session() as sess:
            sess.run(init)

            assert sess.run(x) == 10.0
            sess.run(q_inc)
            assert sess.run(x) == 30.0
            sess.run(q_inc)
            assert sess.run(x) == 50.0

            queue = sess.run(Q.size())
            for i in range(queue):
                print(sess.run(Q.dequeue()))

    def randomSQ():
        sQ = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")
        with tf.Session() as sess:
            for i in range(0, 10):
                sess.run(sQ.enqueue(i * 10))
            for i in range(0, 8):
                print(sess.run(sQ.dequeue()))

    def thread():
        # 1：FIFOQueue 先进先出
        import tensorflow as tf

        q = tf.FIFOQueue(capacity=1000, dtypes="float")
        counter = tf.Variable(0.0, dtype="float")
        increse = tf.assign_add(counter, tf.constant(1.0))
        enqueue_op = q.enqueue([counter])

        qr = tf.train.QueueRunner(q, enqueue_ops=[increse, enqueue_op] * 1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coor = tf.train.Coordinator()
            enqueue_threads = qr.create_threads(sess, coord=coor, start=True
                                                )
            # 主线程
            coor.request_stop()  # 通知其他线程关闭
            for i in range(0, 10):
                try:
                    print(sess.run(q.dequeue()))
                except tf.errors.OutOfRangeError:
                    break
            coor.join(enqueue_threads)  # 关闭后 进行此函数
