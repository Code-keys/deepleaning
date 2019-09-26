from __future__ import print_function
from read import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils
np.random.seed(160)


#网络超参数
NB_E = 200
BAT = 128
VER = 1
NB_C = 10
OP = SGD()
N_HIDDENS = 128
VALIDATION_SPLIT = 0.2


#read  datasets
a, b, c, d = readdata()#  60000   +  10000
a = np.reshape(a,[-1,784])
c = np.reshape(c,[-1,784])
a.astype("float32")
c.astype("float32")
b = np_utils.to_categorical(de_num(b),10)
d = np_utils.to_categorical(de_num(d),10)

#建模 （序贯模型）  else 函数化模型
model = Sequential()
#第一层
model.add(Dense(NB_C,input_shape=(784,)))
model.add(Activation("softmax"))
model.add(Dropout(0.2))
#第二层
model.add(Dense(N_HIDDENS))
model.add(Activation("relu"))
model.add(Dropout(0.2))
#第二层
model.add(Dense(10))
model.add(Activation("relu"))
#model.add(Flatten())
model.summary()

#模型的编译运行
model.compile(loss="MSE",optimizer=OP,metrics=["accuracy"])#编译

history = model.fit(a,b,
                    batch_size=BAT,epochs=NB_E,
                    verbose=1,validation_split=VALIDATION_SPLIT
                    )
#P评估
score = model.evaluate(c,d,verbose=1)

print("score" ,score[0])
print("accuracy" ,score[1])
