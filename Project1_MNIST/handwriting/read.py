import numpy as np
import os

def imshow(img,lable,mun):
    """
    # 显示img数字
    :param data: image
    :param mun: show num
    :return: 0
    """
    import matplotlib.pyplot  as plt
    for i in range(mun):
        img1 = np.resize(img[i], (28, 28))
        plt.imshow(img1)
        print(lable[i])
        plt.show()
    return

def onehot_mun(loaded_y):  # \编码  class
    """
    :param loaded_y: num
    :return: coded
    """
    D = {"0": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "1": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], "2": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         "3": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], "4": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], "5": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         "6": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], "7": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], "8": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         "9": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
    loady = list()
    for i in range(len(loaded_y)):
        change = str(loaded_y[i])
        loady.append(D[(change)])
    return loady

def de_num(onehot):
    num = list()
    for i in onehot:
        num.append(i.index(1))
    return num


def onehot_alph(loaded_y):
    """
    :param loaded_y: 输入为字母列表 或连续字母
    :return:one hot encode
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    onehot_encoded = list()

    for value in loaded_y:
        letter = [0 for _ in range(len(alphabet))]
        letter[alphabet.index(value)] = 1
        onehot_encoded.append(letter)

    return onehot_encoded

def de_alph(loaded_y):
    de_alph = list()
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for i in loaded_y:
        de_alph.append(alphabet[i.index(1)])
    return de_alph


def readdata():
    """
      读取mnist图像数据 类似bmp文件
    :return: 数据/流
    """
    # 加载二进制数据
    def _load_x(dir):
        x = open(dir,"rb")
        x.read(16)  # 所在的位置开始读16个字节
        loaded_x = np.fromfile(x, dtype=np.uint8, count=-1) / 255
        loaded_x = np.reshape(loaded_x, [-1, 28, 28, 1])
        x.close()
        return loaded_x

    def _load_y(dir):
        with open(dir, 'rb') as y:# 解析
            y.read(8)#所在的位置8个 offset 开始字节
            y = np.fromfile(y, dtype=np.uint8)
        return y

    if os.name == "nt":
        DIR = "G:/"
    else:
        DIR = "/volumes/CO_OS/"

    loaded_x = _load_x(DIR + "PYTHON_PRO/Project1_MNIST/handwriting/data/train-images-idx3-ubyte")
    loaded_y = _load_y(DIR + 'PYTHON_PRO/Project1_MNIST/handwriting/data/train-labels-idx1-ubyte')

    loaded_xte = _load_x(DIR + "PYTHON_PRO/Project1_MNIST/handwriting/data/t10k-images-idx3-ubyte")
    loaded_yte = _load_y(DIR + 'PYTHON_PRO/Project1_MNIST/handwriting/data/t10k-labels-idx1-ubyte')

    loaded_y = onehot_mun(loaded_y)
    loaded_yte = onehot_mun(loaded_yte)

    print('load complete!')
    return loaded_x ,loaded_y,loaded_xte,loaded_yte

if __name__ == '__main__':

    a,b ,c,d =  readdata()
    print(len(a))
    print(len(b))
    print(len(c))
    print(len(d))

    b = de_num(b)
    print(b[5])


    imshow(a,b,2)

    a = onehot_alph("abcd")
    print(a)
    print(de_alph(a))