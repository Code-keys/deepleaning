#!/usr/bin/env python 
# -*- coding:utf-8 -*-
def plot():
    import matplotlib.pyplot as plt
    import numpy as np
    i = [x for x in range(300)]
    y = list()
    for x in i:
        y.append(0.001+(1)/(2+np.exp((x-75)*0.04)))
    plt.plot(i,y)
    plt.show()













if __name__ == '__main__':
    plot()