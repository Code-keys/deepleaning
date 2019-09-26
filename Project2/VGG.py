#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from __future__ import print_function
from read import *
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.optimizers import SGD,Adam,RMSprop
from keras.utils import np_utils

#函数化模型 keras
