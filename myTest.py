# -*- coding: utf-8 -*-
from keras.layers.merge import add
import keras
from keras.constraints import maxnorm
from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Dense, Flatten, Embedding, Lambda
from keras.layers import Conv2D,BatchNormalization,PReLU,Input,AveragePooling2D

from keras.layers.convolutional import  MaxPooling2D, ZeroPadding2D

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

inputs = Input(shape=(60, 150, 1))  # 输入是60行150列的单通道图像
x = ZeroPadding2D((2, 2))(inputs)  # 为输入图像做0填充边缘
x1 = Conv2D(32, (5, 5))(x)
x2 = BatchNormalization(axis=1)(x1)
x3 = PReLU()(x2)
x4 = Flatten()(x3)
x5 = Dense(1024)(x4)
ip1 = PReLU(name='ip1')(x)

