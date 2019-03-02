# -*- coding: utf-8 -*-
"""
从总的数据集里面分出一部分不重合的作为验证集
"""

import os
import string
import random
import numpy as np
import shutil
import sys

#Root = os.path.join(os.getcwd(),sys.argv[1]) #getcwd()是获得当前路径
#print os.getcwd()
#sys.argv[1]获取命令行参数的
#outRoot = os.path.join(os.getcwd(),sys.argv[2])
def allset2TrainandVal():
    Root=r"/home/wdxia/Finger_ROI_Database/GRG_3852_train"
    outRoot=r"/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid"
    newDIR=outRoot
    if(not(os.path.exists(newDIR))):
        os.mkdir(newDIR)
    inputPartSet=Root
    outValPartSet = newDIR
    foldername = os.listdir(inputPartSet)

    for i in foldername:
        folder = "%s/%s"%(inputPartSet,i)# 获取每一个文件夹
        pics=os.listdir(folder)# 原每一个文件夹中图片的名字
        N=len(pics)
        newfolder = "%s/%s" % (outValPartSet, i)
        if (not (os.path.exists(newfolder))):
            os.mkdir(newfolder)    # 没有图片也有类别！！?
        if(N>1):
            rng_state = np.random.get_state()
            np.random.shuffle(pics)
            for index in range(int(max(1,N*0.25))):
                srcpic="%s/%s"%(folder,pics[index])
                dstvalpic="%s/%s"%(newfolder,pics[index])
                shutil.move(srcpic,dstvalpic)
            print i

allset2TrainandVal()