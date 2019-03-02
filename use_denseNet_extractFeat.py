# -*- coding: utf-8 -*-

# 输入的只是一个测试数据库，要将其分为三张模板，其余为输入
# 计算输入与所有模板的相似度，排序，计算输入与对应模板相似度排在第1,5,10,20位的准确率
# top1没有识别对的图像，以及他被认错的图像找出来放到一起分析

# 测试用densenet提取出来的图像特征
from __future__ import division
from keras.applications.densenet import DenseNet121
import functools
from keras.models import load_model, Model
from keras.preprocessing import image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Model
from keras.layers import *

iscenterloss = True
load_wei = False
load_full = True

CLASS_NUM = 3852
# import itertools
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if iscenterloss:
    if load_wei:
        # 使用加载权重的方法
        base_model = DenseNet121(include_top=False, weights='imagenet')  # 只有include_top为False时需要设定，并且一般有个最小值的设定
        # 上面类返回的使用个Model类的对象
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)  # 1024维的特征层
        predictions = Dense(CLASS_NUM, activation='softmax')(x)
        lambda_c = 1.5
        input_target = Input(shape=(1,))  # 标签
        centers = Embedding(CLASS_NUM, 1024)(input_target)  # Embedding层用来存放中心
        l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([x, centers])
        base_model = Model(inputs=[base_model.input, input_target], outputs=[predictions, l2_loss])
        base_model.load_weights('', by_name=False)

    if load_full:
        def my_loss(y_true, y_pred):
            return y_pred
        base_model = load_model('/home/wdxia/network_test/model_trained/denseNet_result/denseNet_GRG_3852_centerloss_128and0.5_200-0.060.hdf5', custom_objects={'my_loss':my_loss})

    model = Model(inputs=base_model.input[0], outputs=base_model.get_layer('avg_pool').output)

else:
    base_model = load_model('/home/wdxia/network_test/model_trained/denseNet_result/denseNet_GRG_3852_centerloss_80-0.286.hdf5')  # denseNet_GRG_3852_190-0.121.hdf5 ,denseNet_samplewiseNorm_200-0.113.hdf5
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


classNum = 0
dict_model = dict()
dict_test = dict()
root = '/home/wdxia/Finger_ROI_Database/TH_ROI_complete/TSINGHUA_rawData_ROI'#'/home/wdxia/Finger_ROI_Database/SD_ROI_complete/SDUMLA_rawData_ROI'
imageDirs = os.listdir(root)  # 所有子文件夹的名称


# 和训练时一样的预处理
def preprocess_input(x):
    image_gen = ImageDataGenerator(
        rescale=1./ 255,
        samplewise_center=True,  # set each sample mean to 0
        samplewise_std_normalization=True)
    i = 0
    for batch in image_gen.flow(x, batch_size=1):
        i += 1
        if (i >= 1):
            break
    return batch


# 根据相似度，对相同手指的相似度在总的相似度中做排序
def get_seq_index(dict_model, testFeatures, testLabes):
    print "Get sequence index..."
    # testFeatures = list(dict_test.values())
    # testFeatures = list(itertools.chain.from_iterable(testFeatures))  # 将嵌套的python list合并
    # testLabes = list(dict_test.keys())    # 也要扩展成与特征对应的标签
    num_test = len(testLabes)
    seq_index = []
    # 将测试图像转为了list方便计算
    for i in xrange(num_test):
        neg_similar = []
        pos_similar = 0
        for j in dict_model.keys():
            distances = cosine_similarity(testFeatures[i:i + 1], dict_model[j])  # 循环计算每个测试图像与一个模板图像的相似度，并求最大值
            distances = np.squeeze(distances)
            max_similar = np.max(distances)
            if testLabes[i] == j:
                pos_similar = max_similar
            else:
                neg_similar.append(max_similar)
        sorteed_neg_similar = sorted(neg_similar, reverse=True)  # 从大到小排列

        index = 0  # index 表明要index个才能找到
        for sim in sorteed_neg_similar:
            index += 1
            if pos_similar > sim:
                break
        seq_index.append(index)
    return seq_index


def cal_top_n(seq_index, n):
    print "Calc top n..."
    top_num = 0
    for index in seq_index:
        if index <= n:
            top_num += 1
    acc = top_num / len(seq_index)
    return acc

testLabels = []
testFeatures = []
# 对每一张测试图像 用模型得特征,并将特征和对应标签用字典保存
for imageDir in imageDirs:
    print(imageDir)
    n = 0
    imageFiles = os.listdir(os.path.join(root, imageDir))  # 某一个子文件夹下所有图片的名字
    modelFeatures = []
    for imageFile in imageFiles:
        img_path = os.path.join(root, imageDir, imageFile)
        img = image.load_img(img_path, target_size=(40, 80))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        dense_1_feature = model.predict(x)

        if n < 3:
            modelFeatures.append(dense_1_feature)
            # modelLabels.append(classNum)
            n += 1
        else:
            testFeatures.append(dense_1_feature)
            testLabels.append(classNum)
            n += 1

    modelFeatures = np.asarray(modelFeatures, dtype=np.float64)  # 模板图像的特征
    modelFeatures = np.squeeze(modelFeatures)
    # testFeatures = np.asarray(testFeatures, dtype=np.float64)  # 所有测试图像的特征
    # testFeatures = np.squeeze(testFeatures)
    dict_model[classNum] = modelFeatures
    # dict_test[classNum] = testFeatures
    print('classNum', classNum)
    classNum += 1

# modelFeatures = np.asarray(modelFeatures, dtype=np.float64)  # 模板图像的特征
# modelFeatures = np.squeeze(modelFeatures)
# modelLabels = np.asarray(modelLabels, dtype=np.int32)
#
testFeatures = np.asarray(testFeatures, dtype=np.float64)  # 所有测试图像的特征
testFeatures = np.squeeze(testFeatures)
testLabels = np.asarray(testLabels, dtype=np.int32)

seq_index = get_seq_index(dict_model, testFeatures, testLabels)
print "top 1:",cal_top_n(seq_index, 1)
print "top 10:",cal_top_n(seq_index, 10)
print "top 20:",cal_top_n(seq_index, 20)
print "top 100:",cal_top_n(seq_index, 100)