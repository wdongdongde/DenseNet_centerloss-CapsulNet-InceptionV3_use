# -*- coding: utf-8 -*-
from keras.applications.densenet import DenseNet121
import functools
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from keras.layers import *

from keras.preprocessing.image import ImageDataGenerator
import keras
import os
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import savefig
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 一些参数
MODI_MODEL = False
USE_MODEL = True
CLASS_NUM = 3852 #2289
epochs = 200
batch_size = 32
num_classes = CLASS_NUM
IMG_HEIGHT = 40
IMG_WIDTH = 80
isCenterloss = False

# 模型结构
if MODI_MODEL:
    # 模型结构
    base_model = DenseNet121(include_top=False, weights='imagenet')  # 只有include_top为False时需要设定，并且一般有个最小值的设定
    # base_model.summary()
    # 经过base_model的前几层
    my_firstLayer = base_model.get_layer('conv2_block2_concat').output
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis)(my_firstLayer)
    # x = Dense(1024, activation='relu')(x)
    # x = Flatten()(x)   # 先展平再输入全连接层,但是展平需要知道尺寸
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(CLASS_NUM, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    for layer in model.layers:
        layer.trainable = True
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)
    #     print(layer.trainable)

if USE_MODEL:
    # 模型结构
    base_model = DenseNet121(include_top=False, weights='imagenet')  # 只有include_top为False时需要设定，并且一般有个最小值的设定
    # 上面类返回的使用个Model类的对象
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)  # 1024维的特征层
    predictions = Dense(CLASS_NUM, activation='softmax')(x)

    # 使用centerloss损失函数
    if isCenterloss:
        def _center_loss_func(features, labels, alpha, num_classes):
            feature_dim = features.get_shape()[1]
            # Each output layer use one independed center: scope/centers
            centers = K.zeros([num_classes, feature_dim])
            labels = K.reshape(labels, [-1])
            labels = tf.to_int32(labels)
            centers_batch = tf.gather(centers, labels)
            diff = (1 - alpha) * (centers_batch - features)
            centers = tf.scatter_sub(centers, labels, diff)
            loss = tf.reduce_mean(K.square(features - centers_batch))
            return loss


        def get_center_loss(alpha, num_classes):
            """Center loss based on the paper "A Discriminative
               Feature Learning Approach for Deep Face Recognition"
               (http://ydwen.github.io/papers/WenECCV16.pdf)
            """
            @functools.wraps(_center_loss_func)
            def center_loss(y_true, y_pred):
                return _center_loss_func(y_pred, y_true, alpha, num_classes)

            return center_loss

        center_loss = get_center_loss(1.5, num_classes)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='sgd', loss=center_loss, metrics=['accuracy'])

        # lambda_c = 1.5
        # input_target = Input(shape=(1,))  # single value ground truth labels as inputs
        # centers = Embedding(CLASS_NUM, 1024)(input_target)
        # l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')(
        #     [x, centers])
        # model = Model(inputs=[base_model.input, input_target], outputs=[predictions, l2_loss])
        # for layer in model.layers:
        #     layer.trainable = True
        # model.compile(optimizer=Adam(), loss=["categorical_crossentropy", lambda y_true, y_pred: y_pred],
        #               loss_weights=[1, lambda_c], metrics=['accuracy'])
    # 使用普通的softmax损失
    else:
        # x = Dense(1024, activation='relu')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        for i, layer in enumerate(base_model.layers):
            print(i, layer.name)
        # 控制哪几层需要训练
        # for layer in model.layers[:50]:
        #     layer.trainable = False
        for layer in model.layers:
            layer.trainable = True
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()


# 数据的增强
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        samplewise_std_normalization=True,
        rotation_range=15,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=10,  # randomly shift images horizontally
        height_shift_range=5,  # randomly shift images vertically
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        vertical_flip=False,  # randomly flip images
)

valid_datagen = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,
                                   samplewise_std_normalization=True)

train_generator = train_datagen.flow_from_directory(
    '/home/wdxia/Finger_ROI_Database/GRG_3852_train',
    # '/home/wdxia/Finger_ROI_Database/Database',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    # color_mode='grayscale',
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    '/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid',
    # '/home/wdxia/Finger_ROI_Database/Database_split_valid',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    # color_mode='grayscale',
    class_mode='categorical')


# 回调函数
modelcheckpoint = keras.callbacks.ModelCheckpoint(
        '/home/wdxia/network_test/model_trained/denseNet_result/denseNet_GRG_3852_{epoch:02d}-{val_loss:.3f}.hdf5',
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', period=10)

csv_logger = keras.callbacks.CSVLogger('/home/wdxia/network_test/model_trained/denseNet_result/denseNet_GRG_3852_training.log')

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        # 创建一个图
        # plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  # plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)  # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')  # 给x，y轴加注释
        plt.legend(loc="upper right")  # 设置图例显示位置
        #plt.show()
        savefig("/home/wdxia/network_test/model_trained/denseNet_result/FVdenseNet_GRG_3852_loss.jpg")

loss_history = LossHistory()

# 先训练新加的几层
# for layer in base_model.layers:
#     layer.trainable = False

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=25734 // 32 + 1,  # 训练图像数/batch_size 3180 3180 // 32
#     epochs=50,
#    )  # 只用训练数据训练，这一步可以省掉试一下


# 继续训练指定层

model.fit_generator(
    train_generator,
    steps_per_epoch=50866 // 32 + 1,  # 25734 // 32 + 1,  # 训练图像数/batch_size 3180 3180 // 32
    epochs=epochs,
    callbacks=[modelcheckpoint, csv_logger, loss_history],
    validation_data=validation_generator,
    validation_steps=15144 // 32 + 1)  # 7972 // 32 + 1)  # 测试图像数/batch_size 636  636 // 32

# 绘制acc-loss曲线
loss_history.loss_plot('epoch')


