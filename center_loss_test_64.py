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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 一些参数
MODI_MODEL = False
USE_MODEL = True
CLASS_NUM = 3852 #2289
epochs = 200
batch_size = 64
num_classes = CLASS_NUM
IMG_HEIGHT = 40
IMG_WIDTH = 80
isCenterloss = True

# 模型结构

if USE_MODEL:
    def my_loss(y_true, y_pred):
        return y_pred

    # 模型结构
    base_model = DenseNet121(include_top=False, weights='imagenet')  # 只有include_top为False时需要设定，并且一般有个最小值的设定
    # 上面类返回的使用个Model类的对象
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)  # 1024维的特征层
    predictions = Dense(CLASS_NUM, activation='softmax')(x)
    lambda_c = 1.5
    input_target = Input(shape=(1,))  # 标签
    centers = Embedding(CLASS_NUM, 1024)(input_target)  # Embedding层用来存放中心
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')([x, centers])
    model = Model(inputs=[base_model.input, input_target], outputs=[predictions, l2_loss])
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy', my_loss], loss_weights=[1., lambda_c], metrics=['accuracy'])
    # lambda y_true, y_pred: y_pred表示输入真实标签和预测标签，输出预测标签，这里的预测标签即l2loss，所以目标是最小化交叉熵损失函数及l2损失
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
    batch_size=64,
    # color_mode='grayscale',
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    '/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid',
    # '/home/wdxia/Finger_ROI_Database/Database_split_valid',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64,
    # color_mode='grayscale',
    class_mode='categorical')


# 回调函数
modelcheckpoint = keras.callbacks.ModelCheckpoint(
        '/home/wdxia/network_test/model_trained/denseNet_result/denseNet_GRG_3852_centerloss_64_{epoch:02d}-{val_loss:.3f}.hdf5',
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', period=10)

csv_logger = keras.callbacks.CSVLogger('/home/wdxia/network_test/model_trained/denseNet_result/denseNet_GRG_3852_centerloss_64_training.log')

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
        savefig("/home/wdxia/network_test/model_trained/denseNet_result/FVdenseNet_GRG_3852_centerloss_64_loss.jpg")

loss_history = LossHistory()

# 先训练新加的几层
# for layer in base_model.layers:
#     layer.trainable = False

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=25734 // 64 + 1,  # 训练图像数/batch_size 3180 3180 // 64
#     epochs=50,
#    )  # 只用训练数据训练，这一步可以省掉试一下


# 继续训练指定层

# inputs=[base_model.input, input_target] 模型的输入包括了训练数据 以及其标签，则在batch训练时应该即训练数据，也训练标签
def get_batch_input(data_generator, batch_size):
    # 生成的数据
    # labels = data_generator.classes  # 数字标签
    while True:
        tmp = data_generator.next()
        labels_oneshot = tmp[1]  # 二进制编码
        labels = np.argmax(labels_oneshot, axis=1)
        imgs = tmp[0]
        # print labels.shape[0]
        if labels.shape[0] != batch_size or imgs.shape[0]!=batch_size:
            labels_oneshot = tmp[1]  # 二进制编码
            labels = np.argmax(labels_oneshot, axis=1)
            imgs = tmp[0]
        random_labels = np.random.rand(len(labels), 1)
        yield ([imgs, labels], [labels, random_labels])


model.fit_generator(
    get_batch_input(train_generator, 64),
    steps_per_epoch=50866 // 64,  # 25734 // 64 + 1,  # 训练图像数/batch_size 3180 3180 // 64
    epochs=epochs,
    callbacks=[modelcheckpoint, csv_logger, loss_history], # , csv_logger, loss_history
    validation_data=get_batch_input(validation_generator, 64),
    validation_steps=15144 // 64)  # 7972 // 64 + 1)  # 测试图像数/batch_size 636  636 // 64

# 绘制acc-loss曲线
loss_history.loss_plot('epoch')

# model.save('/home/wdxia/network_test/model_trained/denseNet_result/test.df5')








