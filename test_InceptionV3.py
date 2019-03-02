# -*- coding: utf-8 -*-
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import math

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 构建不带分类器的预训练模型
base_model = InceptionV3(weights='imagenet', include_top=False)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)   # 全局平均池化后尺寸就变为（batch_size,channels(featuremap数量）），方便连接全连接层

# 添加一个全连接层
x = Dense(1024, activation='relu')(x)

# 添加一个分类器
predictions = Dense(2289, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

model.summary()
# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 在新的数据集上训练几代

train_datagen = ImageDataGenerator(
    # rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=True,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by dataset std
    samplewise_std_normalization=True,  # divide each input by its std
    rotation_range=15,  # randomly rotate images in 0 to 180 degrees
    width_shift_range=10,  # randomly shift images horizontally
    height_shift_range=5,  # randomly shift images vertically
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    vertical_flip=False,  # randomly flip images
)


valid_datagen = ImageDataGenerator(rescale=1. / 255)

target_img_height = 90
target_img_width = 180
train_samples = 25614
valid_samples = 8092  # 公司库和公开库的合集
epochs = 100
train_batch_size = 32
valid_batch_size = 32

train_generator = train_datagen.flow_from_directory(
    '/home/users/wdd/myDatabase/SD_TH_ML_CP_set',
    target_size=(target_img_height, target_img_width),
    batch_size=32,
    # color_mode='grayscale',
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    '/home/users/wdd/myDatabase/SD_TH_ML_CP_set_splitValidSet',
    target_size=(target_img_height, target_img_width),
    batch_size=32,
    # color_mode='grayscale',
    class_mode='categorical')

# 下面是一些回调函数
# 如果检测的值（损失或准确度）没有提升，则提前停止
# earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, mode='auto')
# 保存中间最好的模型
# 每隔periodepoch检查一次，保存这段epoch内最好的模型
modelcheckpoint = keras.callbacks.ModelCheckpoint('/home/users/wdd/model_trained/InceptionV3_{target_img_height:03d}_{target_img_width:03d}_{epoch:02d}-{val_loss:.2f}.hdf5',
                                                  monitor='val_loss', verbose=0, save_best_only=True,
                                                  save_weights_only=False, mode='auto', period=20)
csv_logger = keras.callbacks.CSVLogger('/home/users/wdd/model_trained/training_{target_img_height:03d}_{target_img_width:03d}.log')


# 记录损失历史(自定义的回调函数）
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


loss_history = LossHistory()

model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // train_batch_size,  # 训练图像数/batch_size 3180 3180 // 32
    epochs=epochs,
    callbacks=[modelcheckpoint, csv_logger, loss_history],
    validation_data=validation_generator,
    validation_steps=valid_samples // valid_batch_size)  # 测试图像数/batch_size 636  636 // 32



model.save('/home/users/wdd/model_trained/InceptionV3_{target_img_height:03d}_{target_img_width:03d}.h5')
