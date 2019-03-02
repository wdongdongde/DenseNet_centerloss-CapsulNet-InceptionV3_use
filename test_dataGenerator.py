# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
# 用山东大学测试数据库看下数据增强的效果
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

validation_generator = train_datagen.flow_from_directory(
    '/home/users/wdd/myDatabase/Finger_ROI/SDUMLA_splitValidSet',
    target_size=(90, 180),
    batch_size=32,
    # color_mode='grayscale',
    class_mode='categorical',
    save_to_dir='/home/users/wdd/myDatabase/SDUMLA_gen_test',
    save_prefix='gen'
)




