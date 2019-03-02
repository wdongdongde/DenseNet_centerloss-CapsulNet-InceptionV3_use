# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import savefig

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True


CLASS_NUM = 2289
IMG_WIDTH = 28
IMG_HEIGHT = 28

#  -----------------------------------------------函数-----------------------------------------------  #
#  读取图像
def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data = np.array(im)
    return data


#  the squashing function ,use 0.5 instead of 1 in hinton's paper
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


#  define our own softmax function instead of K.softmax
#  because K.softmax can not specify axis
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


#  define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) +
                 lamb * (1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


#  -----------------------------------------------类---------------------------------------------- #
''' a capsule implementation with pure keras
 there are 2 versions of capsule 
 one is like dense layer (for fixed shape input)  one is like timedistributed dense(for various length input)
 the input shape of capsule must be (batch_size,input_num_capsule, input_dim_capsule)
 the output shape is (batch_size,num_capsule,dim_capsule)'''
'''用 Model子类定制自己的模型，网络层定义在_init__中，前向传播在call中指定'''


class Capsule(Layer):

    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **Kwargs):
        super(Capsule, self).__init__(**Kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule, self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule, self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    ''' Following the routing algorithnm from Hinton's paper but replace b=b+<u,v> with b=<u,v> '''

    def call(self, inputs):
        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)    # （1,128,10*16）一维卷积相当于全连接层
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])
        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs, (batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])  # (batch_size,10,64)
        for i in range(self.routings):
            c = softmax(b, 1)  # 这里为什么要加个softmax
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))   # 将短向量缩小至0 长向量接近单位向量
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])  # 带耦合系数的输出和不带耦合系数的输出之间进行互相调节
                if (K.backend() == 'theano'):
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = {'num_capsule': self.num_capsule, 'dim_capsule':self.dim_capsule, 'routings':self.routings, 'share_weights':self.share_weights, 'activation':self.activation}
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#  ---------------------------------------------------网络结构------------------------------------------------------- #
# A common Conv2D model
input_image = Input(shape=(None, None, 3))
x = Conv2D(64, (3, 3), activation='relu')(input_image)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = AveragePooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)   # 最后这里出来128个featuremap
"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.
the output of final model is the lengths of 10 Capsule, whose dim=16.
the length of Capsule is the proba,
so the problem becomes a 10 two-classification problem."""
x = Reshape((-1, 128))(x) # -1代表自动推断维度，最后得到的是（batch_size,推断维度，128)  (batch_size,input_num_capsule, input_dim_capsule)
# as first layer in a Sequential model
# model = Sequential()
# model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension
capsule = Capsule(CLASS_NUM, 16, 3, True)(x)    # 第一列是分类数
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)   # 已经是属于那一类为1其余为0的一个2289为的向量
print(output)

model = Model(inputs=input_image, outputs=output)

# we use a margin loss
model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
model.summary()
data_augmentation = True

#  -----------------------------------------------------导入数据 训练模型 保存------------------------------------- #
if __name__ == "__main__":
    epochs = 500
    batch_size = 32
    num_classes = CLASS_NUM
    img_height = IMG_HEIGHT
    img_width = IMG_WIDTH
   # num_images = 4880

    # 下面注释的是人工的数据载入和划分方式
    # num_images_per_finger = 8
    # # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # database_root = '/home/users/wdd/myDatabase/Finger_ROI/TSINGHUA_rawData_ROI'
    # # images = []
    # # labels = []
    # images = np.zeros((num_images, img_height, img_width, 1), dtype=np.int32)
    # labels = np.zeros((num_images, ), dtype=np.int32)
    #
    # img_index = 0
    # for root, dirs, files in os.walk(database_root, topdown=False):
    #     for name in files:
    #         print("image:", img_index, os.path.join(root, name))
    #         # images.append(read_image(os.path.join(root, name)))
    #         # labels.append(os.path.join(root, name).split("/")[-2])
    #         image = load_img(os.path.join(root, name), target_size=(img_height, img_width), grayscale=True,
    #                          interpolation='bilinear')
    #         image = np.expand_dims(image, axis=2)
    #         images[img_index] = image  # img_to_array(image)
    #         label = math.floor(img_index / num_images_per_finger)
    #         print("label:", label)
    #         labels[img_index] = label
    #         img_index += 1
    #
    # images, labels = shuffle(images, labels)
    # print("images:", images.shape)
    # print("labels:", labels.shape)
    # x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=30)
    # print(x_train.shape)
    # print(x_test.shape)
    # x_train = x_train.astype('float32')  # Copy of the array, cast to a specified type.
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # y_train = utils.to_categorical(y_train, num_classes)  # Converts a class vector (integers) to binary class matrix.
    # # # y_train = np.squeeze(y_train)  #  Remove single-dimensional entries from the shape of an array.
    # y_test = utils.to_categorical(y_test, num_classes)

    # we can compare the performance with or without data augmentation

    # if not data_augmentation:
    #     print('Not using data augmentation.')
    #     print(x_train.shape)
    #     print(y_train.shape)
    #     model.fit(
    #         x_train,
    #         y_train,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         validation_data=(x_test, y_test),
    #         shuffle=True)
    #
    #     model.get_config()
    #     file_path = '/home/users/wdd/capsNet_test/trained_model/capsNet_TH_noAugmentaion.h5'
    #     model.save(file_path)
    #     plot_model(model, to_file='/home/users/wdd/capsNet_test/trained_model/capsNet_TH_noAugmentaion.png')
    # else:
    #     print('Using real-time data augmentation.')
    #
    #     #  This will do preprocessing and realtime data augmentation:
    #
    #     datagen = ImageDataGenerator(
    #         featurewise_center=True,  # set input mean to 0 over the dataset
    #
    #         samplewise_center=True,  # set each sample mean to 0
    #
    #         featurewise_std_normalization=True,  # divide inputs by dataset std
    #
    #         samplewise_std_normalization=True,  # divide each input by its std
    #
    #         zca_whitening=False,  # apply ZCA whitening
    #
    #         zca_epsilon=1e-06,  # epsilon for ZCA whitening
    #
    #         rotation_range=15,  # randomly rotate images in 0 to 180 degrees
    #
    #         width_shift_range=10,  # randomly shift images horizontally
    #
    #         height_shift_range=5,  # randomly shift images vertically
    #
    #         shear_range=0.,  # set range for random shear
    #
    #         zoom_range=0.,  # set range for random zoom
    #
    #         channel_shift_range=0.,  # set range for random channel shifts
    #
    #         # set mode for filling points outside the input boundaries
    #
    #         fill_mode='nearest',
    #
    #         cval=0.,  # value used for fill_mode = "constant"
    #
    #         horizontal_flip=True,  # randomly flip images
    #
    #         vertical_flip=False,  # randomly flip images
    #
    #         # set rescaling factor (applied before any other transformation)
    #
    #         rescale=None,
    #
    #         # set function that will be applied on each input
    #
    #         preprocessing_function=None,
    #
    #         # image data format, either "channels_first" or "channels_last"
    #
    #         data_format=None
    #
    #         )
    #
    #     # Compute quantities required for feature-wise normalization
    #
    #     # (std, mean, and principal components if ZCA whitening is applied).
    #
    #     datagen.fit(x_train)
    #
    #     # Fit the model on the batches generated by datagen.flow().
    #
    #     model.fit_generator(
    #
    #         datagen.flow(x_train, y_train, batch_size=batch_size),
    #
    #         steps_per_epoch=len(x_train)/batch_size,
    #
    #         epochs=epochs,
    #
    #         validation_data=(x_test, y_test),
    #
    #         workers=4)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,
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
        '/home/wdxia/Finger_ROI_Database/Database',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        # color_mode='grayscale',
        class_mode='categorical')

    validation_generator = valid_datagen.flow_from_directory(
        '/home/wdxia/Finger_ROI_Database/Database_split_valid',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        # color_mode='grayscale',
        class_mode='categorical')

    modelcheckpoint = keras.callbacks.ModelCheckpoint(
        '/home/wdxia/network_test/model_trained/capsNet_result/FVcapsNet_28x28_{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', period=10)
    csv_logger = keras.callbacks.CSVLogger('/home/wdxia/network_test/model_trained/capsNet_result/training_28x28.log')


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
            # plt.show()
            savefig("/home/wdxia/network_test/model_trained/capsNet_result/capsNet_28x28_loss.jpg")


    loss_history = LossHistory()

    model.fit_generator(
        train_generator,
        steps_per_epoch=25734 // 32 + 1,  # 训练图像数/batch_size 3180 3180 // 32
        epochs=epochs,
        callbacks=[ csv_logger, loss_history],
        validation_data=validation_generator,
        validation_steps=7972 // 32 + 1)  # 测试图像数/batch_size 636  636 // 32

    loss_history.loss_plot('epoch')








