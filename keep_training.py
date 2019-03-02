from keras.models import Model
from keras.models import load_model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import keras
import os
from keras.optimizers import SGD
from keras import backend as K

# 载入训练过的模型
model = load_model('/home/wdxia/network_test/model_trained/FVdenseNet.hdf5')
CLASS_NUM = 2289

# 继续训练
if __name__ == "__main__":
    epochs = 100
    batch_size = 32
    num_classes = CLASS_NUM
    IMG_HEIGHT = 40
    IMG_WIDTH = 80
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
        samplewise_center=False,  # set each sample mean to 0
        rotation_range=15,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=10,  # randomly shift images horizontally
        height_shift_range=5,  # randomly shift images vertically
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        vertical_flip=False,  # randomly flip images
    )

    valid_datagen = ImageDataGenerator(rescale=1. / 255)

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

    # modelcheckpoint = keras.callbacks.ModelCheckpoint(
    #     '/home/wdxia/network_test/model_trained/FVcapsNet_{epoch:02d}-{val_loss:.2f}.hdf5',
    #     monitor='val_loss', verbose=0, save_best_only=True,
    #     save_weights_only=False, mode='auto', period=20)
    csv_logger = keras.callbacks.CSVLogger('/home/wdxia/network_test/model_trained/denseNet_training.log')

    # 记录损失历史(自定义的回调函数）
    class LossHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    loss_history = LossHistory()

    model.fit_generator(
        train_generator,
        steps_per_epoch=25734 // 32 + 1,  # 训练图像数/batch_size 3180 3180 // 32
        epochs=epochs,
        callbacks=[ csv_logger, loss_history],
        validation_data=validation_generator,
        validation_steps=7972 // 32 + 1)  # 测试图像数/batch_size 636  636 // 32

    print(loss_history)
    model.save('/home/wdxia/network_test/model_trained/FVdenseNet.hdf5')

