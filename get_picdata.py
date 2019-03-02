from __future__ import division
import numpy as np
import os
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from keras import utils
from sklearn.utils import shuffle
from PIL import Image
import matplotlib.pyplot as plt


import math
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 128
num_classes = 610
img_height = 60
img_width = 120
num_images = 4880
num_images_per_finger = 8
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
database_root = '/home/users/wdd/myDatabase/Finger_ROI/TSINGHUA_rawData_ROI'
# images = []
# labels = []
images = np.zeros((num_images, img_height, img_width, 1), dtype=np.int32)
# images = np.zeros((num_images,img_width, img_height,1), dtype=np.int32)
labels = np.zeros((num_images,), dtype=np.int32)

img_index = 0
for root, dirs, files in os.walk(database_root, topdown=False):
    for name in files:
        print("image:", img_index, os.path.join(root, name))
        # images.append(read_image(os.path.join(root, name)))
        # labels.append(os.path.join(root, name).split("/")[-2])
        image = load_img(os.path.join(root, name), target_size=(img_height, img_width), grayscale=True)
        plt.figure("test")
        plt.imshow(image)
        plt.show()
        images[img_index] = image  # img_to_array(image)
        label = math.floor(img_index/num_images_per_finger)
        print("label:", label)
        labels[img_index] = label
        img_index += 1

images, labels = shuffle(images, labels)
print("images:", images.shape)
print("labels:", labels.shape)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=30)
print(x_train.shape)
print(x_test.shape)
x_train = x_train.astype('float32')  #  Copy of the array, cast to a specified type.
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)   #  Converts a class vector (integers) to binary class matrix.
# # y_train = np.squeeze(y_train)  #  Remove single-dimensional entries from the shape of an array.
y_test = utils.to_categorical(y_test, num_classes)
# # y_test = np.squeeze(y_test)
x_train.tofile('train_TH.bin')
x_test.tofile('vali_TH.bin')
y_train.tofile('train_label_TH.bin')
y_test.tofile('vali_label_TH.bin')
print('ok')
