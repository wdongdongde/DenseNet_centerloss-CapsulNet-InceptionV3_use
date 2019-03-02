from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 256
num_classes = 10
epochs = 4 #  所有训练样本通过网络一次

img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()  #mist.py文件里面定义了load_data()函数

if(K.image_data_format()=='channels_first'):
    x_train = x_train.reshape(x_train.shape[0],1,img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0],1,img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0],  img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#  字段类型的转换
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#  数据中的每个像素取值转换到0-1之间
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#  将标注的0-9数值转换为一个长度为10的one-hot编码（标签一般需要转one-hot）
#  从tutorials中导入则无需以下两步
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#  下面开始搭建模型的架构，这里使用sequential

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#  dropout将在训练过程中每次更新参数时按一定的概率随机断开输入神经元，用于防止过拟合
model.add(Dropout(0.25))
#  flatten 将多维输入一维化，常用在从卷积层到全连接层的过渡
model.add(Flatten())
#  Dense层即为全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


#  编译用来配置模型的学习过程，下面包括交叉熵损失函数 Adadelta优化器，指标列表在分类问题上一般设置为metrics=['accuracy']
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

#  fit函数用来指定模型训练的epoch数
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
#  fit会显示用多少张图片训练和验证，batch_size=256,则将6000张图分成6000/256份，每训练一个batch有一个损失和准确率acc，
#  全部6000个样本训练完后会计算一个val_loss和val_acc  ，即处理完一次所有的训练数据后再去测试


score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])   #  验证集最后得到的损失
print('test accuracy:', score[1])  #  验证集最后得到的准确率


