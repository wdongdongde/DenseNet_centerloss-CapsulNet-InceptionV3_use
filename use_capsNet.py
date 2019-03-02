
from keras.preprocessing import image
from keras.models import load_model
from keras.models import save_model
from keras.models import Model
import numpy as np
import os
from keras.layers import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])
        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs, (batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if (K.backend() == 'theano'):
                    o = K.sum(o, axis=1)
        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = {'num_capsule': self.num_capsule, 'dim_capsule':self.dim_capsule, 'routings':self.routings, 'share_weights':self.share_weights, 'activation':self.activation}
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


model = load_model('/home/wdxia/network_test/model_trained/FVcapsNet_test.hdf5', custom_objects={'Capsule':Capsule, 'margin_loss':margin_loss,'squash':squash,'softmax':softmax})
img_path = '/home/wdxia/Finger_ROI_Database/Database/001_1/01_ROI.jpg'
img = image.load_img(img_path, target_size=(28, 28))  # Loads an image into PIL format
x = image.img_to_array(img)  # Converts a PIL Image instance to a Numpy array.
x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
features = model.predict(x)
print(features)
