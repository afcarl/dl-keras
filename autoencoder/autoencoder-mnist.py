'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependencies: keras 
Usage: python3 <this file>
'''

import numpy as np
import keras
from keras.layers import Activation, Dense, Dropout, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
import math
from PIL import Image
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_labels = np.amax(y_train) + 1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
dropout = 0.4
n_filters = 32

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

inputs = Input(shape=input_shape)
x = inputs
filters = n_filters
for i in range(2):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               padding='same')(x)
    x = MaxPooling2D()(x)
    filters = filters * 2

x = Flatten()(x)
x = Dense(16)(x)
x = Dense(7*7*64)(x)
x = Reshape((7, 7, 64))(x)
filters = 64
for i in range(2):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
               padding='same')(x)
    x = UpSampling2D()(x)
    filters = int(filters / 2)

x = Conv2DTranspose(filters=1, kernel_size=kernel_size,
               padding='same')(x)

outputs = Activation('sigmoid')(x)

model = Model(inputs, outputs)
model.summary()

model.compile(loss='mse',
              optimizer='adam', metrics=['accuracy'])
model.fit(x_train, x_train, 
          validation_data=(x_test, x_test),
          epochs=1, batch_size=batch_size)

x_dec = model.predict(x_test)
img = combine_images(np.concatenate([x_test[:8],x_dec[:8]]))
image = img * 255
image = image.astype(np.uint8)
image = Image.fromarray(image)
image.save("input_and_recon.png")
image.show()
