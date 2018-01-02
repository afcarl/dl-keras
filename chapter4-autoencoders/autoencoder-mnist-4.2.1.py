'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependencies: keras with tensorflow backend
Usage: python3 <this file>
'''

import numpy as np
import keras
from keras.layers import Activation, Dense, Dropout, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras.utils import plot_model
import math
from PIL import Image
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
dropout = 0.4
n_filters = 16
latent_dim = 16

# utility function for combining several images into
# matrix of images for easy visualization
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
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image

# Build the Autoencoder model
# First build the Encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
filters = n_filters
# stacks of BN-ReLU-Conv2D-MaxPooling
for i in range(2):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    filters = filters * 2
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               padding='same')(x)
    x = MaxPooling2D()(x)

# shape info needed to create decoder model
shape = x.shape.as_list()

# create a 16-dim latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# instatiate encoder model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# Build the Decoder model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# stacks of BN-ReLU-Transposed Conv2D-UpSampling
for i in range(2):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                        padding='same')(x)
    x = UpSampling2D()(x)
    filters = int(filters / 2)

x = Conv2DTranspose(filters=1, kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='decoder')
autoencoder.summary()
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)

# MSE loss function, Adam optimizer
autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.fit(x_train, x_train,
                validation_data=(x_test, x_test),
                epochs=1, batch_size=batch_size)

# predict the autoencoder output fr test data
x_dec = autoencoder.predict(x_test)

# combine input and decoded images then plot
img = combine_images(np.concatenate([x_test[:8], x_dec[:8]]))
image = img * 255
image = image.astype(np.uint8)
image = Image.fromarray(image)
image.save("input_and_decoded.png")
image.show()

