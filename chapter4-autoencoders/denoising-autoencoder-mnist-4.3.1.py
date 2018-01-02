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
import matplotlib.pyplot as plt

# MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_noisy = x_train +\
                np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_test_noisy = x_test +\
               np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
dropout = 0.4
filters = 16
latent_dim = 16

# Build the Autoencoder model
# First build the Encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Stack of BN-ReLU-Conv2D-MaxPooling blocks
for i in range(2):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    filters = filters * 2
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               padding='same')(x)
    x = MaxPooling2D()(x)

# Shape info needed to build decoder model
shape = x.shape.as_list()

# Generate a 16-dim latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# Instantiate Encoder model
encoder = Model(inputs, latent, name='encoder')

# Build the Decoder model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# Stack of BN-ReLU-Transposed Conv2D-UpSampling blocks
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

# Instantiate Decoder model
decoder = Model(latent_inputs, outputs, name='decoder')

# Autoencoder = Encoder + Decoder
# Instantiate Autoencoder model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autodecoder')

# Mean Square Error (MSE) loss function, Adam optimizer
autoencoder.compile(loss='mse', optimizer='adam')

# Train the autoencoder for 1 epoch
autoencoder.fit(x_train_noisy, x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=1, batch_size=batch_size)

# Predict the Autoencoder output from test data
x_decoded = autoencoder.predict(x_test_noisy)

# Display the 1st 8 input and decoded images
imgs = np.concatenate([x_test_noisy[:8], x_decoded[:8]])
imgs = imgs.reshape((4, 4, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.savefig('noisyinput_and_decoded.png')
