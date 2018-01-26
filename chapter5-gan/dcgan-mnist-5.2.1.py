'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras

import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image


def generator(inputs, image_size):
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)
        if filters == layer_filters[-1]:
            x = Activation('sigmoid')(x)
        else:
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

    generator = Model(inputs, x, name='generator')
    return generator

def discriminator(inputs):
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]
    strides = 2
    
    x = inputs
    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)
        # if filters != layer_filters[0]:
        #    x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs, x, name='discriminator')
    return discriminator


def train(models,
          x_train,
          batch_size=128,
          train_steps=5000,
          latent_size=100):
    generator, discriminator, adversarial = models
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    for i in range(train_steps):
        random_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
        train_images = x_train[random_indexes, :, :, :]
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_images = generator.predict(noise)
        x = np.concatenate((train_images, fake_images))
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0
        metrics = discriminator.train_on_batch(x, y)
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, accuracy)

        y = np.ones([batch_size, 1])
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        metrics = adversarial.train_on_batch(noise, y)
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%s  [adversarial loss: %f, acc: %f]" % (log, loss, accuracy)
        print(log)
        if (i+1)%save_interval==0:
            if (i+1)==train_steps:
                save2file = False
            else:
                save2file = True
            plot_images(generator,
                        noise_input=noise_input,
                        save2file=save2file,
                        step=(i+1))


def plot_images(generator,
                noise_input,
                save2file=True,
                step=0):
     filename = "mnist_dcgan_%d.png" % step
     images = generator.predict(noise_input)
     plt.figure(figsize=(10,10))
     num_images = images.shape[0]
     image_size = images.shape[1]
     rows = int(math.sqrt(noise_input.shape[0]))
     for i in range(num_images):
        plt.subplot(rows, rows, i+1)
        image = images[i, :, :, :]
        image = np.reshape(image, [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
     plt.tight_layout()
     if save2file:
        plt.savefig(filename)
        plt.close('all')
     else:
        plt.show()


latent_size = 100
# MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255

input_shape = (image_size, image_size, 1)
inputs = Input(shape=input_shape, name='discriminator_input')
discriminator = discriminator(inputs)
optimizer = RMSprop(lr=0.0002, decay=6e-8)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
discriminator.summary()

input_shape = (latent_size, )
inputs = Input(shape=input_shape, name='z_input')
generator = generator(inputs, image_size)
generator.summary()

optimizer = RMSprop(lr=0.0001, decay=3e-8)
adversarial = Model(inputs, discriminator(generator(inputs)), name='dcgan')
adversarial.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
adversarial.summary()

models = (generator, discriminator, adversarial)
train(models, x_train, latent_size=latent_size)
