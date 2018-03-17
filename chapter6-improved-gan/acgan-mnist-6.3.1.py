'''Trains ACGAN on MNIST using Keras

This version of CGAN is similar to DCGAN. The difference mainly
is that the z-vector of geneerator is conditioned by a one-hot label
to produce specific fake images. The discriminator is trained to
discriminate real from fake images that are conditioned on
specific one-hot labels.

[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

[2] Odena, Augustus, Christopher Olah, and Jonathon Shlens. 
"Conditional image synthesis with auxiliary classifier gans." 
arXiv preprint arXiv:1610.09585 (2016).
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical

import numpy as np
import math
import matplotlib.pyplot as plt


def generator(inputs, y_labels, image_size):
    """Build a Generator Model

    Inputs are concatenated before Dense layer.
    Stacks of BN-ReLU-Conv2DTranpose to generate fake images
    Output activation is sigmoid instead of tanh in orig DCGAN.
    Sigmoid converges easily.

    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        y_labels (Layer): Input layer for one-hot vector to condition
            the inputs
        image_size: Target size of one side (assuming square image)

    # Returns
        Model: Generator Model
    """
    image_resize = image_size // 4
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = keras.layers.concatenate([inputs, y_labels], axis=1)
    x = Dense(image_resize * image_resize * layer_filters[0])(x)
    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)

    x = Activation('sigmoid')(x)
    generator = Model([inputs, y_labels], x, name='generator')
    return generator


def discriminator(inputs, num_labels, image_size):
    """Build a Discriminator Model

    Inputs are concatenated after Dense layer.
    Stacks of LeakyReLU-Conv2D to discriminate real from fake
    The network does not converge with BN so it is not used here
    unlike in DCGAN paper.

    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)
        y_labels (Layer): Input layer for one-hot vector to condition
            the inputs
        image_size: Target size of one side (assuming square image)

    # Returns
        Model: Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs
    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    y_class = Dense(layer_filters[-2])(x)
    y_class = Dense(num_labels)(x)
    y_class = Activation('softmax', name='label')(y_class)

    y_source = Dense(1)(x)
    y_source = Activation('sigmoid', name='source')(y_source)
    discriminator = Model(inputs, [y_source, y_class], name='discriminator')
    return discriminator


def train(models, data, params):
    """Train the Discriminator and Adversarial Networks

    Alternately train Discriminator and Adversarial networks by batch
    Discriminator is trained first with properly real and fake images
    Adversarial is trained next with fake images pretending to be real
    Discriminator inputs are conditioned by train labels for real images,
    and random labels for fake images
    Adversarial inputs are conditioned by random labels
    Generate sample images per save_interval

    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        data (list): x_train, y_train data
        params (list): Network parameters

    """
    generator, discriminator, adversarial = models
    x_train, y_train = data
    batch_size, latent_size, train_steps, num_labels = params
    save_interval = 500
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    for i in range(train_steps):
        # Pick random real images and their labels
        rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
        real_images = x_train[rand_indexes, :, :, :]
        real_labels = y_train[rand_indexes, :]
        # Generate fake images and their labels
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                           batch_size)]

        fake_images = generator.predict([noise, fake_labels])
        x = np.concatenate((real_images, fake_images))

        y_labels = np.concatenate((real_labels, fake_labels))

        # Label real and fake images
        y = np.ones([2 * batch_size, 1])
        y[batch_size:, :] = 0
        # Train the Discriminator network
        metrics = discriminator.train_on_batch(x, [y, y_labels])
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, accuracy)

        # Generate fake images and their labels
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                           batch_size)]
        # Label fake images as real
        y = np.ones([batch_size, 1])
        # Train the Adversarial network
        metrics = adversarial.train_on_batch([noise, fake_labels], [y, fake_labels])
        loss = metrics[0]
        accuracy = metrics[1]
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, accuracy)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False
            plot_images(generator,
                        noise_input=noise_input,
                        noise_class=noise_class,
                        show=show,
                        step=(i + 1))


def plot_images(generator,
                noise_input,
                noise_class,
                show=False,
                step=0):
    """Generate fake images and plot them

    For visualization purposes, generate fake images given labels
    then plot them in a square grid

    # Arguments
        generator (Model): The Generator Model for fake images generation
        noise_input (ndarray): Array of z-vectors
        noise_class (ndarray): Array of labels
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images

    """
    filename = "mnist_cgan_%d.png" % step
    images = generator.predict([noise_input, noise_class])
    print("Labels: ", np.argmax(noise_class, axis=1))
    plt.figure(figsize=(2.4, 2.4))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = images[i, :, :, :]
        image = np.reshape(image, [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


# MNIST dataset
(x_train, y_train), (_, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255

num_labels = np.amax(y_train) + 1
y_train = to_categorical(y_train)

# Network parameters
# The latent or z vector is 100-dim
latent_size = 100
batch_size = 64
train_steps = 40000
lr = 0.0002
decay = 6e-8
input_shape = (image_size, image_size, 1)
label_shape = (num_labels, )

# Build Discriminator Model
inputs = Input(shape=input_shape, name='discriminator_input')

discriminator = discriminator(inputs, num_labels, image_size)
# [1] uses Adam, but discriminator converges easily with RMSprop
optimizer = RMSprop(lr=lr, decay=decay)
# loss = {'source': 'binary_crossentropy', 'label': 'categorical_crossentropy'}
loss = ['binary_crossentropy', 'categorical_crossentropy']
# loss_weights = {'source': 1., 'label': 1,}
discriminator.compile(loss=loss,
                      # loss_weights=loss_weights,
                      optimizer=optimizer,
                      metrics=['accuracy'])
discriminator.summary()

# Build Generator Model
input_shape = (latent_size, )
inputs = Input(shape=input_shape, name='z_input')
y_labels = Input(shape=label_shape, name='y_labels')
generator = generator(inputs, y_labels, image_size)
generator.summary()

# Build Adversarial Model = Generator + Discriminator
optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
discriminator.trainable = False
adversarial = Model([inputs, y_labels],
                    discriminator(generator([inputs, y_labels])),
                    name='acgan')
# loss = ['binary_crossentropy', 'categorical_crossentropy']
adversarial.compile(loss=loss,
                    optimizer=optimizer,
                    metrics=['accuracy'])
adversarial.summary()

# Train Discriminator and Adversarial Networks
models = (generator, discriminator, adversarial)
data = (x_train, y_train)
params = (batch_size, latent_size, train_steps, num_labels)
train(models, data, params)
