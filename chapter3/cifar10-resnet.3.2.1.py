"""Trains a ResNet on the CIFAR10 dataset.

ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

# Training params.
batch_size = 32
epochs = 200
data_augmentation = True
num_classes=10

#           |      |           | Orig Paper|           | Orig Paper|
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | sec/epoch
#           |      | %Accuracy | %Accuracy | %Accuracy | %Accuracy | GTX 1080Ti
# ResNet20  |  3   | 91.95     | 91.25     | -----     | -         | 58
# ResNet32  |  5   | 92.00     | 92.49     | -----     | -         | 96
# ResNet44  |  7   | 91.07     | 92.83     | -----     | -         | 128
# ResNet56  |  9   | 90.25     | 93.03     | 92.46     | -         | 163 (100)
# ResNet110 |  18  | 90.23     | 93.39     | 92.46     | 93.63     | 330 (180)
n = 3
depth = n * 6 + 2

# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Subtracting pixel mean improves accuracy
subtract_mean_pixel = True

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
# We assume data format "channels_last".
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

if K.image_data_format() == 'channels_first':
    img_rows = x_train.shape[2]
    img_cols = x_train.shape[3]
    channels = x_train.shape[1]
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channels = x_train.shape[3]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

# Normalize data.
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

if subtract_mean_pixel:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    x_train /= 128.
    x_test /= 128.
    # x_train_mean = np.mean(x_train, axis=0)
    # x_train -= x_train_mean
    # x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Learning rate scheduler - called every epoch as part of callbacks
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def conv_bn(inputs, num_filters=16, kernel_size=3, strides=1):
    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    return x

def conv_bn_relu(inputs, num_filters=16, kernel_size=3, strides=1):
    x = conv_bn(inputs=inputs,
                num_filters=num_filters,
                kernel_size=kernel_size,
                strides=strides)
    x = Activation('relu')(x)
    return x
    
def bn_relu_conv(inputs, num_filters=16, kernel_size=3, strides=1):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)
    return x
    
def resnet_v1(input_shape, depth, num_classes=10):
    # Start model definition.
    assert (depth - 2) % 6 == 0, 'depth should be 6n+2 (eg 20, 32, 44 in [a])'
    inputs = Input(shape=input_shape)
    num_filters = 16
    num_sub_blocks = int((depth - 2) / 6)

    x = conv_bn_relu(inputs=inputs)
    # Instantiate convolutional base (stack of blocks).
    for i in range(3):
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = conv_bn_relu(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = conv_bn(inputs=y,
                        num_filters=num_filters)
            if is_first_layer_but_not_first_block:
                x = conv_bn(inputs=x,
                            num_filters=num_filters,
                            kernel_size=1,
                            strides=strides)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = 2 * num_filters

    # Add classifier on top.
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v1_bottleneck(input_shape, depth, num_classes=10):
    # Start model definition.
    assert (depth - 2) % 9 == 0, 'depth should be 9n+2 (eg 56, 110 in [a])'
    inputs = Input(shape=input_shape)
    num_filters_in = 16
    num_filters_out = 64
    filter_multiplier = 4
    num_sub_blocks = int((depth - 2) / 9)

    x = conv_bn_relu(inputs=inputs)
    # Instantiate convolutional base (stack of blocks).
    for i in range(3):
        num_filters_out = num_filters_in * filter_multiplier 
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = conv_bn_relu(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides)
            y = conv_bn_relu(inputs=y,
                             num_filters=num_filters_in)
            y = conv_bn(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1)
            if j == 0:
                x = conv_bn(inputs=x,
                            num_filters=num_filters_out,
                            kernel_size=1,
                            strides=strides)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters_in = num_filters_out

    # Add classifier on top.
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2_bottleneck(input_shape, depth, num_classes=10):
    assert (depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 110 in [b])'
    inputs = Input(shape=input_shape)
    num_filters_in = 16
    num_filters_out = 64
    filter_multiplier = 4
    num_sub_blocks = int((depth - 2) / 9)

    x = Conv2D(num_filters_in,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)

    for i in range(3):
        if i > 0:
            filter_multiplier = 2
        num_filters_out = num_filters_in * filter_multiplier 

        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = bn_relu_conv(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides)
            y = bn_relu_conv(inputs=y,
                             num_filters=num_filters_in)
            y = bn_relu_conv(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1)
            if j == 0:
                x = Conv2D(num_filters_out,
                           kernel_size=1,
                           strides=strides,
                           padding='same',
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
    

if version == 2:
    model = resnet_v2_bottleneck(input_shape=input_shape, depth=depth)
    print('ResNet%d v2 with bottleneck layers' % depth)
else:
    if depth > 44:
        model = resnet_v1_bottleneck(input_shape=input_shape, depth=depth)
        print('ResNet%d v1' % depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)
        print('ResNet%d v1 with bottleneck layers' % depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_resnet_model.{epoch:02d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate decaying.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
# callbacks = [checkpoint, lr_reducer]

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (deg 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
