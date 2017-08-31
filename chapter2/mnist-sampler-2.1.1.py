'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))
unique, counts = np.unique(y_test, return_counts=True)
print("Test labels: ", dict(zip(unique, counts)))

indexes = np.random.randint(0,x_train.shape[0], size=10)
images = x_train[indexes]
labels = y_train[indexes]
for i in range(len(indexes)):
    filename = "mnist%d.png" % labels[i]
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.savefig(filename)
    plt.show()

plt.close('all')
