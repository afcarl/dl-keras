'''
Utility for plotting relu function
Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('grayscale')
x = np.arange(-5, 5, 0.1)
y = np.maximum(0,x)
plt.xlabel('x')
plt.ylabel('relu')
plt.plot(x, y)
plt.grid(b=True)
plt.show()
plt.close('all')
