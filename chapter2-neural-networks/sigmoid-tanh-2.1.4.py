'''
Utility for plotting sigmoid and tanh functions

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
y = 1.0 / (1.0 + np.exp(-x))
z = np.tanh(x)
plt.xlabel('x')
plt.ylabel('sigmoid and tanh')
plt.plot(x, y, label="sigmoid")
plt.plot(x, z, label="tanh")
plt.legend(loc=0)
plt.grid(b=True)
plt.show()
plt.close('all')
