'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

import numpy as np
import matplotlib.pyplot as plt

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
