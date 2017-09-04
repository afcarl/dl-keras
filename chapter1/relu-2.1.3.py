'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.maximum(0,x)
plt.xlabel('x')
plt.ylabel('relu')
plt.plot(x, y)
plt.grid(b=True)
plt.show()
plt.close('all')
