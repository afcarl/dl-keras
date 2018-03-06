'''
Utility for plotting a 2nd deg polynomial and
its derivative

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
x = np.arange(-1, 2, 0.1)
c = [1, -1, -1]
d = [2, -1]
y = np.polyval(c, x)
z = np.polyval(d, x)
plt.xlabel('x')
plt.ylabel('y and dy/dx')
plt.plot(x, y, label="y=x^2 -x -1")
plt.plot(x, z, label="dy/dx ymin at x=0.5")
plt.legend(loc=0)
plt.grid(b=True)
plt.show()
plt.close('all')
