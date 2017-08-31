'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

import numpy as np
import matplotlib.pyplot as plt
#import numpy.polynomial.polynomial as poly

x = np.arange(-1, 2, 0.1)
c = [1, -1, -1]
d = [2,-1]
y = np.polyval(c,x)
z = np.polyval(d,x)
plt.xlabel('x')
plt.ylabel('y and dy/dx')
plt.plot(x, y, label="y=x^2 -x -1")
plt.plot(x, z, label="dy/dx ymin at x=0.5")
plt.legend(loc=0)
plt.grid(b=True)
plt.show()
plt.close('all')
