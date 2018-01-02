'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

import numpy as np
import matplotlib.pyplot as plt
#import numpy.polynomial.polynomial as poly

x = np.arange(-2.5, 2.5, 0.1)
c = [1, -0.2, -5, 0, 4]
d = [4, -0.6, -10, 0]
y = np.polyval(c, x)
z = np.polyval(d, x)
plt.xlabel('x')
plt.ylabel('y and dy/dx')
plt.plot(x, y, label="y=x^4 -0.2x^3 -5x^2 +4")
plt.plot(x, z, label="dy/dx ymin at x=-1.51 & 1.66")
plt.legend(loc=0)
plt.grid(b=True)
plt.show()
plt.close('all')
