'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1,1,0.2)
y = 2*x + 3
plt.xlabel('x')
plt.ylabel('y=f(x)')

noise = np.random.uniform(-0.1, 0.1, y.shape)
yn = y + noise

plt.ylabel('y and yn')
plt.plot(x, y, 'o-', label="y")
plt.plot(x, yn, 's-', label="yn = y + noise")
plt.legend(loc=0)

# plt.plot(x, y, 'bo-')
# plt.plot(x, yn, 'gs-')
plt.grid(b=True)
plt.show()
plt.close('all')
