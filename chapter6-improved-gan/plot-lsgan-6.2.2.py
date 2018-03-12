'''
Utility for plotting gaussian distributions
with different variance

Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

import numpy as np
import matplotlib.pyplot as plt

want_noise = False
# grayscale plot, comment if color is wanted
plt.style.use('grayscale')

mu = [0, 0]
cov = [[25, 0], [0, 25]]
x, y = np.random.multivariate_normal(mu, cov, 100).T
plt.plot(x, y, '+', label="Fake samples")

mu = [0, 0]
cov = [[2, 0], [0, 2]]
x, y = np.random.multivariate_normal(mu, cov, 100).T
plt.plot(x, y, 'o', label="Real samples")

x = np.linspace(-15, 15)
y = x
plt.plot(x, y, '--', label="Sigmoid decision boundary")

x = np.linspace(-15, 15)
y = np.zeros(x.shape) 
plt.plot(x, y, '-', label="Least squares decision boundary")

plt.legend(loc=0)
plt.savefig("lsgan.png")
plt.show()
plt.close('all')
