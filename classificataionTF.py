"""
Classification
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

def sigmoid(z):
    return 1/(1+ np.exp(-z))

sampleZ = np.linspace(-10, 10, 100)
sampleA = sigmoid(sampleZ)

plt.plot(sampleZ, sampleA)
plt.show()

data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
print(data)
print(type(data))
feature = data[0]
label = data[1]
print(feature[:,0])
print(feature[:,1])
plt.scatter(feature[:,0], feature[:,1], c=label)

x = np.linspace(-1, 10, 20)
y = -x + 5
plt.plot(x, y)
plt.show()
