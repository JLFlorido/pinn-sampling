"""
read_groundtruth.py
Reads the groundtruth solution .npz file and prints the shape of it. Excerpt from default code.
This was a test to better understand the shape of the variable.
"""
from numpy import load
import numpy as np

data = np.load("src/burgers/Burgers.npz")
t, x, exact = data["t"], data["x"], data["usol"].T
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
y = exact.flatten()[:, None]
print(xx.shape)
print(X.shape)
print(y.shape)
print(X[:, 1])

