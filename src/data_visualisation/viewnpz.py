from numpy import load
import numpy as np

# data = load("src/burgers/Burgers.npz")
# lst = data.files
# for item in lst:
#     print(item)
#     print(data[item].shape)

data = np.load("src/burgers/Burgers.npz")
t, x, exact = data["t"], data["x"], data["usol"].T
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
y = exact.flatten()[:, None]
print(xx.shape)
print(X.shape)
print(y.shape)
print(X[:, 1])

