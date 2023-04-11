# Quick function to load and plot best solution from test.data files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Data For Plot 1 and 2 (The scatter graph)
fname_test = "results/raw/uniform/test.dat"
train_state = np.loadtxt(fname_test, delimiter=" ", skiprows=1)

# # Import Data For Plot3 (2D Contour)
# fname_regular = ""
# reg_stuff = np.loadtxt(fname_regular, delimiter=" ")

# Plot 1 (Distribution of points)
plt.figure(1)
ax = plt.axes()
ax.scatter(
    train_state[:, 0],
    train_state[:, 1],
    marker=".",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")

# Plot 2 (Coloured 3D Scatter graph) #
plt.figure(2)
ax = plt.axes(projection=Axes3D.name)
ax.scatter3D(
    train_state[:, 0],
    train_state[:, 1],
    train_state[:, 2],
    c=train_state[:, 2],
    marker=".",
    s=20,
    cmap="coolwarm",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_title("$u(x,t)$")
# plt.show()

data = np.load("src/burgers/Burgers.npz")
t, x, exact = data["t"], data["x"], data["usol"].T
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
y = exact.flatten()[:, None]
print(exact.shape)
print(y.shape)
y_back = y.reshape((100, 256))
print(y_back.shape)
fig = plt.figure(figsize=(10, 8), dpi=50)
plt.pcolormesh(tt, xx, y_back, cmap="rainbow")
plt.xlabel("t")
plt.ylabel("x")
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label("u(t,x)")
cbar.mappable.set_clim(-1, 1)
plt.show()
