# Quick function to load and plot best solution from test.data files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Data For Plot 1 and 2 (The scatter graph)
fname_test = "test_RAR-D_2000_Try2.dat" #/results/raw/test_RAD_100res.dat"
test_state = np.loadtxt(fname_test, delimiter=" ", skiprows=1)
fname_train = "train_RAD_100res.dat" #/results/raw/test_RAD_100res.dat"
train_state = np.loadtxt(fname_test, delimiter=" ", skiprows=1)

# Plot 1 (Distribution of points)
plt.figure(1)
ax = plt.axes()
ax.scatter(
    test_state[:, 0],
    test_state[:, 1],
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

# Plot 3 (Coloured 3D Scatter graph) #
plt.figure(3)
ax = plt.axes(projection=Axes3D.name)
ax.scatter3D(
    test_state[:, 0],
    test_state[:, 1],
    test_state[:, 2],
    c=test_state[:, 2],
    marker=".",
    s=20,
    cmap="coolwarm",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_title("$u(x,t)$")

plt.show()
