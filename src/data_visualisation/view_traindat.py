# Quick function to load and plot best solution from test.data files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Data For Plot 1 and 2 (The scatter graph)
fname_test = "results/raw/test_uniform.dat"
test_state = np.loadtxt(fname_test, delimiter=" ", skiprows=1)
fname_train = "results/raw/train_uniform.dat"
train_state = np.loadtxt(fname_test, delimiter=" ", skiprows=1)

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

plt.show()
