# Quick function to load and plot best solution from test.data files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Data For Plot 1 and 2 (The scatter graph)
fname_test = "results/raw/test_uniform.dat"
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

# Data for loss and l2 over iterations
fname_traind = "results/raw/train_RAD_default.dat"
train_state = np.loadtxt(fname_traind, delimiter=" ", skiprows=1)
fname_rad_loss = "results/raw/loss_RAD_default.dat"
loss_rad = np.loadtxt(fname_rad_loss, delimiter=" ", skiprows=1)
fname_uni_loss = "results/raw/loss_uniform.dat"
loss_uniform = np.loadtxt(fname_uni_loss, delimiter=" ", skiprows=1)
plt.figure(3)
ax = plt.axes()
ax.plot(
    loss_rad[:, 0],
    loss_rad[:, 1],
    loss_rad[:, 0],
    loss_rad[:, 2],
    loss_uniform[:, 0],
    loss_uniform[:, 1],
    loss_uniform[:, 0],
    loss_uniform[:, 2],
)
ax.set_yscale("log")
plt.figure(4)
ax = plt.axes()
ax.plot(
    loss_uniform[:, 0],
    loss_uniform[:, 1],
    loss_uniform[:, 0],
    loss_uniform[:, 2],
)
ax.set_yscale("log")
# ax.plot()

# plt.figure(4)

plt.show()
