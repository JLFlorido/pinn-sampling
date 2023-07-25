# Quick function to load and plot best solution from test.data files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Data For Plot 1 and 2 (The scatter graph)
fname_test = "results/raw/sol-sampling-test/residualsy.txt"
# fname_test = "results/raw/test_and_train/test_Ham2k.dat"
# fname_test2 = "results/raw/test_and_train/test_Ham5k.dat"
# fname_test3 = "results/raw/test_and_train/test_Ham10k.dat"
# fname_test = "results/raw/test_and_train/test_RAR-D_default.dat"
# fname_test = "results/raw/test_and_train/test_RAD_default.dat"


test_state = np.loadtxt(fname_test, delimiter=" ", skiprows=1)
fname_train = "results/raw/sol-sampling-test/residualsy.txt"
# fname_train = "results/raw/test_and_train/train_Ham2k.dat"
# fname_train2= "results/raw/test_and_train/train_Ham5k.dat"
# fname_train3= "results/raw/test_and_train/train_Ham10k.dat"
# fname_train = "results/raw/test_and_train/train_RAR-D_default.dat"
# fname_train = "results/raw/test_and_train/train_RAD_default.dat"

train_state = np.loadtxt(fname_train, delimiter=" ", skiprows=1)
# train_state2 = np.loadtxt(fname_train2, delimiter=" ", skiprows=1)
# train_state3 = np.loadtxt(fname_train3, delimiter=" ", skiprows=1)

# Plot 1 (Distribution of points)
plt.figure(1)
ax = plt.axes()
ax.scatter(
    train_state[:, 1],
    train_state[:, 0],
    marker=".",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_ylim([-1, 1])
ax.set_xlim([0, 1])

# # Plot 1 but with 3 subplots for the uniform cases.
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
# # Plot the first subplot
# ax1.scatter(train_state[:, 1], train_state[:, 0], marker=".")
# ax1.set_xlabel("$x$")
# ax1.set_ylabel("$t$")
# ax1.set_title("2k Points")
# ax1.set_ylim([-1, 1])
# ax1.set_xlim([0, 1])
# # Plot the second subplot
# ax2.scatter(train_state2[:, 1], train_state2[:, 0], marker=".")
# ax2.set_xlabel("$x$")
# ax2.set_ylabel("$t$")
# ax2.set_title("5k Points")
# ax2.set_ylim([-1, 1])
# ax2.set_xlim([0, 1])
# # Plot the third subplot
# ax3.scatter(train_state3[:, 1], train_state3[:, 0], marker=".")
# ax3.set_xlabel("$x$")
# ax3.set_ylabel("$t$")
# ax3.set_title("10k Points")
# ax3.set_ylim([-1, 1])
# ax3.set_xlim([0, 1])

# Plot 2 (Coloured 3D Scatter graph) #
plt.figure(2)
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
