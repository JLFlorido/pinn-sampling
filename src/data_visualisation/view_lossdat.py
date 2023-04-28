# Quick function to load and plot best solution from test.data files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Loss Data
fname_loss1 = "results/raw/loss_RAR-D_2000_100rounds.dat"
fname_loss2 = "results/raw/loss_RAR-D_2000_Try2.dat"
fname_loss3 = "results/raw/loss_RAR-D_2000_a.dat"
loss_100rounds_bigAdam = np.loadtxt(fname_loss1, delimiter=" ", skiprows=1)
loss_10rounds_smallAdam = np.loadtxt(fname_loss2, delimiter=" ", skiprows=1)
loss_10rounds_bigAdam = np.loadtxt(fname_loss3, delimiter=" ", skiprows=1)

# Data for loss and l2 over iterations
fname_rad_loss = "results/raw/loss_RAD_default.dat"
loss_rad = np.loadtxt(fname_rad_loss, delimiter=" ", skiprows=1)
fname_uni_loss = "results/raw/loss_Ham_noidea.dat"
loss_uniform = np.loadtxt(fname_uni_loss, delimiter=" ", skiprows=1)

# ------------------------------------------------------- Plot 1 --------------------------------------------------------
plt.figure(0)
ax = plt.axes()
ax.plot(loss_uniform[:, 0], loss_uniform[:, 1], label="Uniform")
ax.plot(loss_rad[:, 0], loss_rad[:, 2], label="RAD Test")
ax.plot(
    loss_10rounds_bigAdam[:, 0],
    loss_10rounds_bigAdam[:, 2],
    label="RAR-D, 10 Re-Samples, 15k Adam",
)
ax.plot(
    loss_100rounds_bigAdam[:, 0],
    loss_100rounds_bigAdam[:, 2],
    label="RAR-D, 100 Re-Samples, 15k Adam",
)

ax.set_yscale("log")
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_title("Comparing all methods so far")
ax.legend()

# ------------------------------------------------------- Plot 2 --------------------------------------------------------
plt.figure(2)
ax = plt.axes()
ax.plot(loss_uniform[:, 0], loss_uniform[:, 1], label="Uniform")
ax.plot(loss_rad[:, 0], loss_rad[:, 2], label="RAD Test")
ax.plot(
    loss_10rounds_bigAdam[:, 0],
    loss_10rounds_bigAdam[:, 2],
    label="RAR-D, 10 Re-Samples, 15k Adam",
)
ax.plot(
    loss_100rounds_bigAdam[:, 0],
    loss_100rounds_bigAdam[:, 2],
    label="RAR-D, 100 Re-Samples, 15k Adam",
)

ax.set_yscale("log")
ax.set_xlim(0, 50000)
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_title("Comparing all methods so far")
ax.legend()

# ------------------------------------------------------- Plot 3 --------------------------------------------------------
plt.figure(1)
ax = plt.axes()
ax.plot(
    loss_10rounds_smallAdam[:, 0],
    loss_10rounds_smallAdam[:, 2],
    label="1 500 Adam Steps",
)
ax.plot(
    loss_10rounds_bigAdam[:, 0], loss_10rounds_bigAdam[:, 2], label="15 000 Adam Steps"
)
# ax.plot(loss_100rounds_bigAdam[:, 0], loss_100rounds_bigAdam[:, 2], label="100 Big")

ax.set_yscale("log")
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_title("Comparing RAR-D settings via test loss")
ax.legend()

plt.show()
