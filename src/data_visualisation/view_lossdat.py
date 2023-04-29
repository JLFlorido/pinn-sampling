# Quick function to load and plot best solution from test.data files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Loss Data
fname_loss1 = "results/raw/loss/loss_RAD_default.dat"
fname_loss2 = "results/raw/loss/loss_RAD_RandomInit.dat"
fname_loss3 = "results/raw/loss/loss_RAD_2kSamplePeriod.dat"
loss_1 = np.loadtxt(fname_loss1, delimiter=" ", skiprows=1)
loss_2 = np.loadtxt(fname_loss2, delimiter=" ", skiprows=1)
loss_3 = np.loadtxt(fname_loss3, delimiter=" ", skiprows=1)

# More imports if necessary
# fname_loss4 = "results/raw/.dat"
# loss_4 = np.loadtxt(fname_loss4, delimiter=" ", skiprows=1)
# fname_loss5 = "results/raw/.dat"
# loss_5 = np.loadtxt(fname_loss5, delimiter=" ", skiprows=1)

# ------------------------------------------------------- Plot 1 --------------------------------------------------------
plt.figure(1)
ax = plt.axes()
ax.plot(loss_1[:, 0], loss_1[:, 2], label="Default RAD, 100 Re-samples")
ax.plot(loss_2[:, 0], loss_2[:, 2], label="1k LBFG-S limit, 10 Re-Samples")
ax.plot(loss_3[:, 0], loss_3[:, 2], label="2k LBFG-S limit, 10 Re-Samples")


ax.set_yscale("log")
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_title("Comparing Test Loss across RAD cases")
ax.legend()

# ------------------------------------------------------- Plot 2 --------------------------------------------------------
plt.figure(2)
ax = plt.axes()
ax.plot(loss_1[:, 0], loss_1[:, 2], label="Default RAD, 100 Re-samples")
ax.plot(loss_2[:, 0], loss_2[:, 2], label="1k LBFG-S limit, 10 Re-Samples")
ax.plot(loss_3[:, 0], loss_3[:, 2], label="2k LBFG-S limit, 10 Re-Samples")

ax.set_yscale("log")
ax.set_xlim(0, 50000)
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_title("Comparing Test Loss across RAD cases")
ax.legend()

# ------------------------------------------------------- Plot 3 --------------------------------------------------------
# plt.figure(3)
# ax = plt.axes()
# ax.plot(loss_1[:, 0], loss_1[:, 2], label="")
# ax.plot(loss_2[:, 0], loss_2[:, 2], label="")

# ax.set_yscale("log")
# ax.set_xlabel("Steps")
# ax.set_ylabel("Loss")
# ax.set_title("")
# ax.legend()

plt.show()
