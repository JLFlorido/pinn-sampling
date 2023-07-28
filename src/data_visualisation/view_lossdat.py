# Quick function to load and plot loss vs iterations history from loss.dat files
# Intended for comparing loss vs iterations of various different runs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Loss Data
fname_loss1 = "results/raw/loss/loss_Ham2k.dat"
fname_loss2 = "results/raw/loss/loss_RAD_default.dat"
fname_loss3 = "results/raw/loss/loss_RAR-D_default.dat"
loss_1 = np.loadtxt(fname_loss1, delimiter=" ", skiprows=1)
loss_2 = np.loadtxt(fname_loss2, delimiter=" ", skiprows=1)
loss_3 = np.loadtxt(fname_loss3, delimiter=" ", skiprows=1)

# More imports if necessary
# fname_loss4 = "results/raw/loss/loss_RAR-D_2000_a2.dat"
# loss_4 = np.loadtxt(fname_loss4, delimiter=" ", skiprows=1)
# fname_loss5 = "results/raw/.dat"
# loss_5 = np.loadtxt(fname_loss5, delimiter=" ", skiprows=1)

# ------------------------------------------------------- Plot 1 --------------------------------------------------------
plt.figure(1)
ax = plt.axes()
ax.plot(loss_1[:, 0], loss_1[:, 2], label="Hammersley, 2k")
ax.plot(loss_2[:, 0], loss_2[:, 2], label="RAD, default settings")
ax.plot(loss_3[:, 0], loss_3[:, 2], label="RAR-D, default settings")
# ax.plot(loss_4[:, 0], loss_4[:, 2], label="RAR-D, 10 Re-samples, 15k Adam")


ax.set_yscale("log")
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_title("Comparing Test Loss of All cases")
ax.legend()

# ------------------------------------------------------- Plot 2 --------------------------------------------------------
plt.figure(2)
ax = plt.axes()
ax.plot(loss_1[:, 0], loss_1[:, 2], label="Hammersley, 2k")
ax.plot(loss_2[:, 0], loss_2[:, 2], label="RAD, default settings")
ax.plot(loss_3[:, 0], loss_3[:, 2], label="RAR-D, default settings")
# ax.plot(loss_4[:, 0], loss_4[:, 2], label="RAR-D, 10 Re-samples, 15k Adam")


ax.set_yscale("log")
ax.set_xlim(0, 50000)
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.set_title("Comparing Test Loss across uniform cases")
ax.legend()
# ------------------------------------------------------- Plot 1/2 Alternative subplot --------------------------------------------------------
# Create a figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(loss_1[:, 0], loss_1[:, 2], label="Hammersley, 2k")
ax1.plot(loss_2[:, 0], loss_2[:, 2], label="RAD, default settings")
ax1.plot(loss_3[:, 0], loss_3[:, 2], label="RAR-D, default settings")
# ax1.plot(loss_4[:, 0], loss_4[:, 2], label="RAR-D, 10 Re-samples, 15k Adam")

ax1.set_yscale("log")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(loss_1[:, 0], loss_1[:, 2], label="Hammersley, 2k")
ax2.plot(loss_2[:, 0], loss_2[:, 2], label="RAD, default settings")
ax2.plot(loss_3[:, 0], loss_3[:, 2], label="RAR-D, default settings")
# ax2.plot(loss_4[:, 0], loss_4[:, 2], label="RAR-D, 10 Re-samples, 15k Adam")

ax2.set_yscale("log")
ax2.set_xlim(0, 50000)
ax2.set_xlabel("Steps")
ax2.set_ylabel("Loss")
ax2.legend()
fig.suptitle("Comparing Test Loss")

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
