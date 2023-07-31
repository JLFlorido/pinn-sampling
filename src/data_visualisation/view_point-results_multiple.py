# Quick function to load and plot best solution from test.data files
# Can also use for any file with (x,t,z) format. Used to check residuals and gradients.
# Different filepaths for test and train because default code produces both and was unsure which to use.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Data For Plot 1 and 2 (The scatter graph)
fname1 = "results/plots/Gradients and Curvature/Points_res.txt" # Residuals
fname2 = "results/plots/Gradients and Curvature/Points_dudx.txt" # Jacobian, dudx
fname3 = "results/plots/Gradients and Curvature/Points_dudt.txt" # Jacobian, dudt
fname4 = "results/plots/Gradients and Curvature/Points_du_xx.txt" # Hessian, dxx
fname5 = "results/plots/Gradients and Curvature/Points_du_tt.txt" # Hessian, dtt
fname6 = "results/plots/Gradients and Curvature/Points_du_xt.txt" # Hessian, dxt
fname7 = "results/plots/Gradients and Curvature/Points_du_tx.txt" # Hessian, dtx

case1 = np.loadtxt(fname1, delimiter=" ", skiprows=1)
case2 = np.loadtxt(fname2, delimiter=" ", skiprows=1)
case3 = np.loadtxt(fname3, delimiter=" ", skiprows=1)
case4 = np.loadtxt(fname4, delimiter=" ", skiprows=1)
case5 = np.loadtxt(fname5, delimiter=" ", skiprows=1)
case6 = np.loadtxt(fname6, delimiter=" ", skiprows=1)
case7 = np.loadtxt(fname7, delimiter=" ", skiprows=1)

# ------------------------------------------------------- Plot 1 --------------------------------------------------------
#                                                   Distribution of points
# plt.figure(1)
# ax = plt.axes()
# ax.scatter(
#     case1[:, 1],
#     case1[:, 0],
#     marker=".",
# )
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_ylim([-1, 1])
# ax.set_xlim([0, 1])


# ------------------------------------------------------- Plot 2 --------------------------------------------------------
#                         Plot 1 but with 3 subplots to compare 3 different states.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
# Plot the first subplot
ax1.scatter(case1[:, 1], case1[:, 0], marker=".")
ax1.set_xlabel("$t$")
ax1.set_ylabel("$x$")
ax1.set_title("From Residuals")
ax1.set_ylim([-1, 1])
ax1.set_xlim([0, 1])
# # Plot the second subplot
ax2.scatter(case2[:, 1], case2[:, 0], marker=".")
ax2.set_xlabel("$t$")
ax2.set_ylabel("$x$")
ax2.set_title("$du/dx$")
ax2.set_ylim([-1, 1])
ax2.set_xlim([0, 1])
# Plot the third subplot
ax3.scatter(case3[:, 1], case3[:, 0], marker=".")
ax3.set_xlabel("$t$")
ax3.set_ylabel("$x$")
ax3.set_title("$du/dt$")
ax3.set_ylim([-1, 1])
ax3.set_xlim([0, 1])
# Hessian plots
fig, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2, figsize=(9, 8))
# Plot
ax4.scatter(case4[:, 1], case4[:, 0], marker=".")
ax4.set_xlabel("$t$")
ax4.set_ylabel("$x$")
ax4.set_title("$d^2u/dx^2$")
ax4.set_ylim([-1, 1])
ax4.set_xlim([0, 1])
# Plot
ax5.scatter(case5[:, 1], case5[:, 0], marker=".")
ax5.set_xlabel("$t$")
ax5.set_ylabel("$x$")
ax5.set_title("$d^2u/dt^2$")
ax5.set_ylim([-1, 1])
ax5.set_xlim([0, 1])
# Plot
ax6.scatter(case6[:, 1], case6[:, 0], marker=".")
ax6.set_xlabel("$t$")
ax6.set_ylabel("$x$")
ax6.set_title("$d^2u/dxdt$")
ax6.set_ylim([-1, 1])
ax6.set_xlim([0, 1])
# Plot
ax7.scatter(case7[:, 1], case7[:, 0], marker=".")
ax7.set_xlabel("$t$")
ax7.set_ylabel("$x$")
ax7.set_title("$d^2u/dtdx$")
ax7.set_ylim([-1, 1])
ax7.set_xlim([0, 1])

# ------------------------------------------------------- Plot 3 --------------------------------------------------------
#                                               Coloured 3D Scatter graph
# plt.figure(0)
# ax = plt.axes(projection=Axes3D.name)
# ax.scatter3D(
#     case3[:, 0],
#     case3[:, 1],
#     case3[:, 2],
#     c=case3[:, 2],
#     marker=".",
#     s=20,
#     cmap="coolwarm",
# )
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_title("$PDE Residual$")

# plt.figure(1)
# ax = plt.axes(projection=Axes3D.name)
# ax.scatter3D(
#     case1[:, 0],
#     case1[:, 1],
#     case1[:, 2],
#     c=case1[:, 2],
#     marker=".",
#     s=20,
#     cmap="coolwarm",
# )
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_title("$du/dx$")

# plt.figure(2)
# ax = plt.axes(projection=Axes3D.name)
# ax.scatter3D(
#     case2[:, 0],
#     case2[:, 1],
#     case2[:, 2],
#     c=case2[:, 2],
#     marker=".",
#     s=20,
#     cmap="coolwarm",
# )
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_title("$du/dt$")

# plt.figure(3)
# ax = plt.axes(projection=Axes3D.name)
# ax.scatter3D(
#     case4[:, 0],
#     case4[:, 1],
#     case4[:, 2],
#     c=case4[:, 2],
#     marker=".",
#     s=20,
#     cmap="coolwarm",
# )
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_title("Hessian $dxx$")

# plt.figure(4)
# ax = plt.axes(projection=Axes3D.name)
# ax.scatter3D(
#     case5[:, 0],
#     case5[:, 1],
#     case5[:, 2],
#     c=case5[:, 2],
#     marker=".",
#     s=20,
#     cmap="coolwarm",
# )
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_title("Hessian $dtt$")

# plt.figure(5)
# ax = plt.axes(projection=Axes3D.name)
# ax.scatter3D(
#     case6[:, 0],
#     case6[:, 1],
#     case6[:, 2],
#     c=case6[:, 2],
#     marker=".",
#     s=20,
#     cmap="coolwarm",
# )
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_title("Hessian $dxdt$")

# plt.figure(6)
# ax = plt.axes(projection=Axes3D.name)
# ax.scatter3D(
#     case7[:, 0],
#     case7[:, 1],
#     case7[:, 2],
#     c=case7[:, 2],
#     marker=".",
#     s=20,
#     cmap="coolwarm",
# )
# ax.set_xlabel("$t$")
# ax.set_ylabel("$x$")
# ax.set_title("Hessian $dtdx$")

plt.show()
