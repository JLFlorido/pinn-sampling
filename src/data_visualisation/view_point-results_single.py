# Quick function to load and plot best solution from test.data files
# Can also use for any file with (x,t,z) format. Used to check residuals and gradients.
# Different filepaths for test and train because default code produces both and was unsure which to use.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Data For Plot 1 and 2 (The scatter graph)
# fname = "results/raw/sol-sampling-test/Points_dudx.txt" # Jacobian, dudx
fname = "results/plots/Gradients and Curvature/Points_dudt.txt" # Jacobian, dudt
case = np.loadtxt(fname, delimiter=" ", skiprows=1)

# ------------------------------------------------------- Plot 1 --------------------------------------------------------
#                                                   Distribution of points
plt.figure(1)
ax = plt.axes()
ax.scatter(
    case[:, 1],
    case[:, 0],
    marker=".",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_ylim([-1, 1])
ax.set_xlim([0, 1])

# ------------------------------------------------------- Plot 2 --------------------------------------------------------
#                                               Coloured 3D Scatter graph
# plt.figure(2)
# ax = plt.axes(projection=Axes3D.name)
# ax.scatter3D(
#     case[:, 0],
#     case[:, 1],
#     case[:, 2],
#     c=case[:, 2],
#     marker=".",
#     s=20,
#     cmap="coolwarm",
# )
# ax.set_xlabel("$x$")
# ax.set_ylabel("$t$")
# ax.set_title("$U after resampling according to du-dx$")

plt.show()
