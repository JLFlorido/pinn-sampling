# Quick function to load and plot best solution from test.data files
# Can also use for any file with (x,t,z) format. Used to check residuals and gradients.
# Different filepaths for test and train because default code produces both and was unsure which to use.
import numpy as np
import matplotlib.pyplot as plt

# Import Data For Plot 1 and 2 (The scatter graph)
# fname = "results/raw/sol-sampling-test/Points_dudx.txt" # Jacobian, dudx
fname = "results/plots/1_geometry_based_resampling/txt files/Points_res.txt" # Jacobian, dudt
case = np.loadtxt(fname, delimiter=" ", skiprows=1)

fname2 = "results/plots/1_geometry_based_resampling/txt files/Points_du_xt.txt" # Jacobian, dudt
case2 = np.loadtxt(fname2, delimiter=" ", skiprows=1)

# ------------------------------------------------------- Plot 1 --------------------------------------------------------
#                                                   Distribution of points

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
font_size = 18

# Plot data on the first subplot
ax1.scatter(
    case[:, 1],
    case[:, 0],
    marker=".",
)
ax1.set_xlabel("$t$", fontsize=font_size)
ax1.set_ylabel("$x$", fontsize=font_size)
ax1.set_ylim([-1, 1])
ax1.set_xlim([0, 1])
ax1.set_title("Example point resampling\n using PDE residual", fontsize=font_size)

# Plot data on the second subplot
ax2.scatter(
    case2[:, 1],
    case2[:, 0],
    marker=".",
)
ax2.set_xlabel("$t$", fontsize=font_size)
ax2.set_ylabel("$x$", fontsize=font_size)
ax2.set_ylim([-1, 1])
ax2.set_xlim([0, 1])
ax2.set_title("Example point resampling\n using curvature estimate", fontsize=font_size)

# Adjust the spacing between subplots
plt.tight_layout()

# Save the figure with a DPI of 300
plt.savefig("point_resampling.png", dpi=300)

# Show the plot (optional)
plt.show()
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
