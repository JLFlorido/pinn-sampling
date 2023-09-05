# Quick function to load and plot best solution from test.data files
# Can also use for any file with (x,t,z) format. Used to check residuals and gradients.
# Different filepaths for test and train because default code produces both and was unsure which to use.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#                   2k
fname1 = "results/raw/grad_curvature_estimates/gc_coords.txt" # x,t coordinates
fname2 = "results/raw/grad_curvature_estimates/gc1_ux.txt" # Jacobian, dudx
fname3 = "results/raw/grad_curvature_estimates/gc1_ut.txt" # Jacobian, dudt
fname4 = "results/raw/grad_curvature_estimates/gc1_uxx.txt" # Hessian, dxx
fname5 = "results/raw/grad_curvature_estimates/gc1_utt.txt" # Hessian, dtt
fname6 = "results/raw/grad_curvature_estimates/gc1_uxt.txt" # Hessian, dxt

#                   15k
# fname1 = "results/raw/grad_curvature_estimates/gc_coords.txt" # x,t coordinates
# fname2 = "results/raw/grad_curvature_estimates/gc2_ux.txt" # Jacobian, dudx
# fname3 = "results/raw/grad_curvature_estimates/gc2_ut.txt" # Jacobian, dudt
# fname4 = "results/raw/grad_curvature_estimates/gc2_uxx.txt" # Hessian, dxx
# fname5 = "results/raw/grad_curvature_estimates/gc2_utt.txt" # Hessian, dtt
# fname6 = "results/raw/grad_curvature_estimates/gc2_uxt.txt" # Hessian, dxt

#                   End
# fname1 = "results/raw/grad_curvature_estimates/gc_coords.txt" # x,t coordinates
# fname2 = "results/raw/grad_curvature_estimates/gc3_ux.txt" # Jacobian, dudx
# fname3 = "results/raw/grad_curvature_estimates/gc3_ut.txt" # Jacobian, dudt
# fname4 = "results/raw/grad_curvature_estimates/gc3_uxx.txt" # Hessian, dxx
# fname5 = "results/raw/grad_curvature_estimates/gc3_utt.txt" # Hessian, dtt
# fname6 = "results/raw/grad_curvature_estimates/gc3_uxt.txt" # Hessian, dxt

case1 = np.loadtxt(fname1, delimiter=" ", skiprows=1)
case2 = np.loadtxt(fname2, delimiter=" ", skiprows=1)
case3 = np.loadtxt(fname3, delimiter=" ", skiprows=1)
case4 = np.loadtxt(fname4, delimiter=" ", skiprows=1)
case5 = np.loadtxt(fname5, delimiter=" ", skiprows=1)
case6 = np.loadtxt(fname6, delimiter=" ", skiprows=1)

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

# ------------------------------------------------------- Plot 3 --------------------------------------------------------
#                                               Coloured 3D Scatter graph
fig0 = plt.figure(0)

ax = plt.axes(projection=Axes3D.name)
p0 = ax.scatter3D(
    case1[:, 0],
    case1[:, 1],
    case2,
    c=case2,
    marker=".",
    s=15,
    cmap="viridis",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_title("$du/dx$, 2k Steps")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig0.colorbar(p0)

fig1 = plt.figure(1)
ax = plt.axes(projection=Axes3D.name)
p1 = ax.scatter3D(
    case1[:, 0],
    case1[:, 1],
    case3,
    c=case3,
    marker=".",
    s=15,
    cmap="viridis",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_title("$du/dt$, 2k Steps")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig1.colorbar(p1)

fig2 = plt.figure(2)
ax = plt.axes(projection=Axes3D.name)
p2 = ax.scatter3D(
    case1[:, 0],
    case1[:, 1],
    case4,
    c=case4,
    marker=".",
    s=15,
    cmap="viridis",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_title("$d^2u/dx^2$, 2k Steps")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig2.colorbar(p2)

fig3 = plt.figure(3)
ax = plt.axes(projection=Axes3D.name)
p3 = ax.scatter3D(
    case1[:, 0],
    case1[:, 1],
    case5,
    c=case5,
    marker=".",
    s=15,
    cmap="viridis",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_title("$d^2u/dt^2$, 2k Steps")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig3.colorbar(p3)

fig4 = plt.figure(4)
ax = plt.axes(projection=Axes3D.name)
p4 = ax.scatter3D(
    case1[:, 0],
    case1[:, 1],
    case6,
    c=case6,
    marker=".",
    s=15,
    cmap="viridis",
)
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_title("$d^2u/dxdt$, 2k Steps")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig4.colorbar(p4)

plt.show()
