"""
read_groundtruth.py
Reads the groundtruth solution .npz file and prints the shape of it. Excerpt from default code.
This was a test to better understand the shape of the variable.
"""
from numpy import load
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import deepxde as dde

# ----------------------------------------------- Load Ground Truth solution from .npz file -----------------------------------------------
data = np.load("src/burgers_solbased/Burgers.npz")
t, x, exact = data["t"], data["x"], data["usol"].T # t is (100,1). x is (256, 1). u is (100,256)
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
y = exact.flatten()[:, None]
# print(y)
# print("THIS IS Y ^^^")
# print(X.shape) # (25600,2) This is the first output of the gen_testdata() function
# print(y.shape) # (25600,1) This is the second output of the gen_testdata() function
# print(X[:,0].shape) # This is -1 to 1, the spatial dimension (on y axis). 25600 long
# print(X[:,1].shape) # This is 0 to 1, the time dimension (on x axis). 25600 long
# ----------------------------------------------- Calculating gradients and curvatures -----------------------------------------------
dt = (1/100)
dx = (2/256)

du_dt, du_dx = np.gradient(exact,dt,dx,edge_order=2)
d2u_dt2, d2u_dtdx = np.gradient(du_dt, dt, dx, edge_order=2)
d2u_dxdt, d2u_dx2 = np.gradient(du_dx,dt, dx, edge_order=2)

du_dt_flat = np.abs(du_dt.flatten()[:,None])
du_dx_flat = np.abs(du_dx.flatten()[:,None])
d2u_dt2_flat = np.abs(d2u_dt2.flatten()[:,None])
d2u_dtdx_flat = np.abs(d2u_dtdx.flatten()[:,None])
d2u_dxdt_flat = np.abs(d2u_dxdt.flatten()[:,None])
d2u_dx2_flat = np.abs(d2u_dx2.flatten()[:,None])

#print(du_dt_flat.shape) #25600 long
#print(du_dx_flat.shape) #25600 long
# x=np.squeeze(x)
# t=np.squeeze(t)
# exact=np.squeeze(exact)
# print(x.shape)
# print(t.shape)
# print(exact.shape)
# x = np.array([-1, -0.5, 0, 0.5, 1])
# t = np.array([ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# exact = np.array([
#     [0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
#     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#     [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
#     [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
#     [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
# ])

# print(x.shape)
# print(t.shape)
# print(exact.shape)

#------------------------- testing interpolator
# itp = RegularGridInterpolator( (t, x), exact, method='nearest', bounds_error=False, fill_value=None)

# points = np.array([[-0.990, 0.9],
#                  [-0.001, 0.91112437],
#                  [0.5064721, 0.97652539],
#                  [-0.5064721, 0.97652539],
#                  [0.39936939, 0.5225982 ],
#                  [-0.39936939, 0.5225982]])
# points = points[:,[1, 0]]
# res = itp(points)
# print(res)

# y_pred_local =  np.array([[ 0.0064253], [ 0.34632166], [-0.37742235], [ 0.37742235], [-0.69042899], [ 0.69042899]]) 
# y_pred_local = [x[0] for x in y_pred_local]
# print(y_pred_local)
# y_true_local = res
# l2_error_local = dde.metrics.l2_relative_error(y_true_local, y_pred_local)
# print(l2_error_local)
# ----------------------------------------------- 3D Plots of Ground Truth gradients and curvatures.-----------------------------------------------

fig1 = plt.figure(1)
ax = plt.axes(projection=Axes3D.name)
p1 = ax.scatter3D(
    X[:, 0],
    X[:, 1],
    du_dt_flat,
    c=du_dt_flat,
    s=15,
    cmap="viridis",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$du/dt$ from Ground Truth, using 2nd order accurate FD")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig1.colorbar(p1)

fig2 = plt.figure(2)
ax = plt.axes(projection=Axes3D.name)
p2 = ax.scatter3D(
    X[:, 0],
    X[:, 1],
    du_dx_flat,
    c=du_dx_flat,
    s=15,
    cmap="viridis",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$du/dx$ from Ground Truth, using 2nd order accurate FD")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig2.colorbar(p2)

fig3 = plt.figure(3)
ax = plt.axes(projection=Axes3D.name)
p3 = ax.scatter3D(
    X[:, 0],
    X[:, 1],
    d2u_dt2_flat,
    c=d2u_dt2_flat,
    s=15,
    cmap="viridis",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$d^2u/dt^2$ from Ground Truth, using 2nd order accurate FD")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig3.colorbar(p3)

fig4 = plt.figure(4)
ax = plt.axes(projection=Axes3D.name)
p4 = ax.scatter3D(
    X[:, 0],
    X[:, 1],
    d2u_dx2_flat,
    c=d2u_dx2_flat,
    s=15,
    cmap="viridis",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$d^2u/dx^2$ from Ground Truth, using 2nd order accurate FD")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig4.colorbar(p4)

fig5 = plt.figure(5)
ax = plt.axes(projection=Axes3D.name)
p5 = ax.scatter3D(
    X[:, 0],
    X[:, 1],
    d2u_dxdt_flat,
    c=d2u_dxdt_flat,
    s=15,
    cmap="viridis",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$d^2u/dxdt$ from Ground Truth, using 2nd order accurate FD")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig5.colorbar(p5)

fig6 = plt.figure(6)
ax = plt.axes(projection=Axes3D.name)
p6 = ax.scatter3D(
    X[:, 0],
    X[:, 1],
    d2u_dtdx_flat,
    c=d2u_dtdx_flat,
    s=15,
    cmap="viridis",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$d^2u/dtdtx$ from Ground Truth, using 2nd order accurate FD")
Axes3D.view_init(ax,elev=90,azim=0)
plt.tight_layout()
fig6.colorbar(p6)
plt.show()