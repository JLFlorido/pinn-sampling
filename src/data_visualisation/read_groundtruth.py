"""
read_groundtruth.py
Reads the groundtruth solution .npz file and prints the shape of it. Excerpt from default code.
This was a test to better understand the shape of the variable.
"""
from numpy import load
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load("src/burgers/Burgers.npz")
t, x, exact = data["t"], data["x"], data["usol"].T
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
y = exact.flatten()[:, None]
#print(X[:,0].shape) # This is -1 to 1, the spatial dimension (on y axis). 25600 long
#print(X[:,1].shape) # This is 0 to 1, the time dimension (on x axis). 25600 long
dt = (1/100)
dx = (2/256)

du_dt, du_dx = np.gradient(exact,dt,dx,edge_order=2)
d2u_dt2, d2u_dtdx = np.gradient(du_dt, dt, dx, edge_order=2)
d2u_dxdt, d2u_dx2 = np.gradient(du_dx,dt, dx, edge_order=2)

du_dt_flat = du_dt.flatten()[:,None]
du_dx_flat = du_dx.flatten()[:,None]
d2u_dt2_flat = d2u_dt2.flatten()[:,None]
d2u_dtdx_flat = d2u_dtdx.flatten()[:,None]
d2u_dxdt_flat = d2u_dxdt.flatten()[:,None]
d2u_dx2_flat = d2u_dx2.flatten()[:,None]
#print(du_dt_flat.shape) #25600 long
#print(du_dx_flat.shape) #25600 long

plt.figure(1)
ax = plt.axes(projection=Axes3D.name)
ax.scatter3D(
    X[:, 0],
    X[:, 1],
    du_dt_flat,
    c=du_dt_flat,
    s=20,
    cmap="coolwarm",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$du/dt$ from Ground Truth, using 2nd order accurate FD")

plt.figure(2)
ax = plt.axes(projection=Axes3D.name)
ax.scatter3D(
    X[:, 0],
    X[:, 1],
    du_dx_flat,
    c=du_dx_flat,
    s=20,
    cmap="coolwarm",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$du/dx$ from Ground Truth, using 2nd order accurate FD")

plt.figure(3)
ax = plt.axes(projection=Axes3D.name)
ax.scatter3D(
    X[:, 0],
    X[:, 1],
    d2u_dt2_flat,
    c=d2u_dt2_flat,
    s=20,
    cmap="coolwarm",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$d^2u/dt^2$ from Ground Truth, using 2nd order accurate FD")

plt.figure(4)
ax = plt.axes(projection=Axes3D.name)
ax.scatter3D(
    X[:, 0],
    X[:, 1],
    d2u_dx2_flat,
    c=d2u_dx2_flat,
    s=20,
    cmap="coolwarm",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$d^2u/dx^2$ from Ground Truth, using 2nd order accurate FD")

plt.figure(5)
ax = plt.axes(projection=Axes3D.name)
ax.scatter3D(
    X[:, 0],
    X[:, 1],
    d2u_dxdt_flat,
    c=d2u_dxdt_flat,
    s=20,
    cmap="coolwarm",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$d^2u/dxdt$ from Ground Truth, using 2nd order accurate FD")

plt.figure(6)
ax = plt.axes(projection=Axes3D.name)
ax.scatter3D(
    X[:, 0],
    X[:, 1],
    d2u_dtdx_flat,
    c=d2u_dtdx_flat,
    s=20,
    cmap="coolwarm",
    marker=".",
)
ax.set_ylabel("$t$")
ax.set_xlabel("$x$")
ax.set_xlim([-1, 1])
ax.set_ylim([0, 1])
ax.set_title("$d^2u/dtdtx$ from Ground Truth, using 2nd order accurate FD")
plt.show()