# Quick function to load and plot best solution from test.data files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

fname_test = "results/raw/uniform/test.dat"
train_state = np.loadtxt(fname_test, delimiter=" ", skiprows=1)
# data = train_state[:, 2]
# x = train_state[:, 0]
# y = train_state[:, 0]

# X, Y = np.meshgrid(x, y)


# print(np.shape(X))


plt.figure()
ax = plt.axes(projection=Axes3D.name)
ax.scatter3D(
    train_state[:, 0],
    train_state[:, 1],
    train_state[:, 2],
    ".",
)
plt.show()
# Z = pd.pivot_table(train_state, index="x", columns="y", values="z").T.values
# X_unique = np.sort(train_state.x.unique())
# Y_unique = np.sort(train_state.y.unique())
# X, Y = np.meshgrid(X_unique, Y_unique)
# print(X_unique.shape)
# print(Y_unique.shape)

# df = pd.DataFrame(dict(x=train_state[:, 0], t=train_state[:, 1], u=train_state[:, 2]))
# xcol, tcol, ucol = "x", "t", "u"
# df = df.sort_values(by=[xcol, tcol])
# print(train_state.shape)
# xvals = df[xcol].unique()
# print(xvals.shape)
# print(train_state[:, 0].shape)
# tvals = df[tcol].unique()
# print(tvals.shape)
# print(train_state[:, 1].shape)
# uvals = df[ucol].values.reshape(len(xvals), len(tvals))

# plt.figure()
# ax = plt.axes()
# ax = plt.plot(train_state[:, 0], train_state[:, 1], ".")
# # ax.pcolormesh(
# #     tvals,
# #     xvals,
# #     uvals,
# #     cmap="rainbow",
# # # )
# # ax.set_xlabel("$t$")
# # ax.set_ylabel("$x$")
# # ax.set_title("$u(x,t)$")  # ("$y_{}$".format(i + 1))

# plt.show()
