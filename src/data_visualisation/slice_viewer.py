"""
slice_viewer.py This reads in error and point information and shows how what info was used to create what resample.
"""
# ---- imports ----
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
# ---- ---- -----
# ---- true ----
# def gen_testdata(): # This function opens the ground truth solution. Need to change path directory for running in ARC.
#     data = np.load("./Burgers2.npz")
#     t, x, exact = data["t"], data["x"], data["exact"].T
#     xx, tt = np.meshgrid(x, t)
#     X = np.vstack((np.ravel(xx), np.ravel(tt))).T
#     y = exact.flatten()[:, None]
#     return X, y
# X_test, y_true = gen_testdata()
# y_true=np.squeeze(y_true)
# # ---- ---- ----

# For error, with c1 k1
# Points
fname_0_dt = "results/raw/ac/dpde_dt_0.txt"
fname_1_dt = "results/raw/ac/dpde_dt_1.txt"
fname_2_dt = "results/raw/ac/dpde_dt_2.txt"
fname_19_dt = "results/raw/ac/dpde_dt_19.txt"
fname_39_dt = "results/raw/ac/dpde_dt_39.txt"
fname_59_dt = "results/raw/ac/dpde_dt_59.txt"
fname_79_dt = "results/raw/ac/dpde_dt_79.txt"
fname_97_dt = "results/raw/ac/dpde_dt_97.txt"
fname_98_dt = "results/raw/ac/dpde_dt_98.txt"
fname_99_dt = "results/raw/ac/dpde_dt_99.txt"

dt_pred_0 = np.loadtxt(fname_0_dt)
dt_pred_1 = np.loadtxt(fname_1_dt)
dt_pred_2 = np.loadtxt(fname_2_dt)
dt_pred_19 = np.loadtxt(fname_19_dt)
dt_pred_39 = np.loadtxt(fname_39_dt)
dt_pred_59 = np.loadtxt(fname_59_dt)
dt_pred_79 = np.loadtxt(fname_79_dt)
dt_pred_97 = np.loadtxt(fname_97_dt)
dt_pred_98 = np.loadtxt(fname_98_dt)
dt_pred_99 = np.loadtxt(fname_99_dt)

dt_predictions = [dt_pred_0, dt_pred_1, dt_pred_2, dt_pred_19, dt_pred_39, dt_pred_59, dt_pred_79, dt_pred_97, dt_pred_98, dt_pred_99]
# Error
fname_0_dx = "results/raw/ac/dpde_dt_0.txt"
fname_1_dx = "results/raw/ac/dpde_dt_1.txt"
fname_2_dx = "results/raw/ac/dpde_dt_2.txt"
fname_19_dx = "results/raw/ac/dpde_dt_19.txt"
fname_39_dx = "results/raw/ac/dpde_dt_39.txt"
fname_59_dx = "results/raw/ac/dpde_dt_59.txt"
fname_79_dx = "results/raw/ac/dpde_dt_79.txt"
fname_97_dx = "results/raw/ac/dpde_dt_97.txt"
fname_98_dx = "results/raw/ac/dpde_dt_98.txt"
fname_99_dx = "results/raw/ac/dpde_dt_99.txt"
dx_pred_0 = np.loadtxt(fname_0_dx)
dx_pred_1 = np.loadtxt(fname_1_dx)
dx_pred_2 = np.loadtxt(fname_2_dx)
dx_pred_19 = np.loadtxt(fname_19_dx)
dx_pred_39 = np.loadtxt(fname_39_dx)
dx_pred_59 = np.loadtxt(fname_59_dx)
dx_pred_79 = np.loadtxt(fname_79_dx)
dx_pred_97 = np.loadtxt(fname_97_dx)
dx_pred_98 = np.loadtxt(fname_98_dx)
dx_pred_99 = np.loadtxt(fname_99_dx)
dx_predictions = [dx_pred_0, dx_pred_1, dx_pred_2, dx_pred_19, dx_pred_39, dx_pred_59, dx_pred_79, dx_pred_97, dx_pred_98, dx_pred_99]
fname_xtest = "results/raw/ac/x_test.txt"
X_test = np.loadtxt(fname_xtest)

# error_xtest_0 = np.abs(ypred_on_xtest_0 - y_true)
# error_xtest_1 = np.abs(ypred_on_xtest_1 - y_true)
# error_xtest_2 = np.abs(ypred_on_xtest_2 - y_true)
# error_xtest_19 = np.abs(ypred_on_xtest_19 - y_true)
# error_xtest_39 = np.abs(ypred_on_xtest_39 - y_true)
# error_xtest_59 = np.abs(ypred_on_xtest_59 - y_true)

# sample_size = 100000

# # Randomly select a subset of points
# indices = np.random.choice(len(X_test), sample_size, replace=False)
# X_test = X_test[indices]
# error_xtest_0 = error_xtest_0[indices]
# error_xtest_1 = error_xtest_1[indices]
# error_xtest_2 = error_xtest_2[indices]
# error_xtest_19 = error_xtest_19[indices]
# error_xtest_39 = error_xtest_39[indices]
# error_xtest_59 = error_xtest_59[indices]

# fig, axs = plt.subplots(1, 2, figsize=(12, 4))
# scatter1 = axs[0].scatter(X_test[:, 1], X_test[:, 0], c=dt_pred_0, cmap='coolwarm', marker='o', s=4)
# fig.colorbar(scatter1, ax=axs[0])
# scatter2 = axs[1].scatter(X_test[:, 1], X_test[:, 0], c=dt_pred_1, cmap='coolwarm', marker='o', s=4)
# fig.colorbar(scatter1, ax=axs[1])
# plt.tight_layout()
fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns


title_numbers = [1, 2, 3, 19, 39, 59, 79, 97, 98, 99]
for i, ax in enumerate(axs.flatten()):
    divnorm = TwoSlopeNorm(vcenter=0)
    scatter = ax.scatter(X_test[:, 1], X_test[:, 0], c=dx_predictions[i], norm=divnorm, cmap='coolwarm', marker='o', s=4)
    fig.colorbar(scatter, ax=ax)
    ax.set_title(f'Iteration # {title_numbers[i]}')
plt.suptitle("Taking $d/dx$ of PDE residual at different resampling iterations")
plt.tight_layout()
# Show the figure
plt.show()