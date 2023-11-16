"""
slice_viewer.py This reads in error and point information and shows how what info was used to create what resample.
"""
# ---- imports ----
import numpy as np
import matplotlib.pyplot as plt
# ---- ---- -----
# ---- true ----
def gen_testdata(): # This function opens the ground truth solution. Need to change path directory for running in ARC.
    data = np.load("./Burgers2.npz")
    t, x, exact = data["t"], data["x"], data["exact"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y
X_test, y_true = gen_testdata()
y_true=np.squeeze(y_true)
# ---- ---- ----

# For error, with c1 k1
# Points
fname_0_points = "results/raw/New folder1/p_points_error_0_k1.0_c0.0.txt"
fname_1_points = "results/raw/New folder1/p_points_error_1_k1.0_c0.0.txt"
fname_2_points = "results/raw/New folder1/p_points_error_2_k1.0_c0.0.txt"
fname_19_points = "results/raw/New folder1/p_points_error_19_k1.0_c0.0.txt"
fname_39_points = "results/raw/New folder1/p_points_error_39_k1.0_c0.0.txt"
fname_59_points = "results/raw/New folder1/p_points_error_59_k1.0_c0.0.txt"
points_0 = np.loadtxt(fname_0_points)
points_1 = np.loadtxt(fname_1_points)
points_2 = np.loadtxt(fname_2_points)
points_19 = np.loadtxt(fname_19_points)
points_39 = np.loadtxt(fname_39_points)
points_59 = np.loadtxt(fname_59_points)

# Error
fname_0_ypred = "results/raw/New folder1/p_ypred-on-xtest_error_0_k1.0_c0.0.txt"
fname_1_ypred = "results/raw/New folder1/p_ypred-on-xtest_error_1_k1.0_c0.0.txt"
fname_2_ypred = "results/raw/New folder1/p_ypred-on-xtest_error_2_k1.0_c0.0.txt"
fname_19_ypred = "results/raw/New folder1/p_ypred-on-xtest_error_19_k1.0_c0.0.txt"
fname_39_ypred = "results/raw/New folder1/p_ypred-on-xtest_error_39_k1.0_c0.0.txt"
fname_59_ypred = "results/raw/New folder1/p_ypred-on-xtest_error_59_k1.0_c0.0.txt"
ypred_on_xtest_0 = np.loadtxt(fname_0_ypred)
ypred_on_xtest_1 = np.loadtxt(fname_1_ypred)
ypred_on_xtest_2 = np.loadtxt(fname_2_ypred)
ypred_on_xtest_19 = np.loadtxt(fname_19_ypred)
ypred_on_xtest_39 = np.loadtxt(fname_39_ypred)
ypred_on_xtest_59 = np.loadtxt(fname_59_ypred)
 
error_xtest_0 = np.abs(ypred_on_xtest_0 - y_true)
error_xtest_1 = np.abs(ypred_on_xtest_1 - y_true)
error_xtest_2 = np.abs(ypred_on_xtest_2 - y_true)
error_xtest_19 = np.abs(ypred_on_xtest_19 - y_true)
error_xtest_39 = np.abs(ypred_on_xtest_39 - y_true)
error_xtest_59 = np.abs(ypred_on_xtest_59 - y_true)

sample_size = 100000

# Randomly select a subset of points
indices = np.random.choice(len(X_test), sample_size, replace=False)
X_test = X_test[indices]
error_xtest_0 = error_xtest_0[indices]
error_xtest_1 = error_xtest_1[indices]
error_xtest_2 = error_xtest_2[indices]
error_xtest_19 = error_xtest_19[indices]
error_xtest_39 = error_xtest_39[indices]
error_xtest_59 = error_xtest_59[indices]

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
scatter1 = axs[0].scatter(X_test[:, 1], X_test[:, 0], c=error_xtest_59, cmap='viridis', marker='o', s=4)
fig.colorbar(scatter1, ax=axs[0], label='Error Values')
scatter2 = axs[1].scatter(points_59[:, 1], points_59[:, 0], marker='o', color='b', s=4)
for ax in axs:
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
fig.suptitle('Resample 60', fontsize=16)
plt.tight_layout()

# Show the figure
plt.show()