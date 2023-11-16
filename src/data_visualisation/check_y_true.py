"""
check_y_true.py This reads in error and point information and shows how what info was used to create what resample.
"""
# ---- imports ----
import numpy as np
import matplotlib.pyplot as plt
# ---- ---- -----
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
indices = np.random.choice(len(X_test), sample_size, replace=False)
X_test = X_test[indices]
ypred_on_xtest_0=ypred_on_xtest_0[indices]
ypred_on_xtest_1=ypred_on_xtest_1[indices]
ypred_on_xtest_2=ypred_on_xtest_2[indices]
ypred_on_xtest_19=ypred_on_xtest_19[indices]
ypred_on_xtest_39=ypred_on_xtest_39[indices]
ypred_on_xtest_59=ypred_on_xtest_59[indices]
error_xtest_0 = error_xtest_0[indices]
error_xtest_1 = error_xtest_1[indices]
error_xtest_2 = error_xtest_2[indices]
error_xtest_19 = error_xtest_19[indices]
error_xtest_39 = error_xtest_39[indices]
error_xtest_59 = error_xtest_59[indices]
y_true = y_true[indices]

# Create scatter plots for each variable
scatter_variables = [ypred_on_xtest_0, ypred_on_xtest_1, ypred_on_xtest_2, ypred_on_xtest_19, ypred_on_xtest_39, ypred_on_xtest_59]
scatter_variables2 = [error_xtest_0, error_xtest_1, error_xtest_2, error_xtest_19, error_xtest_39, error_xtest_59]

# Figure 0
fig0, ax0 = plt.subplots(figsize=(8, 6))
scatter0 = ax0.scatter(X_test[:, 1], X_test[:, 0], c=y_true, cmap='viridis', marker='o', s=4)
plt.colorbar(scatter0, ax=ax0, label='Predicted Values')
ax0.set_xlabel('X-axis label')
ax0.set_ylabel('Y-axis label')
ax0.set_title('Figure 0: Scatter Plot with y_true')

# Figure 1
fig1, axs1 = plt.subplots(2, 3, figsize=(15, 8))  # Create a 2x3 grid of subplots for Figure 1

for i in range(2):
    for j in range(3):
        index = i * 3 + j
        scatter1 = axs1[i, j].scatter(X_test[:, 1], X_test[:, 0], c=scatter_variables[index], cmap='viridis', marker='o', s=2)
        fig1.colorbar(scatter1, ax=axs1[i, j], label='Error Values')

plt.tight_layout()  # Ensures that subplots don't overlap

# Figure 2
fig2, axs2 = plt.subplots(2, 3, figsize=(15, 8))  # Create a 2x3 grid of subplots for Figure 2

for i in range(2):
    for j in range(3):
        index = i * 3 + j
        scatter2 = axs2[i, j].scatter(X_test[:, 1], X_test[:, 0], c=scatter_variables2[index], cmap='viridis', marker='o', s=2)
        fig2.colorbar(scatter2, ax=axs2[i, j], label='Error Values')

plt.tight_layout()  # Ensures that subplots don't overlap
plt.show()