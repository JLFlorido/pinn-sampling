# Quick function to load and plot best solution from test.data files
# Can also use for any file with (x,t,z) format. Used to check residuals and gradients.
# Different filepaths for test and train because default code produces both and was unsure which to use.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Import Res Data
fname0 = "results/raw/loose_end_1/res_result_0"
fname1 = "results/raw/loose_end_1/res_result_1"
fname2 = "results/raw/loose_end_1/res_result_2"
fname3 = "results/raw/loose_end_1/res_result_3"
fname4 = "results/raw/loose_end_1/res_result_4"
fname5 = "results/raw/loose_end_1/res_result_5"
fname6 = "results/raw/loose_end_1/res_result_6"
fname7 = "results/raw/loose_end_1/res_result_7"
fname8 = "results/raw/loose_end_1/res_result_8"
fname9 = "results/raw/loose_end_1/res_result_9"

case0 = np.loadtxt(fname0, delimiter=" ", skiprows=1)
case1 = np.loadtxt(fname1, delimiter=" ", skiprows=1)
case2 = np.loadtxt(fname2, delimiter=" ", skiprows=1)
case3 = np.loadtxt(fname3, delimiter=" ", skiprows=1)
case4 = np.loadtxt(fname4, delimiter=" ", skiprows=1)
case5 = np.loadtxt(fname5, delimiter=" ", skiprows=1)
case6 = np.loadtxt(fname6, delimiter=" ", skiprows=1)
case7 = np.loadtxt(fname7, delimiter=" ", skiprows=1)
case8 = np.loadtxt(fname8, delimiter=" ", skiprows=1)
case9 = np.loadtxt(fname9, delimiter=" ", skiprows=1)

# Import u data
u_fname0 = "results/raw/loose_end_1/u_result_0"
u_fname1 = "results/raw/loose_end_1/u_result_1"
u_fname2 = "results/raw/loose_end_1/u_result_2"
u_fname3 = "results/raw/loose_end_1/u_result_3"
u_fname4 = "results/raw/loose_end_1/u_result_4"
u_fname5 = "results/raw/loose_end_1/u_result_5"
u_fname6 = "results/raw/loose_end_1/u_result_6"
u_fname7 = "results/raw/loose_end_1/u_result_7"
u_fname8 = "results/raw/loose_end_1/u_result_8"
u_fname9 = "results/raw/loose_end_1/u_result_9"

u_case0 = np.loadtxt(fname0, delimiter=" ", skiprows=1)
u_case1 = np.loadtxt(fname1, delimiter=" ", skiprows=1)
u_case2 = np.loadtxt(fname2, delimiter=" ", skiprows=1)
u_case3 = np.loadtxt(fname3, delimiter=" ", skiprows=1)
u_case4 = np.loadtxt(fname4, delimiter=" ", skiprows=1)
u_case5 = np.loadtxt(fname5, delimiter=" ", skiprows=1)
u_case6 = np.loadtxt(fname6, delimiter=" ", skiprows=1)
u_case7 = np.loadtxt(fname7, delimiter=" ", skiprows=1)
u_case8 = np.loadtxt(fname8, delimiter=" ", skiprows=1)
u_case9 = np.loadtxt(fname9, delimiter=" ", skiprows=1)

# Import Ground Truth data
data = np.load("src/burgers/Burgers.npz")
t, x, exact = data["t"], data["x"], data["usol"].T
xx, tt = np.meshgrid(x, t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
y = exact.flatten()[:, None]
# ------------------------------------------------------- Math --------------------------------------------------------
#                                                 Taking 10-run average

u_average = np.average((u_case0,u_case1,u_case2,u_case3,u_case4,u_case5,u_case6,u_case7,u_case8,u_case9), axis=0)

res_average = np.average((case0,case1,case2,case3,case4,case5,case6,case7,case8,case9), axis=0)
y1 = y[:25599].flatten()
y2 = y[1:25600].flatten()
u_diff1 = np.abs(u_average-y1)
u_diff2 = np.abs(u_average-y2)
u_error0 = np.abs(u_case0-y1)
u_error1 = np.abs(u_case1-y1)
u_error2 = np.abs(u_case2-y1)
u_error3 = np.abs(u_case3-y1)
u_error4 = np.abs(u_case4-y1)
# ------------------------------------------------------- Plotting --------------------------------------------------------
#                                                  Distribution of points
# The goal is to compare the loss at every point with the error at every point.
# Create the figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

scatter1 = ax1.scatter(X[:25599,1], X[:25599,0], c=res_average, cmap='coolwarm',vmax=1)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Residual')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title('Residual, average of 10 cases')

scatter2 = ax2.scatter(X[:25599,1], X[:25599,0], c=u_diff1, cmap='coolwarm',vmax=1)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Error')
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title('Error, average of 10 runs')
plt.tight_layout()

# Figure 2
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

scatter3 = ax3.scatter(X[:25599, 1], X[:25599, 0], c=res_average, cmap='coolwarm',vmax=4)
cbar3 = plt.colorbar(scatter3, ax=ax3)
cbar3.set_label('Residual case 0')
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_title('Residual, average of 10 runs')

scatter4 = ax4.scatter(X[:25599, 1], X[:25599, 0], c=u_diff1, cmap='coolwarm',vmax=4)
cbar4 = plt.colorbar(scatter4, ax=ax4)
cbar4.set_label('Error')
ax4.set_xlabel('t')
ax4.set_ylabel('x')
ax4.set_title('Error, average of 10 runs')
plt.tight_layout()

# Figure 3
fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))

scatter5 = ax5.scatter(X[:25599, 1], X[:25599, 0], c=res_average, cmap='coolwarm')
cbar5 = plt.colorbar(scatter5, ax=ax5)
cbar5.set_label('Residual case 0')
ax5.set_xlabel('t')
ax5.set_ylabel('x')
ax5.set_title('Residual, average of 10 runs')

scatter6 = ax6.scatter(X[:25599, 1], X[:25599, 0], c=u_diff1, cmap='coolwarm')
cbar6 = plt.colorbar(scatter6, ax=ax6)
cbar6.set_label('Error')
ax6.set_xlabel('t')
ax6.set_ylabel('x')
ax6.set_title('Error, average of 10 runs')
plt.tight_layout()

print(max(u_error0))
plt.show()