"""
slice_viewer_2.py This reads and displays points before and after the resample, as well as the 
prediction step that is used to resample the points. You have to specify which method to read in by manually changing the code.

"""
# ---- imports ----
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker
# Insert method here: pde, pdext, uxt.
method = "pde"

# Import filenames
fname_0_points = f"results/raw/slices/points_before_{method}_0.txt"
fname_49_points = f"results/raw/slices/points_before_{method}_49.txt"

fname_0_pred = f"results/raw/slices/pred_sol_{method}_0.txt"
fname_49_pred = f"results/raw/slices/pred_sol_{method}_49.txt"
fname_0_pred_x = f"results/raw/slices/pred_x_{method}_0.txt"
fname_49_pred_x = f"results/raw/slices/pred_x_{method}_49.txt"

fname_0_output_points = f"results/raw/slices/points_after_{method}_0.txt"
fname_49_output_points = f"results/raw/slices/points_after_{method}_49.txt"

points_0=np.loadtxt(fname_0_points, unpack=True)
points_49=np.loadtxt(fname_49_points, unpack=True)

pred_0=np.loadtxt(fname_0_pred)
pred_49=np.loadtxt(fname_49_pred)
pred_x_0=np.loadtxt(fname_0_pred_x, unpack=True)
pred_x_49=np.loadtxt(fname_49_pred_x, unpack=True)

output_points_0=np.loadtxt(fname_0_output_points, unpack=True)
output_points_49=np.loadtxt(fname_49_output_points, unpack=True)

points = [points_0,points_49]
pred = [pred_0, pred_49]
pred_x = [pred_x_0, pred_x_49]
out = [output_points_0,output_points_49]
title_numbers = [1, 49]

def fmt(x, pos):
    return f"{x:.0e}".replace('e', 'E').replace('+', '')

# for i in range(2):
#     plt.figure(figsize=(10, 3))
#     # plt.title(f"Method: {method}")
#     plt.title("Method: PDE, H")
#     plt.subplots_adjust(left=0.05, bottom=0.08, right=0.99, top=0.93, wspace=0.28, hspace=None)
#     ax1 = plt.subplot(131)
#     ax1.scatter(points[i][1, :], points[i][0, :], marker='o', s=4, color='blue')
#     ax1.set_title(f'Points before resample #{title_numbers[i]}', fontsize=9)
#     ax1.tick_params(axis='both', which='both', labelsize=9)
#     ax1.set_xlim(0, 1)
#     ax1.set_ylim(-1, 1)

#     ax2 = plt.subplot(132)
#     divnorm = TwoSlopeNorm(vcenter=0)
#     scatter = ax2.scatter(pred_x[i][1, :], pred_x[i][0, :], c=pred[i], norm=divnorm, cmap='coolwarm', marker='x', s=1)
#     ax2.set_title(f'Prediction {title_numbers[i]}', fontsize=9)
#     ax2.tick_params(axis='both', which='both', labelsize=9)
#     cbar = plt.colorbar(scatter, ax=ax2, format=ticker.FuncFormatter(fmt))
#     cbar.ax.tick_params(labelsize=9)
#     ax2.set_xlim(0, 1)
#     ax2.set_ylim(-1, 1)

#     ax3 = plt.subplot(133)
#     ax3.scatter(out[i][1, :], out[i][0, :], marker='o', s=4, color='blue')
#     ax3.set_title(f'Points after resample #{title_numbers[i]}', fontsize=9)
#     ax3.tick_params(axis='both', which='both', labelsize=9)
#     ax3.set_xlim(0, 1)
#     ax3.set_ylim(-1, 1)


#     plt.tight_layout
#     plt.savefig(f'results/plots/Visualising Resamples/{method}_L{title_numbers[i]}.eps', format='eps')
#     plt.show()

for i in [1]:
    plt.figure(figsize=(8, 3))
    # plt.title(f"Method: {method}")
    plt.title("Method: PDE, H")
    plt.subplots_adjust(left=0.09, bottom=0.155, right=0.99, top=0.93, wspace=0.32, hspace=None)

    ax2 = plt.subplot(121)
    divnorm = TwoSlopeNorm(vcenter=0)
    scatter = ax2.scatter(pred_x[i][1, :], pred_x[i][0, :], c=pred[i], norm=divnorm, cmap='coolwarm', marker='x', s=1)
    ax2.set_title(r'Y($\mathcal{X}$) - PDE Residual', fontsize=9)
    ax2.tick_params(axis='both', which='both', labelsize=9)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    cbar = plt.colorbar(scatter, ax=ax2, format=ticker.FuncFormatter(fmt))
    cbar.ax.tick_params(labelsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1, 1)

    ax3 = plt.subplot(122)
    ax3.scatter(out[i][1, :], out[i][0, :], marker='o', s=4, color='blue')
    ax3.set_title(f'Resulting Point Distribution', fontsize=9)
    ax3.tick_params(axis='both', which='both', labelsize=9)
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-1, 1)


    plt.tight_layout
    plt.savefig(f'results/plots/Visualising Resamples/{method}_L{title_numbers[i]}.eps', format='eps')
    plt.show()