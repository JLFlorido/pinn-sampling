"""
slice_viewer_2.py This reads and displays points before and after the resample, as well as the 
prediction step that is used to resample the points. You have to specify which method to read in by manually changing the code.

"""
# ---- imports ----
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
# Insert method here: pde, pdext, uxt.
method = "pdext"

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

for i in range(2):
    plt.figure(figsize=(15, 5))
    plt.title(f"Method: {method}")
    ax1 = plt.subplot(131)
    ax1.scatter(points[i][1, :], points[i][0, :], marker='o', s=4, color='blue')
    ax1.set_title(f'Points before resample #{title_numbers[i]}')

    ax2 = plt.subplot(132)
    divnorm = TwoSlopeNorm(vcenter=0)
    scatter = ax2.scatter(pred_x[i][1, :], pred_x[i][0, :], c=pred[i], norm=divnorm, cmap='coolwarm', marker='x', s=1)
    ax2.set_title(f'Prediction {title_numbers[i]}')
    cbar = plt.colorbar(scatter, ax=ax2)

    ax3 = plt.subplot(133)
    ax3.scatter(out[i][1, :], out[i][0, :], marker='o', s=4, color='blue')
    ax3.set_title(f'Points after resample #{title_numbers[i]}')


    plt.tight_layout
    plt.show()