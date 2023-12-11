"""
slice_viewer.py This reads in the pred, the corresponding points, and the resulting point resample.
"""
# ---- imports ----
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

fname_0_points = "results/raw/be/slices/res_uxt_0_points.txt"
fname_1_points = "results/raw/be/slices/res_uxt_1_points.txt"
fname_49_points = "results/raw/be/slices/res_uxt_49_points.txt"
fname_99_points = "results/raw/be/slices/res_uxt_99_points.txt"

fname_0_pred = "results/raw/be/slices/res_uxt_0_prediction.txt"
fname_1_pred = "results/raw/be/slices/res_uxt_1_prediction.txt"
fname_49_pred = "results/raw/be/slices/res_uxt_49_prediction.txt"
fname_99_pred = "results/raw/be/slices/res_uxt_99_prediction.txt"


fname_0_output_points = "results/raw/be/slices/res_uxt_0_output_points.txt"
fname_1_output_points = "results/raw/be/slices/res_uxt_1_output_points.txt"
fname_49_output_points = "results/raw/be/slices/res_uxt_49_output_points.txt"
fname_99_output_points = "results/raw/be/slices/res_uxt_99_output_points.txt"

points_0=np.loadtxt(fname_0_points)
points_1=np.loadtxt(fname_1_points)
points_49=np.loadtxt(fname_49_points)
points_99=np.loadtxt(fname_99_points)

pred_0=np.loadtxt(fname_0_pred)
pred_1=np.loadtxt(fname_1_pred)
pred_49=np.loadtxt(fname_49_pred)
pred_99=np.loadtxt(fname_99_pred)

output_points_0=np.loadtxt(fname_0_output_points)
output_points_1=np.loadtxt(fname_1_output_points)
output_points_49=np.loadtxt(fname_49_output_points)
output_points_99=np.loadtxt(fname_99_output_points)

points = [points_0,points_1,points_49,points_99]
pred = [pred_0, pred_1, pred_49, pred_99]
out = [output_points_0,output_points_1,output_points_49,output_points_99]
title_numbers = [1, 2, 49, 99]
fig, axs = plt.subplots(4, 2, figsize=(16,12))

for i, ax in enumerate(axs[:,0]):
    divnorm = TwoSlopeNorm(vcenter=0)
    scatter = ax.scatter(points[i][:, 1], points[i][:, 0], c=pred[i], norm=divnorm, cmap='coolwarm', marker='o', s=2)
    ax.set_title(f'Slice from resample {title_numbers[i]}')
    fig.colorbar(scatter, ax=ax)

for i, ax in enumerate(axs[:, 1]):
    ax.scatter(out[i][:, 1], out[i][:, 0], marker='o', s=4, color='blue')
    ax.set_title(f'Slice from resample {title_numbers[i]}')

plt.tight_layout
plt.show()