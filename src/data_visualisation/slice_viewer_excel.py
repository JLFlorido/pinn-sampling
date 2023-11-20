"""
slice_viewer_excel.py This reads in error and point information and shows how what info was used to create what resample. Plots two figures with 6 subfigures each for each loop.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Load the Excel file
excel_file = 'results/spreadsheets/Results - Point Distributions.xlsx'  # Replace with your file path
sheet_name = "PDE_Gradients_xt"
df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')

# Extract data
x = df.iloc[4:25603, 4].values  # Assuming x is in the 5th column (index 4)
y = df.iloc[4:25603, 3].values  # Assuming y is in the 4th column (index 3)

# Point Resampling
x1 = df.iloc[4:2004, 6].values
t1 = df.iloc[4:2004, 7].values
x2 = df.iloc[4:2004, 9].values
t2 = df.iloc[4:2004, 10].values
x3 = df.iloc[4:2004, 12].values
t3 = df.iloc[4:2004, 13].values
x4 = df.iloc[4:2004, 15].values
t4 = df.iloc[4:2004, 16].values
x5 = df.iloc[4:2004, 18].values
t5 = df.iloc[4:2004, 19].values
x6 = df.iloc[4:2004, 21].values
t6 = df.iloc[4:2004, 22].values

# Extract u values
u_indices = [5, 8, 11, 14, 17, 20]  # Assuming u for values 19, 39, 59, 79, and 99 are in columns 9, 12, 15, 18, and 21
u_indices2 = ['u_5','u_8','u_11','u_14','u_17','u_20']
u_values = {f'u_{i}': df.iloc[4:25603, i].values for i in u_indices}
u_titles = [0, 19, 39, 59, 79, 99]

# Create a 3x2 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Plot scatter plots for each u value in subplots
for ax, (u_index, u_data) in zip(axs.flatten(), u_values.items()):
    divnorm = TwoSlopeNorm(vcenter=0)
    scatter = ax.scatter(x, y, c=u_data, norm=divnorm, cmap='coolwarm', alpha=0.7)
    ax.set_xlabel('Time, t')
    ax.set_ylabel('Distance, x')
    ax.set_title(f'Loop {u_titles[u_indices2.index(u_index)]}')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    fig.colorbar(scatter, ax=ax)
    fig.suptitle('Prediction of dL/dxt over test points, c1k1', fontsize=16)
plt.tight_layout()

fig2, axs2 = plt.subplots(2, 3, figsize=(12, 8))
pairs_with_titles = [(x1, t1, 'Loop 0'), (x2, t2, 'Loop 19'), (x3, t3, 'Loop 39'), (x4, t4, 'Loop 59'), (x5, t5, 'Loop 79'), (x6, t6, 'Loop 99')]

# Plot scatter plots for each x, t pair in subplots with swapped axes
for ax, (x_data, t_data, title) in zip(axs2.flatten(), pairs_with_titles):
    scatter = ax.scatter(t_data, x_data, alpha=1, s=4)  # Swap x and t here
    ax.set_xlabel('Time, t')
    ax.set_ylabel('Distance, x')  # Swap x and t here
    ax.set_title(f'Corresponding Resample - {title}')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)


fig2.suptitle('Point resamples according to dL/dxt, c1k1', fontsize=16)
plt.tight_layout()

plt.show()