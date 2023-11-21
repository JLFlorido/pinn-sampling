"""
view_final_results.py Objective is to be able to read in the additional info file for final points and import the final solution over X_Test and plot it. Should be simple aim is just to view it.
"""
import numpy as np
import matplotlib.pyplot as plt

# Define the file path
file_path = "results/raw/additional_info/allencahn_uxt_D3_Hammersley_k0.5c1.0_N2000_L100_finalypred.dat"

# Load data skipping the first row
data = np.loadtxt(file_path, skiprows=1)

# Extract columns
y = data[:, 0]
x = data[:, 1]
u = data[:, 2]

# Plot scatter plot
plt.scatter(x, y, c=u, cmap='viridis', marker='o', alpha=0.7)
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('y')
plt.title('U pred of A.C. equation\nusing curvature based resampling')
plt.xlim(0,1)
plt.ylim(-1,1)
plt.show()