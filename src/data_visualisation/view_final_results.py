"""
view_final_results.py Objective is to be able to read in the additional info file for final points and import the final solution over X_Test and plot it. Should be simple aim is just to view it.
"""
import numpy as np
import matplotlib.pyplot as plt
IC="5"
# Define the file path
# file_path = "results/raw/additional_info/allencahn_uxt_D3_Hammersley_k0.5c1.0_N2000_L100_finalypred.dat" #  allen cahn
file_path = f"results/raw/additional_info/ic/burgers_IC{IC}_pdedxt_D3_Hammersley_k0.5c1.0_N2000_L100_finalypred.dat" # pdedxt
# file_path = f"results/raw/additional_info/ic/burgers_IC{IC}_uxt_D3_Hammersley_k0.5c1.0_N2000_L100_finalypred.dat" # uxt
# file_path = f"results/raw/additional_info/ic/burgers_IC{IC}_residual_D3_Random_k1.0c1.0_N2000_L100_finalypred.dat" # wu

# Load data skipping the first row
data = np.loadtxt(file_path, skiprows=1)

# Extract columns
y = data[:, 0]
x = data[:, 1]
u = data[:, 2]

unique_x = np.unique(x)
unique_y = np.unique(y)

# Create vectors (a) and (b)
a = unique_x
b = unique_y
a, b = np.meshgrid(a,b)
ab_matrix = np.zeros_like(a)
print(ab_matrix.shape)
# Populate (a,b) matrix with corresponding u values
for i in range(len(x)):
    idx_a = np.where(unique_x == x[i])[0][0]
    idx_b = np.where(unique_y == y[i])[0][0]
    ab_matrix[idx_b, idx_a] = u[i]
plt.pcolormesh(a, b, ab_matrix, cmap='rainbow')
plt.xlabel('Time (t)')
plt.ylabel('Space (x)')
plt.title(f'Pinn Solution for IC_{IC}, Using PDE Curvature')
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label("u(t,x)")
cbar.mappable.set_clim(-1, 1)
plt.show()

# Plot scatter plot
# plt.scatter(x, y, c=u, cmap='rainbow', marker='o', alpha=0.7)
# plt.colorbar(label='u')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'U pred of IC_{IC}')
# plt.xlim(0,1)
# plt.ylim(-1,1)
# plt.show()

