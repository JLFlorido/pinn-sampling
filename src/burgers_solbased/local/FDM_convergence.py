"""
FDM_convergence.py Loads all .npz and calculates difference between them, plotting that plus the solutions of some cases.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interp2d
IC="2"


data_256 = np.load(f"src/burgers_solbased/Burgers_IC_{IC}_N256_nt101.npz")
t_256, x_256, u_256 = data_256["t"], data_256["x"], data_256["exact"].T

data_800 = np.load(f"src/burgers_solbased/Burgers_IC_{IC}_N800_nt201.npz")
t_800, x_800, u_800 = data_800["t"], data_800["x"], data_800["exact"].T

data_1600 = np.load(f"src/burgers_solbased/Burgers_IC_{IC}_N1600_nt401.npz")
t_1600, x_1600, u_1600 = data_1600["t"], data_1600["x"], data_1600["exact"].T
u_1600to800 = u_1600[::2,::2]

data_3200 = np.load(f"src/burgers_solbased/Burgers_IC_{IC}_N3200_nt801.npz")
_, _, u_3200 = data_3200["t"], data_3200["x"], data_3200["exact"].T
u_3200to1600 = u_3200[::2,::2]

data_6400 = np.load(f"src/burgers_solbased/Burgers_IC_{IC}_N6400_nt1601.npz")
_, _, u_6400 = data_6400["t"], data_6400["x"], data_6400["exact"].T
u_6400to3200 = u_6400[::2,::2]

data_12800 = np.load(f"src/burgers_solbased/Burgers_IC_{IC}_N12800_nt3201.npz")
t_12800, x_12800, u_12800 = data_12800["t"], data_12800["x"], data_12800["exact"].T
u_12800to6400 = u_12800[::2,::2]

# Interpolating to 256x100 to compare to it

interpolator_800 = interp2d(x_800, t_800, u_800, kind='linear')
interpolator_12800 = interp2d(x_12800, t_12800, u_12800, kind='linear')
new_x = np.linspace(-1, 1, 257)
new_t = np.linspace(0, 1, 101)
u_800to256 = interpolator_800(new_x,new_t)
# -- -- -- -- -- -- -- -- -- --
L2diff_256 = ((u_800to256 - u_256) ** 2).mean(axis=None)
print("The mean error in u for 256->800 was ", L2diff_256)
L2diff_800 = ((u_1600to800 - u_800) ** 2).mean(axis=None)
print("The mean error in u for 800->1600 was ", L2diff_800)
L2diff_1600 = ((u_3200to1600 - u_1600) ** 2).mean(axis=None)
print("The mean error in u for 1600->3200 was ", L2diff_1600)
L2diff_3200 = ((u_6400to3200 - u_3200) ** 2).mean(axis=None)
print("The mean error in u for 3200->6400 was ", L2diff_3200)
L2diff_6400 = ((u_12800to6400 - u_6400) ** 2).mean(axis=None)
print("The mean error in u for 6400->12800 was ", L2diff_6400)
# -- -- -- -- -- -- -- -- -- --
# Calculating error from 12800 for all
u_12800to3200 = u_12800[::4,::4]
u_12800to1600 = u_12800[::8,::8]
u_12800to800 = u_12800[::16,::16]
u_12800to256 = interpolator_12800(new_x,new_t)
L2err_256 = ((u_12800to256 - u_256) ** 2).mean(axis=None)
print("The mean error in u for 256->12800 was ", L2err_256)
L2err_800 = ((u_12800to800 - u_800) ** 2).mean(axis=None)
print("The mean error in u for 800->12800 was ", L2err_800)
L2err_1600 = ((u_12800to1600 - u_1600) ** 2).mean(axis=None)
print("The mean error in u for 1600->12800 was ", L2err_1600)
L2err_3200 = ((u_12800to3200 - u_3200) ** 2).mean(axis=None)
print("The mean error in u for 3200->12800 was ", L2err_3200)
L2err_6400 = ((u_12800to6400 - u_6400) ** 2).mean(axis=None)
print("The mean error in u for 6400->12800 was ", L2err_6400)
# -- -- -- -- -- -- -- -- -- --
# grid_x, grid_t = np.mgrid[-1:1:400j, 0:1:400j]
# u_400to800 = griddata(,u_400, (grid_x, grid_t),method='cubic')
L2diff_values = [L2diff_256,L2diff_800, L2diff_1600, L2diff_3200, L2diff_6400]
L2err_values = [L2err_256,L2err_800, L2err_1600, L2err_3200, L2err_6400]
numbers_after_underscore = [256, 800, 1600, 3200, 6400]

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(numbers_after_underscore, L2diff_values, color='blue')
plt.title(f'For IC {IC}: FDM Independence, difference between consecutive')
plt.xlabel('Grid size')
plt.ylabel('Mean Square Difference')
plt.yscale('log')  # Set y-axis to logarithmic scale

plt.figure(figsize=(10, 6))
plt.scatter(numbers_after_underscore, L2err_values, color='blue')
plt.title(f'For IC {IC}: FDM Independence, difference to finest')
plt.xlabel('Grid size')
plt.ylabel('Mean Square Difference')
plt.yscale('log')  # Set y-axis to logarithmic scale

divnorm = colors.TwoSlopeNorm(vcenter=0)
divnorm2 = colors.TwoSlopeNorm(vcenter=0)

plt.figure(figsize=(10, 6))
contour_plot = plt.contourf(t_256, x_256, u_256.T, levels=40, cmap='rainbow', norm=divnorm)  # Inverted the axes
plt.colorbar(contour_plot, label='u(x, t)')  # Add colorbar with label
plt.title('I.C. 2: u(x, t) for N=256, from FDM')
plt.xlabel('t')  # Updated xlabel
plt.ylabel('x')  # Updated ylabel

plt.figure(figsize=(10, 6))
contour_plot = plt.contourf(t_12800, x_12800, u_12800.T, levels=40, cmap='rainbow', norm=divnorm2)  # Inverted the axes
plt.colorbar(contour_plot, label='u(x, t)')  # Add colorbar with label
plt.title('I.C. 2: u(x, t) for N=12800, from FDM')
plt.xlabel('t')  # Updated xlabel
plt.ylabel('x')  # Updated ylabel

plt.show()