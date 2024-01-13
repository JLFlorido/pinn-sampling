import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
IC="2"

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

# -- -- -- -- -- -- -- -- -- --
L2err_1600 = ((u_3200to1600 - u_1600) ** 2).mean(axis=None)
print("The mean error in u for 1600->3200 was ", L2err_1600)
L2err_3200 = ((u_6400to3200 - u_3200) ** 2).mean(axis=None)
print("The mean error in u for 3200->6400 was ", L2err_3200)
L2err_6400 = ((u_12800to6400 - u_6400) ** 2).mean(axis=None)
print("The mean error in u for 6400->12800 was ", L2err_6400)

# -- -- -- -- -- -- -- -- -- --
# grid_x, grid_t = np.mgrid[-1:1:400j, 0:1:400j]
# u_400to800 = griddata(,u_400, (grid_x, grid_t),method='cubic')

L2err_values = [L2err_1600, L2err_3200, L2err_6400]
numbers_after_underscore = [1600, 3200, 6400]

# Scatter plot
plt.scatter(numbers_after_underscore, L2err_values, color='blue')
plt.title('FDM Independence')
plt.xlabel('Grid size')
plt.ylabel('Mean Square Difference')

divnorm = colors.TwoSlopeNorm(vcenter=0)

plt.figure(figsize=(10, 6))
contour_plot = plt.contourf(t_1600, x_1600, u_1600.T, levels=50, cmap='rainbow', norm=divnorm)  # Inverted the axes
plt.colorbar(contour_plot, label='u(x, t)')  # Add colorbar with label
plt.title('I.C. 2: u(x, t) for N=1600, from FDM')
plt.xlabel('t')  # Updated xlabel
plt.ylabel('x')  # Updated ylabel

plt.figure(figsize=(10, 6))
contour_plot = plt.contourf(t_12800, x_12800, u_12800.T, levels=50, cmap='rainbow', norm=divnorm)  # Inverted the axes
plt.colorbar(contour_plot, label='u(x, t)')  # Add colorbar with label
plt.title('I.C. 2: u(x, t) for N=12800, from FDM')
plt.xlabel('t')  # Updated xlabel
plt.ylabel('x')  # Updated ylabel

plt.show()