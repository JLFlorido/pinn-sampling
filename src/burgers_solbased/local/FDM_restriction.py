"""
FDM_restriction.py Restricting via interpolation the solutions from a fine FDM grid so they occupy less memory and are quicker to run for PINNs.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interp2d
IC="3"
v="0.01"
# data_base = np.load("src/burgers_solbased/Burgers.npz")
# data_base2 = np.load("src/burgers_solbased/Burgers2.npz")
# print(data_base["usol"].shape)
# print(data_base2["exact"].shape)
# quit()
# data_12800 = np.load(f"src/burgers_solbased/npzs/Burgers_IC_{IC}_N12800_nt3201.npz")  #for IC
data_12800 = np.load(f"src/burgers_solbased/npzs/Burgers_v{v}_N12800_nt3201.npz") # For V

t_12800, x_12800, u_12800 = data_12800["t"], data_12800["x"], data_12800["exact"]
# Interpolating to 256x100 to compare to it

interpolator_12800 = interp2d(t_12800, x_12800, u_12800, kind='linear')
new_x = np.linspace(-1, 1, 256)
new_t = np.linspace(0, 1, 100)
u_12800to256 = interpolator_12800(new_t,new_x)
# np.savez(f"src/burgers_solbased/Burgers_IC_{IC}.npz", t=new_t, x=new_x, usol=u_12800to256) # IC
np.savez(f"src/burgers_solbased/Burgers_v_{v}.npz", t=new_t, x=new_x, usol=u_12800to256) # viscosity


print(u_12800to256.shape)
# -- -- -- -- -- -- -- -- -- --
# Plot the original and reduced matrices
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.imshow(u_12800, extent=[0, 1, -1, 1], aspect='auto', cmap='rainbow')
plt.title('Original Matrix')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.imshow(u_12800to256, extent=[0, 1, -1, 1], aspect='auto', cmap='rainbow')
plt.title('Reduced Matrix')
plt.colorbar()
plt.show()