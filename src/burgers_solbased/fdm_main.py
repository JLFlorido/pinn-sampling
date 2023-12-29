"""
fdm_main.py
In this python script I use odeint to solve the burger's PDE via taking finite differences.
"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import time

# u is a vector with every point being at a different x
N = 1600 # 400 # 4080            # 510, 1020, 2040, 4080
nt = 401  # 101  # 1601           # 200+1, 400+1, 800+1, 1600+1
dx = 2 / N
b = 0.01 / np.pi  # viscosity term

print("Begin Calculation...\n")
start_time = time.time()
# Define Function to pass to odeint
def burgers(u_int, t, b, N):
    u_local = np.zeros([N + 1])
    u_local[1:N] = u_int
    dudt = np.zeros([N - 1])

    for i in range(1, N):
        dudt[i - 1] = b * (u_local[i + 1] - 2 * u_local[i] + u_local[i - 1]) / (
            dx * dx
        ) - u_local[i] * (u_local[i + 1] - u_local[i - 1]) / (
            2 * dx
        )  # Changed and commented out previous version
    return dudt


# Define Initial Conditions
# Sine wave:
x0 = np.linspace(-1, 1, N + 1) 
u0 = -np.sin(np.pi * x0)
print(u0.shape)
u0 =np.array([-0.5546515, -0.5816987, -0.61135036, -0.64371306, -0.67871433, -0.71597695,
 -0.7546634, -0.79336095, -0.83013034, -0.8628186, -0.88958085, -0.90934026,
 -0.92193127, -0.9278887, -0.9282956, -0.9240429, -0.91630363, -0.9059326,
 -0.89365745, -0.88005584, -0.8655763, -0.85056096, -0.8352698, -0.81989604,
 -0.8045846, -0.78944093, -0.7745417, -0.759941, -0.74567616, -0.73177177,
 -0.7182424, -0.70509547, -0.69233286, -0.6799524, -0.66794896, -0.65631515,
 -0.6450422, -0.6341198, -0.62353736, -0.6132839, -0.6033478, -0.593718,
 -0.5843832, -0.5753321, -0.5665543, -0.5580391, -0.54977685, -0.541757,
 -0.5339702, -0.5264072, -0.51905966, -0.5119194, -0.50497776, -0.49822754,
 -0.49166125, -0.485272, -0.47905302, -0.472998, -0.46710062, -0.46135515,
 -0.45575607, -0.4502981, -0.44497612, -0.4397852, -0.4347208, -0.42977852,
 -0.42495403, -0.42024317, -0.41564214, -0.41114748, -0.4067554, -0.4024625,
 -0.3982657, -0.39416167, -0.39014745, -0.38622028, -0.3823774, -0.37861612,
 -0.37493393, -0.37132844, -0.3677974, -0.36433843, -0.36094943, -0.3576284,
 -0.3543732, -0.35118216, -0.34805337, -0.34498498, -0.34198165, -0.34002754,
 -0.7733372, -0.77113885, -0.7646126, -0.7581747, -0.75182635, -0.74556756,
 -0.73939705, -0.7333148, -0.72731996, -0.7214113, -0.7155881, -0.7098492,
 -0.7041938, -0.69862044, -0.6931278, -0.6877149, -0.6823806, -0.6771233,
 -0.6719414, -0.66683453, -0.6618008, -0.65683883, -0.65194786, -0.64712644,
 -0.64237344, -0.6376874, -0.6330668, -0.62851125, -0.62401915, -0.6195889,
 -0.61521983, -0.61091065, -0.60666066, -0.6024685, -0.5983329, -0.59425277,
 -0.59022766, -0.5862561, -0.5823372, -0.57847, -0.5746535, -0.57088673,
 -0.5671688, -0.5634989, -0.559876, -0.55629957, -0.5527687, -0.5492823,
 -0.5458398, -0.54244035, -0.5390831, -0.5357674, -0.53249246, -0.5292576,
 -0.5260623, -0.52290547, -0.51978666, -0.5167054, -0.51366067, -0.51065207,
 -0.507679, -0.50474054, -0.5018365, -0.49896592, -0.49612853, -0.49332386,
 -0.4905511, -0.48780993, -0.4850996, -0.48241973, -0.47976986, -0.47714943,
 -0.47455803, -0.47199515, -0.4694604, -0.46695325, -0.46447328, -0.46201983,
 -0.45959297, -0.4571919, -0.45481634, -0.45246583, -0.45014015, -0.44783872,
 -0.4455613, -0.4433075, -0.4410767, -0.43887663, -0.4367386, -0.6270471,
 -0.6800066, -0.6768351, -0.67352855, -0.6702531, -0.6670084, -0.66379374,
 -0.6606091, -0.6574541, -0.6543282, -0.651231, -0.6481624, -0.6451221,
 -0.64210933, -0.639124, -0.63616586, -0.6332344, -0.6303293, -0.6274503,
 -0.624597, -0.6217693, -0.6189662])
def double_points_linear_interpolation(original_array):
    doubled_array = []

    for i in range(len(original_array) - 1):
        doubled_array.append(original_array[i])
        # Linear interpolation to insert a new point
        new_point = (original_array[i] + original_array[i + 1]) / 2.0
        doubled_array.append(new_point)

    # Add the last point from the original array
    doubled_array.append(original_array[-1])

    return np.array(doubled_array)
u0 = double_points_linear_interpolation(u0)
print(u0.shape)
u0 = double_points_linear_interpolation(u0)
print(u0.shape)
u0 = double_points_linear_interpolation(u0)
# print(u0.shape)

u0_int = u0[1:N]

# Define Time Interval
t = np.linspace(0, 1, nt)

# print(len(t))
# Where the solution is computed
# sol = odeint(burgers, u0, t, args=(b))
u_int = odeint(burgers, u0_int, t, args=(b, N))
u_int = np.transpose(u_int)
# Writing BC into array
## u[:, 0] = u0
u = np.zeros([N + 1, nt])
u[0, :] = 0
u[N, :] = 0
u[1:N, :] = u_int
print(t.shape)
print(x0.shape)
print(u.shape)

# Print time taken for main code to run
print("--- %s seconds ---" % (time.time() - start_time))

# Export Data
# np.savez(f"Burgers_N{N}_nt{nt}.npz", t=t, x=x0, exact=u)
print("\n --- Data Exported ---")

# Plot graph
fig = plt.figure(figsize=(7, 4))
plt.pcolormesh(t, x0, u, cmap="rainbow")
plt.xlabel("t")
plt.ylabel("x")
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label("u(t,x)")
# cbar.mappable.set_clim(-1, 1)
plt.show()

