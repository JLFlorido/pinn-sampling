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
def generate_sine_wave(invert,amplitude, frequency, N):
    amplitude = amplitude
    frequency = frequency
    invert = invert

    x = np.linspace(-1, 1, N+1)
    y = invert*amplitude * np.sin(2 * np.pi * frequency * x)
    return y

# curve1=generate_sine_wave(-1,0.77,2,N)
# curve2=generate_sine_wave(-1,0.74,5,N)
# curve3=generate_sine_wave(1,1.79,2,N)
# curve4=generate_sine_wave(1,0.53,5,N)

curve1=generate_sine_wave(1, 1.7, 4,N)
curve2=generate_sine_wave(-1, 1.73, 3,N)
curve3=generate_sine_wave(1, 1.81, 2,N)
curve4=generate_sine_wave(-1, 1.08, 5,N)

u0 = curve1 + curve2 + curve3 + curve4
u0= (u0 - np.min(u0)) / (np.max(u0) - np.min(u0))
x0 = np.linspace(-1, 1, N + 1) 

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

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(np.linspace(-1, 1, N+1), curve1, label='Curve 1')
plt.plot(np.linspace(-1, 1, N+1), curve2, label='Curve 2')
plt.plot(np.linspace(-1, 1, N+1), curve3, label='Curve 3')
plt.plot(np.linspace(-1, 1, N+1), curve4, label='Curve 4')
plt.title('Individual Sine Curves')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.linspace(-1, 1, N+1), u0, label='Sum of Curves', color='red')
plt.title('Sum of Sine Curves')

plt.tight_layout()
plt.show()