"""
fdm_main.py
In this python script I use odeint to solve the burger's PDE via taking finite differences.
"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import time
# Sine wave:
def generate_sine_wave(invert,amplitude, frequency, N):
    amplitude = amplitude
    frequency = frequency
    invert = invert

    x = np.linspace(-1, 1, N+1)
    y = invert*amplitude * np.sin(np.pi * frequency * x)
    return y
# Cosine wave:
def generate_cos_wave(invert,amplitude, frequency, N):
    amplitude = amplitude
    frequency = frequency
    invert = invert

    x = np.linspace(-1, 1, N+1)
    y = invert*amplitude * np.cos(2 * np.pi * frequency * x)
    return y
# u is a vector with every point being at a different x
N = 1600 # 400 # 4080            # 510, 1020, 2040, 4080
nt = 401  # 101  # 1601           # 200+1, 400+1, 800+1, 1600+1
dx = 2 / N
b = 0.01 / np.pi  # viscosity term


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

# Different initial conditions:
# IC 2 ### IMPORTANT NOTE frequency needs multiplying by 2 as used 2 * frequency * pi in original sin function
# Only for IC2 and IC3
# curve1=generate_sine_wave(-1,0.77,2,N) # so f = 4
# curve2=generate_sine_wave(-1,0.74,5,N) # so f = 10
# curve3=generate_sine_wave(1,1.79,2,N) # f=4
# curve4=generate_sine_wave(1,0.53,5,N) # f =10

# IC 3 #
# curve1=generate_sine_wave(1, 0.62, 0.5,N) # f=1
# curve2=generate_sine_wave(-1, 1.02, 1.5,N) # f=3
# curve3=generate_sine_wave(1, 1.81, 1.0,N) # f=2

#reverted f for these
# IC 4 
# curve1=generate_sine_wave(1,1,2,N) # f=2

# IC 5
curve1=generate_sine_wave(-1,1.5,1,N) #f=1

# Tests
# curve1=generate_sine_wave(1, 1.06, 1.0,N)
# curve2=generate_sine_wave(-1, 1.82, 1.0,N)
# curve3=generate_sine_wave(-1, 0.94, 0.5,N)
# curve4=generate_cos_wave(1, 1.18, 0.75,N)

# curve1=generate_sine_wave(1, 1.58, 2,N)
# curve2=generate_sine_wave(-1, 0.93, 2,N)
# curve3=generate_sine_wave(-1, 0.64, 1,N)
# curve4=generate_sine_wave(-1, 1.95, 1,N)

u0 = curve1
x0 = np.linspace(-1, 1, N + 1) 

# Interior u0 points
u0_int = u0[1:N]

# Define Time Interval
t = np.linspace(0, 1, nt)

print("Begin Calculation...\n")
start_time = time.time()

# Where the solution is computed
u_int = odeint(burgers, u0, t, args=(b,N+2))
# u_int = odeint(burgers, u0_int, t, args=(b, N))
u_int = np.transpose(u_int)
# Writing BC into array
## u[:, 0] = u0
u = np.zeros([N + 1, nt])
u[0, :] = 0
u[N, :] = 0
u[0:N+1, :] = u_int
print(t.shape)
print(x0.shape)
print(u.shape)

#-------------------------------------------------------------------------------
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
cbar.mappable.set_clim(-1, 1)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(np.linspace(-1, 1, N+1), curve1, label='Curve 2')
# plt.plot(np.linspace(-1, 1, N+1), curve3, label='Curve 3')
# plt.plot(np.linspace(-1, 1, N+1), curve4, label='Curve 4')
plt.title('Individual Sine Curves')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.linspace(-1, 1, N+1), u0, label='Sum of Curves', color='red')
plt.title('Sum of Sine Curves')

plt.tight_layout()
plt.show()