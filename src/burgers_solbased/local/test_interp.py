import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

# Generate a sample 800x201 matrix
x = np.linspace(-1, 1, 800)
t = np.linspace(0, 1, 201)
u = np.add.outer(t, x)

# Create the interpolation function
interpolator = interp2d(x, t, u, kind='linear')

# Define the new grid for the reduced 256x101 matrix
new_x = np.linspace(-1, 1, 256)
new_t = np.linspace(0, 1, 101)

# Interpolate values on the new grid
new_u = interpolator(new_x, new_t)

# Plot the original and reduced matrices
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.imshow(u, extent=[-1, 1, 0, 1], aspect='auto', cmap='viridis')
plt.title('Original Matrix')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.imshow(new_u, extent=[-1, 1, 0, 1], aspect='auto', cmap='viridis')
plt.title('Reduced Matrix')
plt.colorbar()

plt.tight_layout()
plt.show()