import numpy as np
import matplotlib.pyplot as plt

# Function to generate a sine wave with random amplitude
def generate_sine_wave():
    amplitude = round(np.random.uniform(0.5, 2.0),2)  # Random amplitude between 0.5 and 2.0
    frequency = np.random.randint(1, 6)  # Random frequency between 1 and 5
    invert = np.random.choice([1, -1])
    phase = np.random.uniform(0, 2 * np.pi)  # Random phase between 0 and 2*pi

    x = np.linspace(-1, 1, 1000)
    # y = amplitude * np.sin(2 * np.pi * frequency * x + phase)
    y = invert*amplitude * np.sin(2 * np.pi * frequency * x)
    # y = amplitude * np.sin(2 * np.pi * x)
    
    print(f"{invert}, {amplitude}, {frequency}")
    return y

# Generate four sine waves and add them
curve1 = generate_sine_wave()
curve2 = generate_sine_wave()
curve3 = generate_sine_wave()
curve4 = generate_sine_wave()

sum_curve = curve1 + curve2 + curve3 + curve4

# Plot the individual curves and their sum
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(np.linspace(-1, 1, 1000), curve1, label='Curve 1')
plt.plot(np.linspace(-1, 1, 1000), curve2, label='Curve 2')
plt.plot(np.linspace(-1, 1, 1000), curve3, label='Curve 3')
plt.plot(np.linspace(-1, 1, 1000), curve4, label='Curve 4')
plt.title('Individual Sine Curves')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.linspace(-1, 1, 1000), sum_curve, label='Sum of Curves', color='red')
plt.title('Sum of Sine Curves')

plt.tight_layout()
plt.show()
