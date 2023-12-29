import h5py
import matplotlib.pyplot as plt
fpath = 'src/data_visualisation/1D_Burgers_Sols_Nu0.001.hdf5'
with h5py.File(fpath, 'r') as file:
    t_coordinate = file['t-coordinate'][0:201]
    tensor = file['tensor'][:]
    x_coordinate = file['x-coordinate'][:]
    u = tensor[793, :, :] #452
    # View their shapes
    # print(f"t-coordinate shape: {t_coordinate.shape}")
    # print(f"x-coordinate shape: {x_coordinate.shape}")
    # print(f"tensor shape: {tensor.shape}")
    # print(t_coordinate)
    #
    # plt.figure(figsize=(10, 6))
    # plt.contourf(x_coordinate, t_coordinate, u, cmap='viridis')
    # plt.colorbar(label='Solution u')
    # plt.title('2D Contour Plot of Solution u')
    # plt.xlabel('x-coordinate')
    # plt.ylabel('t-coordinate')

plt.figure(figsize=(10, 6))
plt.pcolormesh(t_coordinate, x_coordinate, u.T, shading='auto', cmap='rainbow') #viridis
plt.colorbar(label='Solution u')
plt.title('2D Contour Plot of Solution u')
plt.xlabel('t-coordinate')
plt.ylabel('x-coordinate')
plt.xlim(0, 1)

plt.figure(figsize=(8, 6))
plt.plot(x_coordinate, u[0,:])  # Transpose for correct plotting
plt.title('Solution u over x at t=0')
plt.xlabel('x-coordinate')
plt.ylabel('Solution u')
plt.grid(True)

# print(f"u shape: {u.shape}")
print(u[:,0])
plt.show()

