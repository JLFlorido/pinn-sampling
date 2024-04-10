import os
import argparse

from tqdm import tqdm
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize_burgers(path, param=None):
    """
    This function animates the Burgers equation

    Args:
    path : path to the desired file
    param: PDE parameter of the data shard to be visualized
    """

    # Read the h5 file and store the data
    if param is not None:
        flnm = "1D_Burgers_Sols_Nu" + str(param) + ".hdf5"
        assert os.path.isfile(path + flnm), 'no such file! '+path + flnm
    else:
        flnm = "1D_Burgers_Sols_Nu0.001.hdf5"

    nb = 0
    with h5py.File(os.path.join(path, flnm), "r") as h5_file:
        xcrd = np.array(h5_file["x-coordinate"], dtype=np.float32)
        print(xcrd.shape)
        data = np.array(h5_file["tensor"], dtype=np.float32)[nb]  # (batch, t, x, channel) --> (t, x, channel)
        print(data.shape)
        quit()
    # Initialize plot
    fig, ax = plt.subplots()

    # Store the plot handle at each time step in the 'ims' list
    ims = []

    for i in tqdm(range(data.shape[0])):
        if i == 0:
            im = ax.plot(xcrd, data[i].squeeze(), animated=True, color="blue")
        else:
            im = ax.plot(xcrd, data[i].squeeze(), animated=True, color="blue")  # show an initial one first
        ax.plot
        ims.append([im[0]])

    # Animate the plot
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save("movie_burgers123.gif", writer=writer)
    print("Animation saved")


# fpath = 'src/data_visualisation/1D_Burgers_Sols_Nu0.001.hdf5'
fpath = 'src/data_visualisation/'

visualize_burgers(fpath, None)
# with h5py.File(fpath, 'r') as file:
#     t_coordinate = file['t-coordinate'][0:201]
#     tensor = file['tensor'][:]
#     x_coordinate = file['x-coordinate'][:]
#     u = tensor[0, :, :] #452
#     # View their shapes
#     # print(f"t-coordinate shape: {t_coordinate.shape}")
#     # print(f"x-coordinate shape: {x_coordinate.shape}")
#     # print(f"tensor shape: {tensor.shape}")
#     # print(t_coordinate)
#     #
#     # plt.figure(figsize=(10, 6))
#     # plt.contourf(x_coordinate, t_coordinate, u, cmap='viridis')
#     # plt.colorbar(label='Solution u')
#     # plt.title('2D Contour Plot of Solution u')
#     # plt.xlabel('x-coordinate')
#     # plt.ylabel('t-coordinate')

# plt.figure(figsize=(10, 6))
# plt.pcolormesh(t_coordinate, x_coordinate, u.T, shading='auto', cmap='rainbow') #viridis
# plt.colorbar(label='Solution u')
# plt.title('2D Contour Plot of Solution u')
# plt.xlabel('t-coordinate')
# plt.ylabel('x-coordinate')
# plt.xlim(0, 1)

# plt.figure(figsize=(8, 6))
# plt.plot(x_coordinate, u[0,:])  # Transpose for correct plotting
# plt.title('Solution u over x at t=0')
# plt.xlabel('x-coordinate')
# plt.ylabel('Solution u')
# plt.grid(True)

# # print(f"u shape: {u.shape}")
# print(u[:,0])
# plt.show()