import numpy as np
from scipy.io import loadmat

def gen_testdata_E3(): # This function opens the ground truth solution. Need to change path directory for running in ARC.
    data = loadmat("src/allen_cahn/usol_D_0.001_k_5.mat")

    t = data["t"]

    x = data["x"]

    u = data["u"]

    # print(f"gen1 t shape: {t.shape}")
    # print(f"gen1 x shape:{x.shape}")
    # print(f"gen1 u shape:{u.shape}")
    dt = dx = 0.01

    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T

    y = u.flatten()[:, None]
    return X, y

def gen_testdata_E4(): # This function opens the ground truth solution. Need to change path directory for running in ARC.
    data = loadmat("src/allen_cahn/usol_D_0.0001_raissi.mat")
    t = data["tt"]
    x = data["x"]
    u = data["uu"]
    u=u.T
    print(f"gen2 t shape: {t.shape}")
    print(f"gen2 x shape:{x.shape}")
    print(f"gen2 u shape:{u.shape}")
    # dx=1/256
    # dt=0.005
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    
    y = u.flatten()[:, None]
    return X,y

X, y = gen_testdata_E3()

X,y = gen_testdata_E4()