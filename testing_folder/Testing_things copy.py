"""
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def gen_testdata(): # This function opens the ground truth solution. Need to change path directory for running in ARC.
    data = np.load("./Burgers2.npz")
    t, x, exact = data["t"], data["x"], data["exact"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    x=np.squeeze(x)
    t=np.squeeze(t)
    exact=np.squeeze(exact)
    itp = RegularGridInterpolator( (t, x), exact, method='linear', bounds_error=False, fill_value=None)
    return X, y, itp

def gen_testdata_original(): # This function opens the ground truth solution. Need to change path directory for running in ARC.
    data = np.load("./Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    x=np.squeeze(x)
    t=np.squeeze(t)
    exact=np.squeeze(exact)
    itp = RegularGridInterpolator( (t, x), exact, method='linear', bounds_error=True, fill_value=None)
    return X, y, itp

X,y,itp = gen_testdata_original()
X=X[:,[1,0]] #t,x for interpolator
print()
test1 = itp([0.1,0]) # this is -1 t, 0x >> 0
test2 = itp([1,1]) # this is 0 t, -1x >> 
print(test1)   
print(test2)
# y_true_resample = itp(X[:,[1,0]])