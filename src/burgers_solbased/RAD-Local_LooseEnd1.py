"""RAD-Local_LooseEnd1.py
This code is an adaption that only cares about the first 15k iterations, as it wants to look at the PDE residual at this point and the actual error in the field before any resampling.
It should make jpinn take repeat as an argument, and use it to save a different file each time.
Should output u to a grid that's the same size as burgers.npz (100x256) (time * space)
Should output pde res to a grid that's (100x256) as well.
Modified in 9 August to save the loss history too. This way can observe whether resampling occurs at a good time when loss is flattening out.
"""
# ---------------------------------------
# Imports and Function definitions
# ---------------------------------------
import deepxde as dde
import numpy as np
from deepxde.backend import tf
import os
from multiprocessing import Pool

os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=1000)

def apply(func,args=None,kwds=None):
    """
    Launch a new process to call the function.
    This can be used to clear Tensorflow GPU memory after model execution.
    """
    with Pool(1) as p:
        if args is None and kwds is None:
            r = p.apply(func)
        elif kwds is None:
            r = p.apply(func, args=args)
        elif args is None:
            r = p.apply(func, kwds=kwds)
        else:
            r = p.apply(func, args=args, kwds=kwds)
    return r

def gen_testdata(): # This function opens the ground truth solution. Need to change path directory for running in ARC.
    data = np.load("./Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

# ---------------------------------------
# Main function
# ---------------------------------------
def jpinn(k=1, c=1, NumDomain=2000, repeat=1):
    def pde(x, y): # Define Burgers PDE
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx
    X_test, _ = gen_testdata() # (25600,2) coordinates and corresponding (25600,1) values of u.

    # Describe the problem using dde structure
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [],
        num_domain=NumDomain,
        num_test=10000,
        train_distribution="pseudo",
    )
    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal") # This defines the NN layers, their size and activation functions.
    def output_transform(x, y):
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
    net.apply_output_transform(output_transform)

    # Initial Training
    model = dde.Model(data, net)
    print("Initial 15000 Adam steps")
    model.compile("adam", lr=0.001)
    model.train(epochs=15000)
    print("Initial L-BFGS steps")
    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    # Obtain outputs and saving
    y_pred = model.predict(X_test) # Predict u.
    y_res = np.abs(model.predict(X_test, operator=pde)).astype(np.float64) # Predict PDE res over X_test grid
    fname_u=f"./results/raw/loose_end_1/u_result_1{repeat}"
    fname_res=f"./results/raw/loose_end_1/res_result_1{repeat}"
    np.savetxt(fname=fname_u, X=y_pred)
    np.savetxt(fname=fname_res, X=y_res)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, 
                loss_fname=f"NoResample_Random_k{k}c{c}_N{NumDomain}_loss_info.dat", 
                train_fname=f"NoResample_Random_k{k}c{c}_N{NumDomain}_finalpoints.dat", 
                test_fname=f"NoResample_Random_k{k}c{c}_N{NumDomain}_finalypred.dat",
                output_dir="./results/raw/loose_end_1")
    
# ---------------------------------------
# Calling main through apply() to prevent memory issues:)
# ---------------------------------------
def main():
    # define variables
    c=1
    k=1
    NumDomain=2000 # default:2000
    
    for repeat in range(10):  
        apply(jpinn, (c, k, NumDomain,repeat)) # Run main, record error history and final accuracy.

if __name__ == "__main__":
    main()