"""Run PINN to solve Burger's Equation using adaptive resampling via REPlacement (REP) based on residual or gradient/curvature information. 
REP-HPC_singleresample.py Trying to figure out why error not working. First, looking at PDF for different sources of info. For this, need to output p(x) (err_eq in the code).

Usage:
    REP-HPC_singleresample.py [--k=<hyp_k>] [--c=<hyp_c>] [--N=<NumDomain>] [--L=<NumResamples> ] [--IM=<InitialMethod>] [--DEP=<depth>] [--INP1=<input1>]
    REP-HPC_singleresample.py -h | --help
Options:
    -h --help                   Display this help message
    --k=<hyp_k>                 Hyperparameter k [default: 1]
    --c=<hyp_c>                 Hyperparameter c [default: 1]
    --N=<NumDomain>             Number of collocation points for training [default: 2000]
    --L=<NumResamples>          Number of times points are resampled [default: 100]
    --IM=<InitialMethod>        Initial distribution method from: "Grid","Random","LHS", "Halton", "Hammersley", "Sobol" [default: Hammersley]
    --DEP=<depth>               Depth of the network [default: 3]
    --INP1=<input1>             First info source, "uxt", "uxut1" etc... [default: residual]
"""
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Initial imports and some function definitions.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
from docopt import docopt
import skopt
from distutils.version import LooseVersion
import deepxde as dde
from deepxde.backend import tf
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import time

os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
dde.config.set_default_float("float64")
# dde.optimizers.config.set_LBFGS_options(maxiter=1000)

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

def quasirandom(n_samples, sampler): # This function creates pseudorandom distributions if initial method is specified.
    space = [(-1.0, 1.0), (0.0, 1.0)]
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        # Remove the first point [0, 0, ...] and the second point [0.5, 0.5, ...], which
        # are too special and may cause some error.
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler = skopt.sampler.Sobol(skip=0, randomize=False)
            return np.array(sampler.generate(space, n_samples + 2)[2:])
    return np.array(sampler.generate(space, n_samples))

def append_to_file(file_path, data):
    try:    
        with open(file_path, 'ab') as file:
            np.savetxt(file,data)
    except Exception as e:
        print(f"An exception occurred: {e}")
        print("Retrying in 5 seconds...")
        time.sleep(5)
        try:
            with open(file_path, 'ab') as file:
                np.savetxt(file, data)
        except Exception as e2:
            print(f"An exception occurred again: {e2}")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main code start
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(k=1, c=1, NumDomain=2000, NumResamples=100, method="Random", depth=3, input1="residual"): # Main Code
    print(f"k equals {k}")
    print(f"c equals {c}")
    print(f"NumDomain equals {NumDomain}")
    print(f"Method equals {method}")
    print(f"Depth equals {depth}")
    print(f"Input 1 equals {input1}")
    start_t = time.time() #Start time.

    def pde(x, y): # Define Burgers PDE
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx
    def du_xt(x,y): # Returns curvature in xt
        return dde.grad.hessian(y,x,i=1,j=0)

    X_test, y_true, itp = gen_testdata() # Ground Truth Solution. (25600,2) coordinates and corresponding (25600,1) values of u.

    # This chunk of code describes the problem using dde structure. Varies depending on prescribed initial distribution.
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    if method == "Grid":
        data = dde.data.TimePDE(
            geomtime, pde, [], num_domain=NumDomain, num_test=10000, train_distribution="uniform"
        )
    elif method == "Random":
        data = dde.data.TimePDE(
            geomtime, pde, [], num_domain=NumDomain, num_test=10000, train_distribution="pseudo"
        )
    elif method in ["LHS", "Halton", "Hammersley", "Sobol"]:
        sample_pts = quasirandom(NumDomain, method)
        data = dde.data.TimePDE(
            geomtime,
            pde,
            [],
            num_domain=0, # Raises eyebrows. Need to check this doesn't cause issue.
            num_test=10000,
            train_distribution="uniform",
            anchors=sample_pts,
        )

    net = dde.maps.FNN([2] + [64] * depth + [1], "tanh", "Glorot normal") # This defines the NN layers, their size and activation functions.

    def output_transform(x, y): # BC
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
    net.apply_output_transform(output_transform)
    
    # Initial Training before resampling
    model = dde.Model(data, net)
    print("Initial 15000 Adam steps")
    model.compile("adam", lr=0.001)
    model.train(epochs=15000, display_every=1000)
    print("Initial L-BFGS steps")
    model.compile("L-BFGS")
    model.train(epochs=1000, display_every=100)

    # Measuring error after initial phase. This information is not used by network to train.
    y_pred = model.predict(X_test)
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    
    print("Finished initial steps. ")
    print(f"l2_relative_error: {l2_error}")

    X = geomtime.random_points(100000)
    if input1 == "residual":
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
    elif input1 == "uxt" or input1 == "utx":
        Y = np.abs(model.predict(X, operator=du_xt)).astype(np.float64)
    elif input1 == "error":
        y_true_resample = itp(X[:,[1,0]])
        y_pred = model.predict(X)
        y_pred = np.squeeze(y_pred)
        Y = np.abs(y_true_resample - y_pred) 

    err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
    print(err_eq.shape)
    err_eq_normalized = (err_eq / sum(err_eq))[:]
    print(err_eq_normalized.shape)
    err_eq_normalized = np.squeeze(err_eq_normalized)
    X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
    X_selected = X[X_ids]
    data.replace_with_anchors(X_selected) # Change current points with selected X points

    model.compile("adam", lr=0.001)
    model.train(epochs=1000, display_every=100)
    model.compile("L-BFGS")
    losshistory, train_state = model.train(epochs=4000, display_every=100)
        
    y_pred = model.predict(X_test)
    error_final = dde.metrics.l2_relative_error(y_true, y_pred)
    time_taken = (time.time()-start_t)

    dde.saveplot(losshistory, train_state, issave=True, isplot=False, 
                 loss_fname=f"singleresample_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_loss_info.dat", 
                 train_fname=f"singleresample_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalpoints.dat", 
                 test_fname=f"singleresample_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalypred.dat",
                # output_dir = "results/raw/performance_results")
                 output_dir="../results/additional_info")

    return error_final, time_taken
 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calling main and saving results
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__)  #Importing all arguments from docopts
    c=float(args['--c'])
    k=float(args['--k'])
    NumDomain=int(args['--N'])
    NumResamples=int(args['--L'])
    method=str(args['--IM'])
    depth=int(args["--DEP"])
    input1=str(args["--INP1"])

    error_final, time_taken = main(c=c, k=k, NumDomain=NumDomain,NumResamples=NumResamples,method=method, depth=depth, input1=input1) # Run main, record error history and final accuracy.

    if np.isscalar(time_taken):
        time_taken = np.atleast_1d(time_taken)
    if np.isscalar(error_final):
        error_final = np.atleast_1d(error_final) # Ensures in correct format for saving.
    
    output_dir = "../results/performance_results"  # Running on Arc
    # output_dir = "results/raw/performance_results"  # Running Locally
    error_final_fname = f"singleresample_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_error_final.txt"
    time_taken_fname = f"singleresample_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_time_taken.txt"
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    error_final_fname = os.path.join(output_dir, error_final_fname)
    time_taken_fname = os.path.join(output_dir, time_taken_fname)

    append_to_file(error_final_fname, error_final)
    append_to_file(time_taken_fname, time_taken)