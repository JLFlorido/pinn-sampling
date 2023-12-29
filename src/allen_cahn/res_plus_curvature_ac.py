"""Run PINN to solve Allen Cahn equation. Only use D=E-3, as the other case is for periodic BC and I've not figured out how to do those yet.
res_plus_curvature_ac.py AC but res plus INP1
Usage:
    res_plus_curvature_ac.py [--D=<DiffCoeff>] [--k=<hyp_k>] [--c=<hyp_c>] [--N=<NumDomain>] [--L=<NumResamples> ] [--IM=<InitialMethod>] [--DEP=<depth>] [--INP1=<input1>]
    res_plus_curvature_ac.py -h | --help
Options:
    -h --help                   Display this help message
    --D=<DiffCoeff>             I think this parameter indicates strength of diffusion. E-3 in Wu, E-4 elsewhere. [default: 0.001]
    --k=<hyp_k>                 Hyperparameter k [default: 1]
    --c=<hyp_c>                 Hyperparameter c [default: 1]
    --N=<NumDomain>             Number of collocation points for training [default: 2000]
    --L=<NumResamples>          Number of times points are resampled [default: 100]
    --IM=<InitialMethod>        Initial distribution method from: "Grid","Random","LHS", "Halton", "Hammersley", "Sobol" [default: Random]
    --DEP=<depth>               Depth of the network [default: 3]
    --INP1=<input1>             Info source, "uxt", "uxut1" etc... [default: residual]
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
import numpy as np
from scipy.io import loadmat
import time

os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=1000)

def gen_testdata_E3(): # This function opens the ground truth solution. Commented out directory is for local
    data = loadmat("usol_D_0.001_k_5.mat") # data = loadmat("src/allen_cahn/usol_D_0.001_k_5.mat")
    t = data["t"]
    x = data["x"]
    u = data["u"]
    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y

def gen_testdata_E4(): # This function opens the ground truth solution. Commented out directory is for local
    data = loadmat("usol_D_0.0001_raissi.mat") # data = loadmat("src/allen_cahn/usol_D_0.0001_raissi.mat")
    t = data["tt"]
    x = data["x"]
    u = data["uu"]
    u=u.T
    dx=1/256
    dt=0.005
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main code start
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def main(diff=0.001, k=1, c=1, NumDomain=2000, NumResamples=100, method="Random", depth=3, input1="residual"): # Main Code
    print(f"D equals {diff}")
    print(f"k equals {k}")
    print(f"c equals {c}")
    print(f"NumDomain equals {NumDomain}")
    print(f"NumResamples equals {NumResamples}")
    print(f"Method equals {method}")
    print(f"Depth equals {depth}")
    print(f"Input1 equals {input1}")
    start_t = time.time() #Start time.

    if diff==0.001:
        def pde(x, y): # Define Allen Cahn Equation
            u = y
            du_xx = dde.grad.hessian(y, x, i=0, j=0)
            du_t = dde.grad.jacobian(y, x, i=0, j=1)
            return du_t - 0.001 * du_xx + 5 * (u**3 - u) # Raissi, Agnastopoulos use E-4. Wu use E-3.
        def dpde_dxt(x,y): #residual wrt y
            u = y
            du_xx = dde.grad.hessian(y, x, i=0, j=0)
            du_t = dde.grad.jacobian(y, x, i=0, j=1)
            pde = du_t - 0.001 * du_xx + 5 * (u**3 - u)
            return dde.grad.hessian(pde, x, i=1, j=0)
    elif diff==0.0001:
        def pde(x, y): # Define Allen Cahn Equation
            u = y
            du_xx = dde.grad.hessian(y, x, i=0, j=0)
            du_t = dde.grad.jacobian(y, x, i=0, j=1)
            return du_t - 0.0001 * du_xx + 5 * (u**3 - u) # Raissi, Agnastopoulos use E-4. Wu use E-3.
        def dpde_dxt(x,y): #residual wrt y
            u = y
            du_xx = dde.grad.hessian(y, x, i=0, j=0)
            du_t = dde.grad.jacobian(y, x, i=0, j=1)
            pde = du_t - 0.0001 * du_xx + 5 * (u**3 - u)
            return dde.grad.hessian(pde, x, i=1, j=0)
    else:
        raise ValueError("Invalid value for 'diff'. Please use either 0.001 or 0.0001.")
        
    def dudx(x,y): # Returns gradient in x
        return dde.grad.jacobian(y, x, i=0, j=0)
    def dudt(x,y): # Returns gradient in y
        return dde.grad.jacobian(y,x, i=0, j=1)
    def du_xt(x,y): # Returns curvature in xt
        return dde.grad.hessian(y,x,i=1,j=0)

    if diff == 0.001:
        X_test, y_true = gen_testdata_E3()
    elif diff == 0.0001: 
        X_test,y_true = gen_testdata_E4()
    else:
        raise ValueError("Invalid value for 'diff'. Please use either 0.001 or 0.0001.")
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
            num_domain=0,
            num_test=10000,
            train_distribution="uniform",
            anchors=sample_pts,
        )

    net = dde.maps.FNN([2] + [64] * depth + [1], "tanh", "Glorot normal") # This defines the NN layers, their size and activation functions.

    def output_transform(x, y):
        x_in = x[:,0:1]
        t_in = x[:,1:2]
        return t_in * (1 + x_in) * (1 - x_in) * y + tf.square(x_in) * tf.cos(np.pi * x_in)
    net.apply_output_transform(output_transform)
    
    # Initial Training before resampling
    model = dde.Model(data, net)
    print("Initial 15000 Adam steps")
    model.compile("adam", lr=0.001)
    model.train(epochs=15000, display_every=300000)
    print("Initial L-BFGS steps")
    model.compile("L-BFGS")
    model.train(display_every=300000)

    # Measuring error after initial phase. This information is not used by network to train.
    y_pred = model.predict(X_test)
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    
    print("Finished initial steps. ")
    print(f"l2_relative_error: {l2_error}")

    for i in range(NumResamples): # Resampling loop begins. 100 is default, first run took ~4 hours...
        X = geomtime.random_points(100000)

        # --- Below, all the different info sources for resampling
        if input1 == "residual":
            Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        elif input1 == "uxt" or input1 == "utx":
            Y = np.abs(model.predict(X, operator=du_xt)).astype(np.float64)
        elif input1 == "uxut1":
            Y1 = np.abs(model.predict(X, operator=dudx)).astype(np.float64)
            Y2 = np.abs(model.predict(X, operator=dudt)).astype(np.float64)
            Y = (Y1+Y2)
        elif input1 == "uxut2":
            Y1 = np.abs(model.predict(X, operator=dudx)).astype(np.float64)
            Y2 = np.abs(model.predict(X, operator=dudt)).astype(np.float64)
            Y = (Y1+Y2)/2
        elif input1 == "uxut3":
            Y1 = np.abs(model.predict(X, operator=dudx)).astype(np.float64)
            Y2 = np.abs(model.predict(X, operator=dudt)).astype(np.float64)
            Y = np.maximum(Y1,Y2)
        elif input1 == "uxut4":
            Y1 = np.abs(model.predict(X, operator=dudx)).astype(np.float64)
            Y2 = np.abs(model.predict(X, operator=dudt)).astype(np.float64)
            Y = np.sqrt((Y1**2)+(Y2**2))
        elif input1 == "pdedxt":
            Y = np.abs(model.predict(X, operator=dpde_dxt))

        Y = Y/sum(Y)
        res = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        res = res/sum(res)
        Y = Y + res
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        X_selected = X[X_ids]

        data.replace_with_anchors(X_selected) # Change current points with selected X points

        model.compile("adam", lr=0.001)
        model.train(epochs=1000, display_every=300000)
        model.compile("L-BFGS")
        losshistory, train_state = model.train(display_every=300000)

        print("!\nFinished loop #{}\n".format(i+1))

    y_pred = model.predict(X_test)
    error_final = dde.metrics.l2_relative_error(y_true, y_pred)
    time_taken = (time.time()-start_t)
    
    # dde.saveplot(losshistory, train_state, issave=True, isplot=False, 
    #              loss_fname=f"allencahn_{diff}_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_loss_info.dat", 
    #              train_fname=f"allencahn_{diff}_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalpoints.dat", 
    #              test_fname=f"allencahn_{diff}_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalypred.dat",
    #              output_dir="../results/additional_info")
    return error_final, time_taken
 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calling main and saving results
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__)
    diff=float(args['--D'])
    c=float(args['--c'])
    k=float(args['--k'])
    NumDomain=int(args['--N'])
    NumResamples=int(args['--L'])
    method=str(args['--IM'])
    depth=int(args["--DEP"])
    input1=str(args["--INP1"])

    error_final, time_taken = main(diff=diff, c=c, k=k, NumDomain=NumDomain,NumResamples=NumResamples,method=method, depth=depth, input1=input1) # Run main, record error history and final accuracy.

    if np.isscalar(time_taken):
        time_taken = np.atleast_1d(time_taken)
    if np.isscalar(error_final):
        error_final = np.atleast_1d(error_final)
    
    output_dir = "../results/performance_results/allen_cahn"  # Replace with your desired output directory path
    error_final_fname = f"allencahn_{diff}_res_plus_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_error_final.txt"
    time_taken_fname = f"allencahn_{diff}_res_plus_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_time_taken.txt"
    
    # If results directory does not exist, create it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Define the full file paths
    error_final_fname = os.path.join(output_dir, error_final_fname)
    time_taken_fname = os.path.join(output_dir, time_taken_fname)
    
    # Define function to append to file. The try/exception was to ensure when ran as task array that saving won't fail in the rare case that
    # the file is locked for saving by a different job.
    def append_to_file(file_path, data):
        try:    
            with open(file_path, 'ab') as file:
                np.savetxt(file,data)
                print(f"Saved to \"{output_dir}\" directory successfully")
        except Exception as e:
            print(f"An exception occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            try:
                with open(file_path, 'ab') as file:
                    np.savetxt(file, data)
                    print(f"Saved to \"{output_dir}\" directory successfully")
            except Exception as e2:
                print(f"An exception occurred again: {e2}")

    # Use function to append to file.
    append_to_file(error_final_fname, error_final)
    append_to_file(time_taken_fname, time_taken)