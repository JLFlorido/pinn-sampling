"""Run PINN to solve Burger's Equation in 2D using adaptive resampling.
main_be2d.py. 

Usage:
    main_be2d.py [--k=<hyp_k>] [--c=<hyp_c>] [--N=<NumDomain>] [--L=<NumResamples> ] [--IM=<InitialMethod>] [--DEP=<depth>] [--INP1=<input1>]
    main_be2d.py -h | --help
Options:
    -h --help                   Display this help message
    --k=<hyp_k>                 Hyperparameter k [default: 1]
    --c=<hyp_c>                 Hyperparameter c [default: 1]
    --N=<NumDomain>             Number of collocation points for training [default: 200]
    --L=<NumResamples>          Number of times points are resampled [default: 2]
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
import time
# import matplotlib.pyplot as plt

# os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
# dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=1000)

def gen_testdata(): # This function opens the ground truth solution at a fixed uniform set of points.
    X = np.load("./src/burgers_2d/X.npy")
    X = X[0,:]
    Y = np.load("./src/burgers_2d/Y.npy")
    Y = Y[:,0]
    T = np.arange(0,1.1,0.1)
    xx,yy,tt = np.meshgrid(X,Y,T)
    xyt = np.vstack((np.ravel(xx), np.ravel(yy),  np.ravel(tt))).T
    results_u = np.load("./src/burgers_2d/results_u.npy")
    u = results_u.flatten()[:,None]
    results_v = np.load("./src/burgers_2d/results_v.npy")
    v = results_v.flatten()[:,None]
    return xyt, u, v

def quasirandom(n_samples, sampler): # This function creates pseudorandom distributions if initial method is specified.
    space = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
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

def main(k=1, c=1, NumDomain=2000, NumResamples=100, method="Random", depth=3, input1="residual"): # Main Code
    print(f"k equals {k}")
    print(f"c equals {c}")
    print(f"NumDomain equals {NumDomain}")
    print(f"Method equals {method}")
    print(f"Depth equals {depth}")
    print(f"Input1 equals {input1}")
    start_t = time.time() #Start time.

    def pde(x, u): # Define Burgers PDE; x has components x,y,t. u has components u and v.
        u_vel, v_vel = u[:, 0:1], u[:, 1:2] # This is needed for calculation of pde_u and pde_v
        du_x = dde.grad.jacobian(u, x, i=0, j=0) # For Jacobian, i dicates u or v, j dictates x, y or t
        du_t = dde.grad.jacobian(u, x, i=0, j=2)
        du_xx = dde.grad.hessian(u, x,component=0, i=0, j=0) # For hessian, component dictates u or v and i,j dictates x,y,t
        du_y = dde.grad.jacobian(u, x, i=0, j=1)
        du_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)
        pde_u = du_t + (u_vel * du_x) + (v_vel * du_y) - (0.01 / np.pi * du_xx) - (0.01 / np.pi * du_yy)
        
        dv_x = dde.grad.jacobian(u, x, i=1, j=0)
        dv_t = dde.grad.jacobian(u, x, i=1, j=2)
        dv_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
        dv_y = dde.grad.jacobian(u, x, i=1, j=1)
        dv_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)
        pde_v = dv_t + (u_vel * dv_x) + (v_vel * dv_y) - (0.01 / np.pi * dv_xx) - (0.01 / np.pi * dv_yy)
        return [pde_u, pde_v]
    
    # This chunk of code describes the problem using dde structure. Varies depending on prescribed initial distribution.
    spacedomain = dde.geometry.Rectangle(xmin=[0, 0],xmax=[1,1]) # The x,y domain is a rectangle with corners 0,0 to 1,1.
    timedomain = dde.geometry.TimeDomain(0, 1) # Time domain is a line from 0 to 1.
    geomtime = dde.geometry.GeometryXTime(spacedomain, timedomain)

    # if method == "Grid":
    #     data = dde.data.TimePDE(
    #         geomtime, pde, [], num_domain=NumDomain, num_test=10000, train_distribution="uniform"
    #     )
    # elif method == "Random":
    #     data = dde.data.TimePDE(
    #         geomtime, pde, [], num_domain=NumDomain, num_test=10000, train_distribution="pseudo"
    #     )
    # elif method in ["LHS", "Halton", "Hammersley", "Sobol"]:
    #     sample_pts = quasirandom(NumDomain, method)
    #     data = dde.data.TimePDE(
    #         geomtime,
    #         pde,
    #         [],
    #         num_domain=0, # Raises eyebrows. Need to check this doesn't cause issue.
    #         num_test=10000,
    #         train_distribution="uniform",
    #         anchors=sample_pts,
    #     )
    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        [], 
        num_domain=NumDomain, 
        num_test=None, 
        train_distribution="pseudo"
    )
    # net = dde.maps.FNN([3] + [64] * depth + [2], "tanh", "Glorot normal") # 3 Input nodes for x,y and t; 2 outputs for u and v.
    net = dde.nn.FNN([3] + [64] * depth + [2], "tanh", "Glorot normal") # 3 Input nodes for x,y and t; 2 outputs for u and v.

    def output_transform(x, y): # BC

        u = tf.sin(np.pi * x[:, 0:1]) * tf.cos(np.pi * x[:, 1:2]) + (tf.sin(np.pi * x[:, 0:1])) * (x[:, 2:3]) * y[:,0]
        v = tf.cos(np.pi * x[:, 0:1]) * tf.sin(np.pi * x[:, 1:2]) + (tf.sin(np.pi * x[:, 1:2])) * (x[:, 2:3]) * y[:,1]
        
        return dde.backend.stack((u, v), axis=1)
    # net.apply_output_transform(output_transform)
    
    model = dde.Model(data, net)


    # Initial Training before resampling
    print("Initial 15000 Adam steps")
    model.compile("adam", lr=0.001)
    model.train(epochs=15, display_every=1000)
    # print("Initial L-BFGS steps")
    # model.compile("L-BFGS")
    # model.train(display_every=1000)
    print("Initial training done with no errors")
    # Measuring error after initial phase. This information is not used by network to train.
    xyt, u_true, v_true = gen_testdata()
    print(xyt.shape)
    pred = model.predict(xyt)
    print(pred.shape)
    # print(pred_u.shape)
    # print(pred_v.shape)
    # print(pred_u)
    print("What does y_pred look like...")
    quit()
    print(u_pred.shape)
    print(v_pred.shape)
    print(u_pred)



    l2_error_u = dde.metrics.l2_relative_error(u_true, u_pred)
    l2_error_v = dde.metrics.l2_relative_error(v_true, v_pred)

    print("Finished initial steps. ")
    print(f"l2_relative_error_u: {l2_error_u}")
    print(f"l2_relative_error_v: {l2_error_v}")

    quit()

    for i in range(NumResamples): # Resampling loop begins. 100 is default, first run took ~4 hours...
        X = geomtime.random_points(100000)
        # --- Below, all the different info sources for resampling
        if input1 == "residual" or input1 == "pde":
            Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        X_selected = X[X_ids]

        data.replace_with_anchors(X_selected) # Change current points with selected X points

        model.compile("adam", lr=0.001)
        model.train(epochs=1000, display_every=300000)
        model.compile("L-BFGS")
        model.train(display_every=300000)

        print("!\nFinished loop #{}\n".format(i+1))

    y_pred = model.predict(X_test)
    error_final = dde.metrics.l2_relative_error(y_true, y_pred)
    time_taken = (time.time()-start_t)
    
    # dde.saveplot(losshistory, train_state, issave=False, isplot=False, 
    #              loss_fname=f"replacement_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_loss_info.dat", 
    #              train_fname=f"replacement_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalpoints.dat", 
    #              test_fname=f"replacement_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalypred.dat",
    #              output_dir="../results/additional_info")
    return error_final, time_taken
 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calling main and saving results
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__)
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
        error_final = np.atleast_1d(error_final)
    
    output_dir = "../results/performance_results"  # Replace with your desired output directory path
    error_final_fname = f"replacement_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_error_final.txt"
    time_taken_fname = f"replacement_{input1}_D{depth}_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_time_taken.txt"
    
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
        except Exception as e:
            print(f"An exception occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            try:
                with open(file_path, 'ab') as file:
                    np.savetxt(file, data)
            except Exception as e2:
                print(f"An exception occurred again: {e2}")

    # Use function to append to file.
    append_to_file(error_final_fname, error_final)
    append_to_file(time_taken_fname, time_taken)