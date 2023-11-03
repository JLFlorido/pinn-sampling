"""REP-Local.py Run PINN locally using apply() method, allowing to loop without memory issues. 
Solves Burger's Equation using adaptive resampling (RAD) based on either residuals or gradient/curvature information.
"""
import deepxde as dde
import numpy as np
from deepxde.backend import tf
import time
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

def jpinn(k=1, c=1, NumDomain=2000, NumResamples=100): # Main Code
    # NumDomain = 2000 # Number of collocation points
    start_t = time.time()
    def pde(x, y): # Define Burgers PDE
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx
    
    # def dudx(x,y): # Returns gradient in x
    #     return dde.grad.jacobian(y, x, i=0, j=0)
    # def dudt(x,y): # Returns gradient in y
    #     return dde.grad.jacobian(y,x, i=0, j=1)
    def du_xt(x,y): # Returns curvature in xt
        return dde.grad.hessian(y,x,i=1,j=0)
    # def du_tx(x,y): # Returns curvature in tx. Identical to above
    #     return dde.grad.hessian(y,x,i=0,j=1)
    # def du_xx(x,y): # Returns curvature in xx
    #     return dde.grad.hessian(y,x,i=0,j=0)
    # def du_tt(x,y): # Returns curvature in tt
    #     return dde.grad.hessian(y,x,i=1,j=1)

    X_test, y_true = gen_testdata() # (25600,2) coordinates and corresponding (25600,1) values of u.

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
    ) # This chunk of code describes the problem using dde structure

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal") # This defines the NN layers, their size and activation functions.

    def output_transform(x, y):
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    net.apply_output_transform(output_transform)

    # Initial Training before resampling
    model = dde.Model(data, net)
    print("Initial 15000 Adam steps")
    model.compile("adam", lr=0.001)
    model.train(epochs=15000)
    print("Initial L-BFGS steps")
    model.compile("L-BFGS")
    model.train()

    # Measuring error after initial phase. This information is not used by network to train.
    y_pred = model.predict(X_test)
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    error_hist = [l2_error]
    
    print("!\nFinished initial steps\n")
    print(f"l2_relative_error: {l2_error}")

    for i in range(NumResamples): # Resampling loop begins. 100 is default, first run took ~4 hours...
        X = geomtime.random_points(100000)

        # --- Below, all the different info sources for resampling. Comment out the ones you won't use --- 
        # Original method. This code is how new points were selected originally from residual information
        # Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # # Using du_dx
        # Y = np.abs(model.predict(X, operator=dudx)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # # Using du_dt
        # Y = np.abs(model.predict(X, operator=dudt)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # # Using u_xx
        # Y = np.abs(model.predict(X, operator=du_xx)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # # Using u_tt
        # Y = np.abs(model.predict(X, operator=du_tt)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # # Using u_tx
        # Y = np.abs(model.predict(X, operator=du_tx)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # Using u_xt
        Y = np.abs(model.predict(X, operator=du_xt)).astype(np.float64)
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        X_selected = X[X_ids]

        data.replace_with_anchors(X_selected) # Change current points with selected X points

        # print("1000 Adam Steps")
        model.compile("adam", lr=0.001)
        model.train(epochs=1000)
        # print("LBFG-S Steps")
        model.compile("L-BFGS")
        losshistory, train_state = model.train()

        y_pred = model.predict(X_test)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        error_hist.append(l2_error)
        print("Finished loop #{}".format(i+1))
        print(f"l2_relative_error: {l2_error}")

    error_final = l2_error
    error_hist = np.array(error_hist)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, 
                 loss_fname=f"RAD_RAND_k{k}c{c}_N{NumDomain}_L{NumResamples}_loss_info.dat", 
                 train_fname=f"RAD_RAND_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalpoints.dat", 
                 test_fname=f"RAD_RAND_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalypred.dat",
                 output_dir="./results/raw/additional_info")
    time_taken = (time.time()-start_t)
    return error_hist, error_final, time_taken
# ----------------------------------------------------------------------
def main():
    # define variables
    c=1
    k=1
    NumDomain=2000
    NumResamples=100
    time_taken_list = []
    error_hist_list = []
    error_final_list = []
    
    for repeat in range(10):  
        error_hist, error_final, time_taken = apply(jpinn, (c, k, NumDomain, NumResamples)) # Run main, record error history and final accuracy.
        time_taken_list.append(time_taken)
        error_final_list.append(error_final)
        error_hist_list.append(error_hist)
        print(f"number {repeat+1} done, took {time_taken} seconds")

    # Define output directory and file names. Should come from doc opts further on.
    output_dir = "./results/performance_results"  # Replace with your desired output directory path
    error_hist_fname = f"RAD_RAND_k{k}c{c}_N{NumDomain}_L{NumResamples}_error_hist.txt"
    error_final_fname = f"RAD_RAND_k{k}c{c}_N{NumDomain}_L{NumResamples}_error_final.txt"
    time_taken_fname = f"RAD_RAND_k{k}c{c}_N{NumDomain}_L{NumResamples}_time_taken.txt"
    
    # If results directory does not exist, create it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Define the full file paths
    error_hist_fname = os.path.join(output_dir, error_hist_fname)
    error_final_fname = os.path.join(output_dir, error_final_fname)
    time_taken_fname = os.path.join(output_dir, time_taken_fname)
    
    # Save
    np.savetxt(error_hist_fname,error_hist_list)
    np.savetxt(error_final_fname,error_final_list)
    np.savetxt(time_taken_fname,time_taken_list)

if __name__ == "__main__":
    main()
