"""REP-Local_errors.py Obtaining error at collocation points, globally as well as loss.
"""
import os
import skopt
from distutils.version import LooseVersion
import deepxde as dde
import numpy as np
from deepxde.backend import tf
import time
from multiprocessing import Pool
from scipy.interpolate import RegularGridInterpolator

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main code start
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def jpinn(k=1, c=1, NumDomain=2000, NumResamples=100, method="Random"): # Main Code
    start_t = time.time()

    def pde(x, y): # Define Burgers PDE
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx
    
    def dudx(x,y): # Returns gradient in x
        return dde.grad.jacobian(y, x, i=0, j=0)
    def dudt(x,y): # Returns gradient in y
        return dde.grad.jacobian(y,x, i=0, j=1)
    def du_xt(x,y): # Returns curvature in xt
        return dde.grad.hessian(y,x,i=1,j=0)
    def du_tx(x,y): # Returns curvature in tx. Identical to above
        return dde.grad.hessian(y,x,i=0,j=1)
    def du_xx(x,y): # Returns curvature in xx
        return dde.grad.hessian(y,x,i=0,j=0)
    def du_tt(x,y): # Returns curvature in tt
        return dde.grad.hessian(y,x,i=1,j=1)

    X_test, y_true, itp = gen_testdata() # (25600,2) coordinates and corresponding (25600,1) values of u.

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

    # NN DENSITY HERE
    net = dde.maps.FNN([2] + [64] * 5 + [1], "tanh", "Glorot normal") # This defines the NN layers, their size and activation functions.

    def output_transform(x, y):
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y
    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    print("Initial 15000 Adam steps")
    model.compile("adam", lr=0.001)
    model.train(epochs=15000, display_every=300000)

    
    print("Initial L-BFGS steps")
    model.compile("L-BFGS")
    model.train(display_every=300000)

    # Measuring error after initial phase. This information is not used by network to train.
    y_pred_local = model.predict(data.train_x_all)
    y_pred_local = [x[0] for x in y_pred_local]
    y_pred = model.predict(X_test)

    local_points=data.train_x_all[:,[1, 0]]
    y_true_local = itp(local_points) # INTERPOLATOR NEEDS to be fed (t,x) not (x,t).

    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    l2_error_local = dde.metrics.l2_relative_error(y_true_local, y_pred_local)

    error_hist = [l2_error]
    error_hist_local = [l2_error_local]
    step_hist = [model.train_state.step]

    print("!\nFinished initial steps\n")
    print(f"l2_relative_error: {l2_error}")

    for i in range(NumResamples): # Resampling loop begins. 100 is default, first run took ~4 hours...
        X = geomtime.random_points(100000)

        # --- Below, all the different info sources for resampling. Comment out the ones you won't use ---
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64) # 1 Using residual
        # Y1 = np.abs(model.predict(X, operator=dudx)).astype(np.float64) # 2 Using du_dx
        # Y2 = np.abs(model.predict(X, operator=dudt)).astype(np.float64) # 3 Using du_dt
        # Y = np.abs(model.predict(X, operator=du_xx)).astype(np.float64) # 4 Using u_xx
        # Y = np.abs(model.predict(X, operator=du_tt)).astype(np.float64) # 5 Using u_tt
        # Y = np.abs(model.predict(X, operator=du_tx)).astype(np.float64) # 6 Using u_tx
        # Y = np.abs(model.predict(X, operator=du_xt)).astype(np.float64) # 7 Using u_xt 
        # Y=(Y1+Y2)
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        X_selected = X[X_ids]

        data.replace_with_anchors(X_selected) # Change current points with selected X points

        # print("1000 Adam Steps")
        model.compile("adam", lr=0.001)
        model.train(epochs=1000,display_every=300000)
        # print("LBFG-S Steps")
        model.compile("L-BFGS")
        losshistory, train_state = model.train(display_every=300000)

        y_pred = model.predict(X_test)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        local_points=data.train_x_all[:,[1, 0]]
        y_pred_local = model.predict(data.train_x_all) # model.predict need (x,t)
        y_pred_local = [x[0] for x in y_pred_local]
        y_true_local = itp(local_points) # Interpolator needs (t,x), hence use of local_points.
        l2_error_local = dde.metrics.l2_relative_error(y_true_local, y_pred_local)
        error_hist.append(l2_error)
        error_hist_local.append(l2_error_local)
        step_hist.append(model.train_state.step)
        print("Finished loop #{}".format(i+1))
        print(f"l2_relative_error: {l2_error}")

    error_final = l2_error
    time_taken = (time.time()-start_t)

    dde.saveplot(losshistory, train_state, issave=True, isplot=False, 
                 loss_fname=f"REP_errors2_5layers_res_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_loss_info.dat", 
                 train_fname=f"REP_errors2_5layers_res_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalpoints.dat", 
                 test_fname=f"REP_errors2_5layers_res_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_finalypred.dat",
                 output_dir="./results/raw/errors_losses")
    
    error_curves = np.column_stack(
        (
            np.array(step_hist),
            np.array(error_hist_local),
            np.array(error_hist),
        )
    )

    return error_curves, error_final, time_taken
# ----------------------------------------------------------------------
def main():
    # define variables
    c=1
    k=1
    NumDomain=2000
    NumResamples=100
    method="Random"
    time_taken_list = []
    error_curves_list = []
    error_final_list = []
    
    for repeat in range(1):  
        error_curves, error_final, time_taken = apply(jpinn, (c, k, NumDomain, NumResamples, method)) # Run main, record error history and final accuracy.
        time_taken_list.append(time_taken)
        error_final_list.append(error_final)
        print(error_curves_list)
        error_curves_list = np.vstack(error_curves_list,error_curves)
        print(error_curves.shape)
        print(error_curves_list)
        print("here")
        print(f"Repeat number {repeat+1} done, took {time_taken} seconds")

    # Define output directory and file names. Should come from doc opts further on.
    output_dir = "./results/raw/performance_results"  # Replace with your desired output directory path
    output_dir_2 = "./results/raw/errors_losses"
    error_curves_fname = f"REP_errors2_5layers_res_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_error_curves.txt"
    error_final_fname = f"REP_errors2_5layers_res_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_error_final.txt"
    time_taken_fname = f"REP_errors2_5layers_res_{method}_k{k}c{c}_N{NumDomain}_L{NumResamples}_time_taken.txt"
    
    # If results directory does not exist, create it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir_2):
        os.mkdir(output_dir_2)

    # Define the full file paths
    error_final_fname = os.path.join(output_dir, error_final_fname)
    time_taken_fname = os.path.join(output_dir, time_taken_fname)
    error_curves_fname = os.path.join(output_dir_2, error_curves_fname)
    
    # Save
    np.savetxt(error_curves_fname,error_curves_list, header="step, error_train, error_test")
    np.savetxt(error_final_fname,error_final_list)
    np.savetxt(time_taken_fname,time_taken_list)

if __name__ == "__main__":
    main()