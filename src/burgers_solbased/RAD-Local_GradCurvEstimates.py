"""RAD-Local_GradCurvEstimates.py
Based on ResVsError, but adapted to save the data used for plotting gradients.
Plots at 2k (Only a limited amount of training with Adam),
at 15+1k (After all initial training with Adam(15k) and LBFG-S (1k))
And at the end of the simulation (After 100 re-samples of 1k Adam + 1k LBFG-S)
"""
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

def jpinn(k=1, c=1, NumDomain=2000, NumResamples=100): # Main Code
    # NumDomain = 2000 # Number of collocation points
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
    model.train(epochs=5000)
    # Insert first grad&curv estimate and save here
    output_dir = "results/raw/grad_curvature_estimates"  # Replace with your desired output directory path
    # If results directory does not exist, create it
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    xcoords_fname = os.path.join(output_dir, "gc_coords.txt")
    np.savetxt(xcoords_fname, X_test)
    result_1_ux = np.abs(model.predict(X_test, operator=dudx)).astype(np.float64) # 1 Using du_dx
    result_1_ut = np.abs(model.predict(X_test, operator=dudt)).astype(np.float64) # 2 Using du_dt
    result_1_uxx = np.abs(model.predict(X_test, operator=du_xx)).astype(np.float64) # 3 Using u_xx
    result_1_utt = np.abs(model.predict(X_test, operator=du_tt)).astype(np.float64) # 4 Using u_tt
    result_1_uxt = np.abs(model.predict(X_test, operator=du_xt)).astype(np.float64) # 5 Using u_xt
    result_1_ux_fname = os.path.join(output_dir, "gc1_ux.txt")
    result_1_ut_fname = os.path.join(output_dir, "gc1_ut.txt")
    result_1_uxx_fname = os.path.join(output_dir, "gc1_uxx.txt")
    result_1_utt_fname = os.path.join(output_dir, "gc1_utt.txt")
    result_1_uxt_fname = os.path.join(output_dir, "gc1_uxt.txt")
    np.savetxt(result_1_ux_fname,result_1_ux)
    np.savetxt(result_1_ut_fname,result_1_ut)
    np.savetxt(result_1_uxx_fname,result_1_uxx)
    np.savetxt(result_1_utt_fname,result_1_utt)
    np.savetxt(result_1_uxt_fname,result_1_uxt)
    # ---------------------------------------------
    model.train(epochs=10000)
    print("Initial L-BFGS steps")
    model.compile("L-BFGS")
    model.train()
    # Insert second grad&curv estimate and save here
    # If results directory does not exist, create it
    result_2_ux = np.abs(model.predict(X_test, operator=dudx)).astype(np.float64) # 1 Using du_dx
    result_2_ut = np.abs(model.predict(X_test, operator=dudt)).astype(np.float64) # 2 Using du_dt
    result_2_uxx = np.abs(model.predict(X_test, operator=du_xx)).astype(np.float64) # 3 Using u_xx
    result_2_utt = np.abs(model.predict(X_test, operator=du_tt)).astype(np.float64) # 4 Using u_tt
    result_2_uxt = np.abs(model.predict(X_test, operator=du_xt)).astype(np.float64) # 5 Using u_xt
    result_2_ux_fname = os.path.join(output_dir, "gc2_ux.txt")
    result_2_ut_fname = os.path.join(output_dir, "gc2_ut.txt")
    result_2_uxx_fname = os.path.join(output_dir, "gc2_uxx.txt")
    result_2_utt_fname = os.path.join(output_dir, "gc2_utt.txt")
    result_2_uxt_fname = os.path.join(output_dir, "gc2_uxt.txt")
    np.savetxt(result_2_ux_fname,result_2_ux)
    np.savetxt(result_2_ut_fname,result_2_ut)
    np.savetxt(result_2_uxx_fname,result_2_uxx)
    np.savetxt(result_2_utt_fname,result_2_utt)
    np.savetxt(result_2_uxt_fname,result_2_uxt)
    # ---------------------------------------------
    # Measuring error after initial phase. This information is not used by network to train.
    y_pred = model.predict(X_test)
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    
    print("!\nFinished initial steps\n")
    print(f"l2_relative_error: {l2_error}")

    for i in range(NumResamples): # Resampling loop begins. 100 is default, first run took ~4 hours...
        X = geomtime.random_points(100000)

        # --- Below, all the different info sources for resampling. Comment out the ones you won't use --- 
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64) # 1 Using residual
        # Y = np.abs(model.predict(X, operator=dudx)).astype(np.float64) # 2 Using du_dx
        # Y = np.abs(model.predict(X, operator=dudt)).astype(np.float64) # 3 Using du_dt
        # Y = np.abs(model.predict(X, operator=du_xx)).astype(np.float64) # 4 Using u_xx
        # Y = np.abs(model.predict(X, operator=du_tt)).astype(np.float64) # 5 Using u_tt
        # Y = np.abs(model.predict(X, operator=du_tx)).astype(np.float64) # 6 Using u_tx
        # Y = np.abs(model.predict(X, operator=du_xt)).astype(np.float64) # 7 Using u_xt 
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
        model.train()

        y_pred = model.predict(X_test)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        print("Finished loop #{}".format(i+1))
        print(f"l2_relative_error: {l2_error}")

    # Insert last grad&curv estimate and save here
    result_3_ux = np.abs(model.predict(X_test, operator=dudx)).astype(np.float64) # 1 Using du_dx
    result_3_ut = np.abs(model.predict(X_test, operator=dudt)).astype(np.float64) # 2 Using du_dt
    result_3_uxx = np.abs(model.predict(X_test, operator=du_xx)).astype(np.float64) # 3 Using u_xx
    result_3_utt = np.abs(model.predict(X_test, operator=du_tt)).astype(np.float64) # 4 Using u_tt
    result_3_uxt = np.abs(model.predict(X_test, operator=du_xt)).astype(np.float64) # 5 Using u_xt
    result_3_ux_fname = os.path.join(output_dir, "gc3_ux.txt")
    result_3_ut_fname = os.path.join(output_dir, "gc3_ut.txt")
    result_3_uxx_fname = os.path.join(output_dir, "gc3_uxx.txt")
    result_3_utt_fname = os.path.join(output_dir, "gc3_utt.txt")
    result_3_uxt_fname = os.path.join(output_dir, "gc3_uxt.txt")
    np.savetxt(result_3_ux_fname,result_3_ux)
    np.savetxt(result_3_ut_fname,result_3_ut)
    np.savetxt(result_3_uxx_fname,result_3_uxx)
    np.savetxt(result_3_utt_fname,result_3_utt)
    np.savetxt(result_3_uxt_fname,result_3_uxt)
    # ---------------------------------------------
# ----------------------------------------------------------------------
def main():
    # define variables
    c=1
    k=1
    NumDomain=2000
    NumResamples=100
    
    for repeat in range(1):  
        apply(jpinn, (c, k, NumDomain, NumResamples)) # Run main, record error history and final accuracy.

if __name__ == "__main__":
    main()