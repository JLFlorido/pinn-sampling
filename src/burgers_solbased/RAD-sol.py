"""Run PINN to solve Burger's Equation using adaptive resampling (RAD) based on gradient/curvature information.

This is the original version of the code, and is able to be looped. However, there is an issue with memory and running beyond 3 exponentially increases the cost.
It was used to test implementation of gradients/hessian and to obtain plots.
"""
import deepxde as dde
import numpy as np
from deepxde.backend import tf
import time

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=1000)


def gen_testdata():
    data = np.load("src/burgers/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

def dudx(x,y):
    return dde.grad.jacobian(y, x, i=0, j=0) # Known from example
# usage: du_dx = model.predict(X,operator=du_dx)
def dudt(x,y):
    return dde.grad.jacobian(y,x, i=0, j=1) # Known from example
def du_xt(x,y):
    return dde.grad.hessian(y,x,i=1,j=0) # Not known, inferred. Checked and i=0, j=1 is identical.
def du_tx(x,y):
    return dde.grad.hessian(y,x,i=0,j=1) # Not known, inferred. Checking, is identical to above
def du_xx(x,y):
    return dde.grad.hessian(y,x,i=0,j=0) # Known from example
def du_tt(x,y):
    return dde.grad.hessian(y,x,i=1,j=1) # Known from example
def main(k, c):
    NumDomain = 1000

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    dde.optimizers.config.set_LBFGS_options(maxiter=1000)

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    X_test, y_true = gen_testdata()

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

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

    def output_transform(x, y):
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    print("About to train adam for 5000") #15k originally
    model.compile("adam", lr=0.001)
    model.train(epochs=5000) #15k originally
    print("About to train L-BFGS - Max should be 1000")
    model.compile("L-BFGS")  # This seems to be taking more than 1000
    # model.train()
    losshistory, train_state = model.train() # To check burgers state pre-resample. This and line below. Delete later
    dde.saveplot(losshistory, train_state, issave=False, isplot=True) # See above
    y_pred = model.predict(X_test)
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    error_hist = [l2_error]
    print(f"l2_relative_error: {l2_error}")

    # ----------------------------------------------------------------------

    print("Now starts the resample loop")
    for i in range(5):
        X = geomtime.random_points(100000)

        # This commented out code obtained gradients and curvature and saved it to check.
        # du_dx = model.predict(X_test,operator=dudx)
        # du_dt = model.predict(X_test,operator=dudt)
        # hess_xx = model.predict(X_test,operator=du_xx)
        # hess_tt = model.predict(X_test,operator=du_tt)
        # hess_xt = model.predict(X_test,operator=du_xt)
        # hess_tx = model.predict(X_test,operator=du_tx)
        # output_dudx = np.hstack((X_test,du_dx))
        # output_dudt = np.hstack((X_test,du_dt))
        # output_hess_xx = np.hstack((X_test,hess_xx))
        # output_hess_tt = np.hstack((X_test,hess_tt))
        # output_hess_xt = np.hstack((X_test,hess_xt))
        # output_hess_tx = np.hstack((X_test,hess_tx))
        # np.savetxt(f"results/raw/sol-sampling-test/xtest-and-dudx.txt", output_dudx)
        # np.savetxt(f"results/raw/sol-sampling-test/xtest-and-dudt.txt", output_dudt)
        # np.savetxt(f"results/raw/sol-sampling-test/xtest-and-hess_xx.txt", output_hess_xx)
        # np.savetxt(f"results/raw/sol-sampling-test/xtest-and-hess_tt.txt", output_hess_tt)
        # np.savetxt(f"results/raw/sol-sampling-test/xtest-and-hess_xt.txt", output_hess_xt)
        # np.savetxt(f"results/raw/sol-sampling-test/xtest-and-hess_tx.txt", output_hess_tx)

        # Original method. This code is how new points were selected originally from residual information
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(
            a=len(X), size=NumDomain, replace=False, p=err_eq_normalized
        )
        X_selected = X[X_ids]
        np.savetxt(f"results/raw/sol-sampling-test/residuals.txt",X_selected)

        # # Using du_dx
        # Y = np.abs(model.predict(X, operator=dudx)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # np.savetxt(f"results/raw/sol-sampling-test/Points_dudx.txt",X_selected)
        # # Using du_dt
        # Y = np.abs(model.predict(X, operator=dudt)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # np.savetxt(f"results/raw/sol-sampling-test/Points_dudt.txt",X_selected)
        # # Using u_xx
        # Y = np.abs(model.predict(X, operator=du_xx)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # np.savetxt(f"results/raw/sol-sampling-test/Points_du_xx.txt",X_selected)
        # # Using u_tt
        # Y = np.abs(model.predict(X, operator=du_tt)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # np.savetxt(f"results/raw/sol-sampling-test/Points_du_tt.txt",X_selected)
        # # Using u_tx
        # Y = np.abs(model.predict(X, operator=du_tx)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # np.savetxt(f"results/raw/sol-sampling-test/Points_du_tx.txt",X_selected)
        # # Using u_xt
        # Y = np.abs(model.predict(X, operator=du_xt)).astype(np.float64)
        # err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        # err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        # X_ids = np.random.choice(a=len(X), size=NumDomain, replace=False, p=err_eq_normalized)
        # X_selected = X[X_ids]
        # np.savetxt(f"results/raw/sol-sampling-test/Points_du_xt.txt",X_selected)

        quit()
        print("Adam going for 1000")
        model.compile("adam", lr=0.001)
        
        print("L-BFGS going for up to 2000")
        model.compile("L-BFGS")
        losshistory, train_state = model.train()

        y_pred = model.predict(X_test)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        error_hist.append(l2_error)
        print("!\nFinished loop #{}\n".format(i))
        print(f"l2_relative_error: {l2_error}")

    error_hist = np.array(error_hist)
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)
    # np.savetxt(f"error_RAD_k_{k}_c_{c}.txt", error)
    
    del model
    return error_hist, l2_error

# ----------------------------------------------------------------------

if __name__ == "__main__":

    time_cost = []
    error_hists = []
    final_errors = []
    for n in range(1):
        start_t = time.time()                           # Start timer
        
        all_error, final_error_only = main(c=1, k=1)    # Run main code
        error_hists.append(all_error)                   # Append error history
        final_errors.append(final_error_only)           # Append final error
        time_cost.append((time.time() - start_t))       # Append time taken

        print("Finished run #{}".format(n+1))           # Print run number, time taken
        print("Time taken: {:.02f}s".format(time.time() - start_t))
        print("\n--------------------------------------------\n")
    # Save matrix with final time and final error of all runs
    np.savetxt(f"results/raw/sol-sampling-test/RAD_RAND_k1c1_N1000_L5_time.txt", time_cost)
    np.savetxt(f"results/raw/sol-sampling-test/RAD_RAND_k1c1_N1000_L5_error.txt", final_errors)
 
    # Print error history shape to check it's correct and then save to file and print that all files have been saved
    error_hists = np.array(error_hists)
    np.savetxt(f"results/raw/sol-sampling-test/RAD_RAND_k1c1_N1000_L10_allerrors.txt", error_hists)
    print("Files saved")