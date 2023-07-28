""" RAD-sol-test.py

This code is a modified copy of the original RAD, but with comments starting line 78 to obtain solution first gradients, through a function defined in lines 22-23.
Made this copy here so can test higher gradients through hessian here too. The data is exported to txt and needs to be read with code in data_visualisation.
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
    return dde.grad.jacobian(y, x, i=0, j=0)
# usage: du_dx = model.predict(X,operator=du_dx)

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
    model.train()
    y_pred = model.predict(X_test)
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    error_hist = [l2_error]
    print(f"l2_relative_error: {l2_error}")

    # ---------------------------------------------------------------------- STUFF OF INTEREST STARTING HERE

    print("Now starts the resample loop")
    for i in range(5):
        X = geomtime.random_points(100000)

        # First evaluation Uncomment this to obtain output of gradients
        # du_dx = model.predict(X_test,operator=dudx)
        # output = np.hstack((X_test,du_dx))
        # np.savetxt(f"results/raw/sol-sampling-test/xtest-and-dudx.txt", output)

        # Original method. Uncomment to obtain output of residuals
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        print(X.shape)
        print(Y.shape)
        output2 = np.hstack((X,Y))
        np.savetxt(f"results/raw/sol-sampling-test/residualsy.txt",output2)
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        print(err_eq.shape)
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        print(err_eq_normalized.shape)
        X_ids = np.random.choice(
            a=len(X), size=NumDomain, replace=False, p=err_eq_normalized
        )
        print(X_ids.shape)
        X_selected = X[X_ids]
        data.replace_with_anchors(X_selected)

        quit()

        print("Adam going for 1000")
        model.compile("adam", lr=0.001)
        model.train(epochs=1000)
        print("L-BFGS going for up to 2000")
        model.compile("L-BFGS")
        losshistory, train_state = model.train()

        y_pred = model.predict(X_test)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        error_hist.append(l2_error)
        print("!\nFinished loop #{}\n".format(i))
        print(f"l2_relative_error: {l2_error}")

    error_hist = np.array(error_hist)
    dde.saveplot(losshistory, train_state, issave=False, isplot=False)
    # np.savetxt(f"error_RAD_k_{k}_c_{c}.txt", error)
    
    del model
    return error_hist, l2_error

# ---------------------------------------------------------------------- STUFF OF INTEREST ABOVE

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
    print(error_hists.shape)
    print(error_hists)
    np.savetxt(f"results/raw/sol-sampling-test/RAD_RAND_k1c1_N1000_L10_allerrors.txt", error_hists)
    print("Files saved")

