import deepxde as dde
import numpy as np
from deepxde.backend import tf
import time
import pdb

dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=1000)


def gen_testdata():
    data = np.load("src/burgers/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def main(k, c):
    NumDomain = 200
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    # dde.optimizers.config.set_LBFGS_options(maxiter=1000)# Not here in original

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
        num_domain=NumDomain // 20,
        train_distribution="pseudo",
    )

    net = dde.maps.FNN([2] + [64] * 3 + [1], "tanh", "Glorot normal")

    def output_transform(x, y):
        return -tf.sin(np.pi * x[:, 0:1]) + (1 - x[:, 0:1] ** 2) * (x[:, 1:]) * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    print("About to train adam for 15000")
    model.compile("adam", lr=0.001)
    model.train(epochs=1000)
    print("About to train L-BFGS - Max should be 1000")
    model.compile("L-BFGS")  # This seems to be taking more than 1000 even with maxiter in
    model.train()
    y_pred = model.predict(X_test)
    l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
    error = [np.array([l2_error])]
    print(f"l2_relative_error: {l2_error}")
    print("Now starts the resample loop")
    for i in range(10):
        X = geomtime.random_points(100000)
        Y = np.abs(model.predict(X, operator=pde)).astype(np.float64)
        err_eq = np.power(Y, k) / np.power(Y, k).mean() + c
        err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
        X_ids = np.random.choice(
            a=len(X), size=((NumDomain//200)*19), replace=False, p=err_eq_normalized
        )
        X_selected=X[X_ids]
        pdb.set_trace()
        data.add_anchors(X_selected) 

        print("Adam going for 1000")
        model.compile("adam", lr=0.001)
        #
        model.train(epochs=1000)
        print("L-BFGS going for 1000")
        model.compile("L-BFGS")
        losshistory, train_state = model.train()
        #Inspect Train_state and whether it updates appropriately

        y_pred = model.predict(X_test)
        l2_error = dde.metrics.l2_relative_error(y_true, y_pred)
        error.append(l2_error)
        print("!\nFinished loop #{}\n".format(i))
        print(f"l2_relative_error: {l2_error}")

    error = np.array(error)
    dde.saveplot(losshistory, train_state, issave=False, isplot=True)

    del model
    
    return error, l2_error


if __name__ == "__main__":
    time_cost = []
    error = []
    for n in range(1):
        start_t = time.time()
        _, error1 = main(c=20, k=0)
        error.append(error1)
        time_cost.append((time.time() - start_t))

        print(
            "\n--------------------------------------------\nFinished run #{}".format(n)
        )
        print("Time taken: {:.02f}s".format(time.time() - start_t))
    error = np.array(error)
    # np.savetxt(f"results/raw/error_and_time/time_RAR-D_default3.txt", time_cost)
    # np.savetxt(f"results/raw/error_and_time/error_RAR-D_default3.txt", error)